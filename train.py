import torch
import torch.nn as nn
import datetime
import os
import logging
import argparse
import importlib
import embedding.models as models
from embedding.models import TransformerEncoder, Projector
from embedding.loss import InfoNCELoss, NTXentInstanceLoss, JSDLogitConsistency
from embedding.training import make_train_val_split, build_train_val_loaders, train_epoch, validate_epoch, EarlyStopping, cosine_schedule_with_warmup, cosine_constrastive_schedule
from embedding.utils.data_utils import compute_normalization_constants
from embedding.utils.cfg_handler import train_config, data_config
from embedding.utils.data_utils import compute_class_weights, load_data, set_safe_thread_count
from embedding.degradation import Degradation

set_safe_thread_count()
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("JEPA") 
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/training_{timestamp}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def main(data_path: str, cfg: train_config, cfg_data: data_config, test_mode: bool = False, outdir: str = "./checkpoints"):

    run_name = f"{cfg.get_model_name()}_{timestamp}"
    logger.info(f"Run name: {run_name}")

    num_epochs = cfg.hp("num_epochs", 400)
    patience = cfg.hp("early_stopping_patience", 100)
    val_split = cfg.get_trdata_cfg("val_split", 0.1)
    pairwise = cfg.get_trdata_cfg("pairwise", False)
    class_weights_setting = cfg_data.get("class_weights", None)
    pfcands = cfg_data.get("pfcands", True)
    preproc_type = cfg.get_trdata_cfg("preproc_type", "PFPreProcessor")
    mixed_prec = cfg.hp("mixed_prec", False)

    # Sweepable
    num_heads = cfg.hp("num_heads", 8)
    num_layers = cfg.hp("num_layers", 4)
    embed_size = cfg.hp("embed_size", 128)
    latent_dim = cfg.hp("latent_dim", 6)
    proj_dim = cfg.hp("proj_dim", 12)
    linear_dim = cfg.hp("linear_dim", None)
    encoder_class = cfg.hp("encoder_class", "TransformerEncoder")  # WP-D: pick the set encoder by name
    use_degradation = cfg.hp("use_degradation", True)              # WP-D: off for clean-baseline runs
    encoder_kwargs = cfg.hp("encoder_kwargs", {}) or {}            # WP-D: extra kwargs for the chosen encoder class
    # Detector eta acceptance for the degradation generator. Default 5.0 = offline PF, so every
    # existing config is unchanged; L1T data spans [-3, 3] and needs 3.0 or the generator would
    # place dead regions in eta bands the detector never fills, understating the real severity.
    degradation_eta_max = cfg.hp("degradation_eta_max", 5.0)
    contrast_temp = cfg.hp("contrast_temp", 0.07)
    contrastive_weight = cfg.hp("contrastive_weight", 0.05) # Min
    contrastive_max = cfg.hp("contrastive_max", None) # Max for schedule, None for fixed
    contrastive_warmup = cfg.hp("contrastive_warmup", 0.05)
    lr = cfg.hp("lr", 1e-3)
    lr_min = cfg.hp("lr_min", 0.0)
    lr_warmup = cfg.hp("lr_warmup", 0.05)
    batch_size = cfg.hp("batch_size", 256)

    # WP-C two-view consistency training. two_view=False reproduces stock behaviour exactly.
    two_view = cfg.hp("two_view", False)
    consistency_weight = cfg.hp("consistency_weight", 1.0)
    consistency_mse_weight = cfg.hp("consistency_mse_weight", 0.1)
    instance_weight = cfg.hp("instance_weight", 0.0)
    normalize_mse = cfg.hp("consistency_mse_normalized", True)
    center_cos = cfg.hp("consistency_cos_centered", True)
    val_bn_batch_stats = cfg.hp("val_bn_batch_stats", True)
    logit_consistency_weight = cfg.hp("logit_consistency_weight", 0.0)
    seed = cfg.hp("seed", None)
    if seed is not None:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        logger.info(f"Seeded RNG with {seed}")
    logger.info(
        f"two_view={two_view} consistency_weight={consistency_weight} "
        f"consistency_mse_weight={consistency_mse_weight} instance_weight={instance_weight} "
        f"consistency_mse_normalized={normalize_mse} consistency_cos_centered={center_cos} logit_consistency_weight={logit_consistency_weight} val_bn_batch_stats={val_bn_batch_stats}"
    )

    logger.info("Scaler for mixed precision training: {}".format(mixed_prec))
    scaler = torch.cuda.amp.GradScaler(enabled=((device=="cuda") and mixed_prec))

    # Load and split
    feature_block, label_block = load_data(
        data_path,
        map_location="cpu", # load on CPU and move to GPU in DataLoader to avoid GPU memory issues
        max_events=-1,
    )
    if test_mode:
        # cfg_data's nevents_per_class is -1 ("keep every available event") in every config we
        # have, so it can't be used to size the test split - use 10% of what actually loaded.
        num_events = int(0.10 * feature_block.shape[0])
        feature_block = feature_block[:num_events]
        label_block = label_block[:num_events]
        logger.info("Test mode enabled: using only 10% of the data for training and validation.")
        logger.info(f"Number of events: {num_events} (10% of total)")

    num_pf_objects = feature_block.shape[1]
    X_tr, y_tr, X_val, y_val, idx_tr, idx_val = make_train_val_split(feature_block, label_block, val_size=val_split)
    del feature_block  # X_tr/X_val are independent copies (index_select); the original full-size
    # tensor is otherwise unused for the rest of training and was being held in memory for no reason.
    assert X_tr.device.type == "cpu"
    assert y_tr.device.type == "cpu"

    # Log num classes
    num_classes = int(label_block.max().item()) + 1
    class_count_tr = torch.bincount(y_tr, minlength=num_classes)
    logger.info("Class counts in training set:")
    for i in range(num_classes):
        logger.info(f"  Class {i}: {class_count_tr[i].item()} events")
    class_count_val = torch.bincount(y_val, minlength=num_classes)
    logger.info("Class counts in validation set:")
    for i in range(num_classes):
        logger.info(f"  Class {i}: {class_count_val[i].item()} events")

    # Build loaders (use train stats for both)
    norm_constants = compute_normalization_constants(X_tr) if not pfcands else {}
    train_loader, val_loader = build_train_val_loaders(
        X_tr, y_tr, X_val, y_val, device=device, batch_size=batch_size, pfcands=pfcands
    )

    # Two modules by planner ruling: symmetries are shared by both views, dead regions
    # are what distinguishes the degraded view (and carry the curriculum counter).
    symmetry = Degradation(
        severity=None, s_max=0.0, p_clean=1.0, curriculum=False,
        p_charged_only=0.0, p_neutral_only=0.0, p_pt_scale=0.0,
        eta_max=degradation_eta_max,
    ).to(device).train() if two_view else None
    # WP-D: use_degradation=false is the clean-baseline arm (d-deepsets-clean); it keeps
    # the dead-region module out entirely. two_view needs it, so the two are exclusive.
    if two_view and not use_degradation:
        raise ValueError("two_view requires use_degradation: the degraded view needs dead regions")
    degradation = Degradation(
        severity=None, rotate_phi=not two_view, reflect_eta=not two_view,
        eta_max=degradation_eta_max,
    ).to(device).train() if use_degradation else None

    preproc_class = getattr(importlib.import_module("embedding.preprocs"), preproc_type)

    preproc = preproc_class(norm_constants).to(device).train()
    logger.info(f"Encoder class: {encoder_class} | training degradation: {use_degradation} | encoder_kwargs: {encoder_kwargs} | degradation_eta_max: {degradation_eta_max}")
    encoder = getattr(models, encoder_class)(
        num_features=preproc.num_features,
        embed_size=embed_size, 
        latent_dim=latent_dim, 
        num_heads=num_heads,
        num_layers=num_layers,
        linear_dim=linear_dim, 
        num_tokens=num_pf_objects if linear_dim is not None else None,
        pairwise=pairwise,
        **encoder_kwargs,
    ).to(device).train()
    projector = Projector(latent_dim, proj_dim, hidden_dim=(proj_dim*4)).to(device).train()
    classifier = nn.Linear(proj_dim, num_classes).to(device).train()

    class_weights = compute_class_weights(label_block, setting=class_weights_setting).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    criterion = InfoNCELoss(temperature=contrast_temp)
    instance_criterion = NTXentInstanceLoss(temperature=contrast_temp)
    jsd_criterion = JSDLogitConsistency()

    optimizer = torch.optim.Adam(
        list(preproc.parameters()) +
        list(encoder.parameters()) + 
        list(projector.parameters()) + 
        list(classifier.parameters()),
        lr=lr
    )

    # Scheduler based on TRAIN steps
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(lr_warmup * total_steps)
    scheduler = cosine_schedule_with_warmup(
        optimizer, 
        warmup_steps, 
        total_steps,
        lr=lr, # max present => scheduled, otherwise use lr_max as fixed min
        lr_min=lr_min
    )

    # Scheduler for contrastive weight
    if contrastive_max is not None:
        contrastive_warmup_steps = int(contrastive_warmup * total_steps)
        contrastive_schedule = cosine_constrastive_schedule(
            weight_min = contrastive_weight,
            weight_max = contrastive_max,
            warmup_steps = contrastive_warmup_steps,
            total_steps = total_steps
        )

    best_val = float("inf")
    es = EarlyStopping(patience=patience, mode="min", min_delta=0.0)

    model_path = os.path.join(outdir, f"{cfg.get_model_name()}_encoder_{timestamp}.pth")
    # Two extra selections alongside the best-val-loss file, which keeps its name and format.
    # Rationale: total val loss mixes CE, contrast and the consistency terms, so the file it
    # picks is "best composite loss", not best AUC and not best robustness.
    # Auxiliary selections live in a SUBDIRECTORY, never beside the primary file: any
    # glob over the checkpoint dir would otherwise pick them up. D's bench used
    # `ls -t ... | head -1` and the organisers' notebook uses
    # sorted(glob("checkpoints/*.pth"))[-1], where "_last.pth" sorts after ".pth" --
    # both would silently evaluate the last epoch instead of best-val-loss.
    aux_dir = os.path.join(outdir, "aux")
    os.makedirs(aux_dir, exist_ok=True)
    aux_base = os.path.basename(model_path)
    bestauc_path = os.path.join(aux_dir, aux_base.replace(".pth", "_bestauc.pth"))
    last_path = os.path.join(aux_dir, aux_base.replace(".pth", "_last.pth"))
    best_val_auc = float("-inf")

    logger.info(f"Starting training for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        tr = train_epoch(
            encoder, 
            projector, 
            classifier,
            ce_loss_fn, 
            criterion,
            train_loader, 
            norm_constants, 
            device,
            optimizer,
            preproc,
            degradation=degradation,
            symmetry=symmetry,
            scheduler=scheduler, 
            contrastive_weight=contrastive_weight if contrastive_max is None else contrastive_schedule,
            pairwise=pairwise, 
            num_classes=num_classes,
            scaler=scaler,
            two_view=two_view,
            consistency_weight=consistency_weight,
            consistency_mse_weight=consistency_mse_weight,
            instance_weight=instance_weight,
            instance_loss=instance_criterion,
            normalize_mse=normalize_mse,
            center_cos=center_cos,
            logit_consistency_weight=logit_consistency_weight,
            logit_consistency_loss=jsd_criterion,
        )
        va = validate_epoch(
            encoder, 
            projector, 
            classifier,
            ce_loss_fn, 
            criterion,
            val_loader, 
            norm_constants, 
            device,
            preproc,
            degradation=degradation,
            symmetry=symmetry,
            contrastive_weight=contrastive_weight if contrastive_max is None else contrastive_schedule,
            pairwise=pairwise, 
            num_classes=num_classes,
            two_view=two_view,
            consistency_weight=consistency_weight,
            consistency_mse_weight=consistency_mse_weight,
            instance_weight=instance_weight,
            instance_loss=instance_criterion,
            normalize_mse=normalize_mse,
            center_cos=center_cos,
            logit_consistency_weight=logit_consistency_weight,
            val_bn_batch_stats=val_bn_batch_stats,
            logit_consistency_loss=jsd_criterion,
        )

        log_str = (
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train: Loss {tr['loss']:.6f}, Contrast {tr['contrast']:.6f}, CrossEntropy {tr['ce']:.6f}, Acc {tr['acc']:.4f}, AUC {tr['auc']:.4f} | "
            f"Val: Loss {va['loss']:.6f}, Contrast {va['contrast']:.6f}, CrossEntropy {va['ce']:.6f}, Acc {va['acc']:.4f}, AUC {va['auc']:.4f}"
        )
        if two_view:
            log_str += (
                f" | TwoView: cons {tr['cons']:.6f}, mse {tr['cons_mse']:.6f}, inst {tr['inst']:.6f}, "
                f"cos_tr {tr['cos']:.4f}, acc_deg_tr {tr['acc_deg']:.4f} | "
                f"cos_val {va['cos']:.4f}, acc_deg_val {va['acc_deg']:.4f} | "
                f"cos_shuf_tr {tr['cos_shuf']:.4f}, cos_shuf_val {va['cos_shuf']:.4f} | "
                f"pop_drift_tr {tr['pop_drift']:.4f}, pop_drift_val {va['pop_drift']:.4f}, "
                f"drift_spread_tr {tr['drift_spread']:.4f}, drift_spread_val {va['drift_spread']:.4f} | "
                f"jsd_tr {tr['jsd']:.6f}, jsd_val {va['jsd']:.6f} | "
                f"spread_tr {tr['lat_spread']:.3f}, offset_tr {tr['lat_offset']:.3f}, "
                f"spread_val {va['lat_spread']:.3f}, offset_val {va['lat_offset']:.3f}"
            )
        logger.info(log_str)

        def _checkpoint():
            return {
                "preproc": preproc.state_dict(),
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "norm_constants": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in norm_constants.items()},
            }

        # save best on validation loss 
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(_checkpoint(), model_path)
            logger.info(f"Saved best encoder to: {model_path}")

        # best val AUC: val loss mixes several terms, so it does not always pick the best latent
        val_auc = va.get("auc", float("nan"))
        if val_auc == val_auc and val_auc > best_val_auc:   # nan-safe
            best_val_auc = val_auc
            torch.save(_checkpoint(), bestauc_path)
            logger.info(f"Saved best-AUC encoder ({val_auc:.4f}) to: {bestauc_path}")

        # last completed epoch, rewritten every epoch so it survives an early kill
        torch.save(_checkpoint(), last_path)

        if es.step(va["loss"]):
            logger.info("Early stopping triggered.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True, help="Path to the data config .yaml file") 
    parser.add_argument("--train_cfg", required=True, help="Path to the training config .yaml file")
    parser.add_argument("--data", required=True, help="Path to the input .pt file")
    parser.add_argument("--outdir", default="./checkpoints", help="Directory to save model checkpoints and logs")
    parser.add_argument("--test_mode", action="store_true", help="If set, runs training with only 10 percent of the data.")
    args = parser.parse_args()

    tr_cfg = train_config(args.train_cfg)
    data_cfg = data_config(args.data_cfg)
    
    logger.info(f"Using train config file: {args.train_cfg}")
    logger.info(f"Entire train config: {tr_cfg.get_entire_cfg()}")
    
    logger.info(f"Using data processing config file: {args.data_cfg}")
    logger.info(f"Entire data processing config: {data_cfg.get_entire_cfg()}")

    main(args.data, tr_cfg, data_cfg, test_mode=args.test_mode, outdir=args.outdir)