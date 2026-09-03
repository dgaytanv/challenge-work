# monitoring.py
import math
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from embedding.dataloader import PFCandsDataset, PUPPIDataset
from embedding.utils.data_utils import delta_r_from_normalized
from embedding.utils.data_utils import EPS
from typing import Union

class EarlyStopping:
    """
    Simple early stopping on a monitored value (default: minimize 'loss').
    mode='min' or 'max'
    """
    def __init__(self, patience=20, mode="min", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False  # do not stop

        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience

def make_train_val_split(features, y, val_size=0.10, random_state=42, y_are_labels=True):
    """
    Stratified split of event tensors into train/val.
    features: [E, N, F] (normalized)
    y (labels):   [E]
    """
    idx = torch.arange(features.shape[0])
    idx_tr, idx_val = train_test_split(
        idx.cpu().numpy(),
        test_size=val_size,
        random_state=random_state,
        stratify=y.cpu().numpy() if y_are_labels else None
    )
    idx_tr = torch.tensor(idx_tr, dtype=torch.long, device=features.device)
    idx_val = torch.tensor(idx_val, dtype=torch.long, device=features.device)

    X_tr = features.index_select(0, idx_tr)
    y_tr = y.index_select(0, idx_tr)
    X_val = features.index_select(0, idx_val)
    y_val = y.index_select(0, idx_val)
    return X_tr, y_tr, X_val, y_val, idx_tr, idx_val

def build_train_val_loaders(
    X_tr, y_tr, X_val, y_val, device, batch_size=2048, pfcands=False
):
    ds_cls = PFCandsDataset if pfcands else PUPPIDataset
    ds_tr  = ds_cls(X_tr,  y_tr,  device)
    ds_val = ds_cls(X_val, y_val, device)
    train_loader = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

def _cls_mask(mask):
    """Prepend the always-attendable CLS slot to a [B, N] padding mask -> [B, N+1]."""
    return torch.cat([
        torch.zeros(mask.size(0), 1, device=mask.device, dtype=torch.bool),
        mask.bool()
    ], dim=1)

def consistency_terms(z_d, z_c):
    """Pull the degraded latent toward the clean latent (stop-grad on clean).

    The eval probe is fit on CLEAN latents and then applied to degraded ones, so this
    per-event invariance on the LATENT (pre-projector) is exactly what has to transfer.
    Returns (cosine_loss, mse_loss, mean_cosine_similarity).
    """
    z_c = z_c.detach()
    cos = F.cosine_similarity(z_d, z_c, dim=-1)
    return (1.0 - cos).mean(), F.mse_loss(z_d, z_c), cos.mean()

def train_epoch(
    encoder, projector, classifier,
    ce_loss_fn, contrastive_loss,
    train_loader, norm_constants, device,
    optimizer, preproc,
    degradation=None,
    scheduler=None, contrastive_weight=0.05,
    pairwise=False, num_classes=4,
    scaler=None,
    two_view=False, consistency_weight=1.0, consistency_mse_weight=0.1,
    instance_weight=0.0, instance_loss=None,
):
    if degradation is not None:
        degradation.train()
    encoder.train(); projector.train(); classifier.train(); preproc.train()

    total_loss = total_contrast = total_ce = 0.0
    total_cons = total_cons_mse = total_inst = total_cos = 0.0
    count = 0
    scheduled_contrst_wght = not (isinstance(contrastive_weight, int) or isinstance(contrastive_weight, float))
    class_metrics = ClassificationMetrics(num_classes)
    deg_metrics = ClassificationMetrics(num_classes) if two_view else None

    for x, mask, labels in train_loader:
        x = x.to(device)
        if not two_view and degradation is not None:
            x = degradation(x)
        mask = mask.to(device)
        labels = labels.to(device)

        if two_view:
            # Two views of the SAME events, concatenated on the batch dim so preproc's
            # BatchNorm and the encoder see both together in one forward.
            x_d = degradation(x) if degradation is not None else x
            x = torch.cat([x, x_d], dim=0)
            mask = torch.cat([mask, mask], dim=0)
            labels = torch.cat([labels, labels], dim=0)

        mask = _cls_mask(mask)

        delta_r = delta_r_from_normalized(x, norm_constants) if pairwise else None

        use_amp = (device == "cuda") and (scaler is not None) and scaler.is_enabled()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            latent = encoder(preproc(x), delta_r, mask)
            embeddings = F.normalize(projector(latent), dim=1)
            loss_constrast = contrastive_loss(embeddings, labels)
            logits = classifier(embeddings)
            loss_ce = ce_loss_fn(logits, labels)

            contrast_weight_value = contrastive_weight.get() if scheduled_contrst_wght else contrastive_weight
            loss = contrast_weight_value * loss_constrast + loss_ce

            if two_view:
                B = latent.size(0) // 2
                z_c, z_d = latent[:B], latent[B:]
                loss_cons, loss_cons_mse, cos_mean = consistency_terms(z_d, z_c)
                loss = loss + consistency_weight * loss_cons + consistency_mse_weight * loss_cons_mse
                if instance_weight > 0.0 and instance_loss is not None:
                    loss_inst = instance_loss(embeddings[:B], embeddings[B:])
                    loss = loss + instance_weight * loss_inst
                else:
                    loss_inst = torch.zeros((), device=latent.device)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if scheduled_contrst_wght:
            contrastive_weight.step()

        if torch.isnan(loss):
            print("NaN loss detected, skipping update")

        bs = x.size(0)
        total_loss    += loss.item() * bs
        total_contrast += loss_constrast.item() * bs
        total_ce       += loss_ce.item() * bs
        count          += bs
        class_metrics.update(logits, labels)
        if two_view:
            total_cons     += loss_cons.item() * bs
            total_cons_mse += loss_cons_mse.item() * bs
            total_inst     += loss_inst.item() * bs
            total_cos      += cos_mean.item() * bs
            deg_metrics.update(logits[B:], labels[B:])

    out = {
        "loss":     total_loss    / count,
        "contrast": total_contrast / count,
        "ce":       total_ce      / count,
        **class_metrics.compute_metrics()
    }
    if two_view:
        out.update({
            "cons":     total_cons     / count,
            "cons_mse": total_cons_mse / count,
            "inst":     total_inst     / count,
            "cos":      total_cos      / count,
            "acc_deg":  deg_metrics.compute_metrics()["acc"],
        })
    return out

@torch.no_grad()
def validate_epoch(
    encoder, projector, classifier,
    ce_loss_fn, contrastive_loss,
    val_loader, norm_constants, device, preproc,
    degradation=None,
    contrastive_weight=0.05,
    pairwise=False, num_classes=4,
    two_view=False, consistency_weight=1.0, consistency_mse_weight=0.1,
    instance_weight=0.0, instance_loss=None,
):
    if degradation is not None:
        degradation.eval()
    encoder.eval(); projector.eval(); classifier.eval(); preproc.eval()

    total_loss = total_contrast = total_ce = 0.0
    total_cons = total_cons_mse = total_inst = total_cos = 0.0
    count = 0
    scheduled_contrst_wght = not (isinstance(contrastive_weight, int) or isinstance(contrastive_weight, float))
    class_metrics = ClassificationMetrics(num_classes)
    deg_metrics = ClassificationMetrics(num_classes) if two_view else None

    for x, mask, labels in val_loader:
        x = x.to(device)
        if not two_view and degradation is not None:
            x = degradation(x)
        mask = mask.to(device)
        labels = labels.to(device)

        if two_view:
            x_d = degradation(x) if degradation is not None else x
            x = torch.cat([x, x_d], dim=0)
            mask = torch.cat([mask, mask], dim=0)
            labels = torch.cat([labels, labels], dim=0)

        mask = _cls_mask(mask)

        delta_r = delta_r_from_normalized(x, norm_constants) if pairwise else None

        latent = encoder(preproc(x), delta_r, mask)
        embeddings = F.normalize(projector(latent), dim=1)
        loss_constrast = contrastive_loss(embeddings, labels)
        logits = classifier(embeddings)
        loss_ce = ce_loss_fn(logits, labels)

        contrast_weight_value = contrastive_weight.get() if scheduled_contrst_wght else contrastive_weight
        loss = contrast_weight_value * loss_constrast + loss_ce

        if two_view:
            B = latent.size(0) // 2
            z_c, z_d = latent[:B], latent[B:]
            loss_cons, loss_cons_mse, cos_mean = consistency_terms(z_d, z_c)
            loss = loss + consistency_weight * loss_cons + consistency_mse_weight * loss_cons_mse
            if instance_weight > 0.0 and instance_loss is not None:
                loss_inst = instance_loss(embeddings[:B], embeddings[B:])
                loss = loss + instance_weight * loss_inst
            else:
                loss_inst = torch.zeros((), device=latent.device)

        bs = x.size(0)
        total_loss     += loss.item() * bs
        total_contrast += loss_constrast.item() * bs
        total_ce       += loss_ce.item() * bs
        count          += bs
        class_metrics.update(logits, labels)
        if two_view:
            total_cons     += loss_cons.item() * bs
            total_cons_mse += loss_cons_mse.item() * bs
            total_inst     += loss_inst.item() * bs
            total_cos      += cos_mean.item() * bs
            deg_metrics.update(logits[B:], labels[B:])

    out = {
        "loss":     total_loss     / count,
        "contrast": total_contrast / count,
        "ce":       total_ce       / count,
        **class_metrics.compute_metrics()
    }
    if two_view:
        out.update({
            "cons":     total_cons     / count,
            "cons_mse": total_cons_mse / count,
            "inst":     total_inst     / count,
            "cos":      total_cos      / count,
            "acc_deg":  deg_metrics.compute_metrics()["acc"],
        })
    return out

def cosine_schedule_with_warmup(
        optimizer: torch.optim.Optimizer, 
        warmup_steps: int, 
        total_steps: int,
        lr: float,
        lr_min: float = 0.0,
    ):
    """
    Cosine learning rate schedule with linear warmup.

    - LR starts at lr_min
    - warms up linearly to lr over warmup_steps
    - then decays with cosine back toward lr_min by total_steps
    """
    lr_delta = lr - lr_min
    def lr_lambda(step):
        if step < warmup_steps: # Linearly go from lr_min to lr_max. If no lr_max, then keep lr const at lr_min)
            return (1/lr) * (lr_min + lr_delta * (step / warmup_steps))
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return (1/lr) * (lr - lr_delta * 0.5 * (1 - math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

class cosine_constrastive_schedule:
    """
    Cosine schedule for the *contrastive weight* (not LR).

    - weight starts at weight_min
    - warms up linearly to weight_max over warmup_steps
    - then decays with cosine back toward weight_min by total_steps

    Stepped once per optimizer step (i.e., per batch).
    """
    def __init__(
        self,
        weight_min: float,
        weight_max: float,
        warmup_steps: int,
        total_steps: int
    ):
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._step = 0
        self.current_weight = self._compute(0)

    def _compute(self, step: int) -> float:
        """
        Max: weight_max at end of warmup (step==warmup_steps)
        Min: weight_min at step==total_steps
        0 <= step <= total_steps
        """

        # Clamp [0, total_steps]
        step = max(0, min(step, self.total_steps))

        if step < self.warmup_steps:
            # Linear warmup: min -> max
            t = step / self.warmup_steps
            return self.weight_min + t * (self.weight_max - self.weight_min)

        # Cosine decay: max -> min
        # progress in [0,1] after warmup
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        # cosine from 1 -> -1 maps to 1 -> 0 with 0.5*(1+cos)
        cos_term = 0.5 * (1 + math.cos(math.pi * progress)) # Max: 1, Min: 0
        return self.weight_min + cos_term * (self.weight_max - self.weight_min)

    def step(self) -> float:
        """
        Advance schedule by 1 step and update current_weight.
        Call once per batch (after optimizer.step()).
        """
        self._step += 1
        self.current_weight = self._compute(self._step)
        return self.current_weight

    def get(self) -> float:
        """Return current weight without advancing."""
        return self.current_weight

    def state_dict(self) -> dict:
        return {
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "_step": self._step,
            "current_weight": self.current_weight,
        }

    def load_state_dict(self, state: dict) -> None:
        self.weight_min = float(state["weight_min"])
        self.weight_max = float(state["weight_max"])
        self.warmup_steps = int(state["warmup_steps"])
        self.total_steps = int(state["total_steps"])
        self._step = int(state["_step"])
        self.current_weight = float(state["current_weight"])

class ClassificationMetrics:
    """
    Utility class to track and compute classification metrics (accuracy, precision, recall, F1).
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fn = torch.zeros(self.num_classes, dtype=torch.long)
        self.tn = torch.zeros(self.num_classes, dtype=torch.long)
        self.num_events = 0
        self.all_probs = []
        self.all_labels = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Update counts based on batch predictions and true labels.
        logits: [B, C] raw model outputs, where C = num_classes
        labels: [B] true class indices
        """

        self.num_events += labels.size(0)
        preds = logits.argmax(dim=1)  # [B]

        for cls in range(self.num_classes):
            cls_preds = (preds == cls) # [B] bool tensor where True means class predicted correctly as cls
            cls_labels = (labels == cls) # [B] bool tensor where True means true label is cls
            self.tp[cls] += (cls_preds & cls_labels).sum().item()
            self.fp[cls] += (cls_preds & ~cls_labels).sum().item()
            self.fn[cls] += (~cls_preds & cls_labels).sum().item()
            self.tn[cls] += (~cls_preds & ~cls_labels).sum().item()

        # .float() first: under autocast logits are fp16 and the softmax/roc_auc_score
        # round-trip yields nan for the train-line AUC (found by WP-D).
        self.all_probs.append(F.softmax(logits.float(), dim=1).detach().cpu())
        self.all_labels.append(labels.detach().cpu())

    def compute_metrics(self) -> dict:
        # TODO: Add more metrics

        accuracy = (self.tp.sum().item()) / self.num_events if self.num_events > 0 else 0.0

        probs = torch.cat(self.all_probs).numpy()
        labels = torch.cat(self.all_labels).numpy()
        try:
            if self.num_classes == 2:
                auc = roc_auc_score(labels, probs[:, 1])
            else:
                auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except ValueError:
            # e.g. not every class present yet (can happen on a tiny --test_mode run)
            auc = float("nan")

        return {
            "auc": auc,
            "acc": accuracy,
        }