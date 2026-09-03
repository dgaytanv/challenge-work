import torch
import torch.nn as nn
from typing import Union
from embedding.utils.data_utils import EPS

class PFPreProcessor(nn.Module):
    def __init__(self, norm_constants: dict = {}):
        super().__init__()
        self.norm_constants = norm_constants
        PDGIDs = [
            211, # h, charged hadrons
            11,  # e
            13,  # mu
            22,  # gamma
            130, # h0
            1,   # h_HF, HF tower identified as a hadron
            2    # egamma_HF, HF tower identified as an EM particle
        ]
        self.register_buffer("avail_pdgIds", torch.tensor(PDGIDs, dtype=torch.long))
        self.num_features_cont = 5 # pt, eta, phi, dxy, dxysig
        self.num_features_disc = 2 + len(self.avail_pdgIds) # is_pf, charge(from pdgId sign), pdgId(one-hot)
        self.num_features = self.num_features_cont + self.num_features_disc
        self.batch_norm = nn.BatchNorm1d(self.num_features_cont)

    def pdgId_to_onehot(self, pdgId_tensor: torch.Tensor) -> torch.Tensor:
        pdgId_tensor = pdgId_tensor.long()  # [B, N]
        one_hot = (pdgId_tensor.unsqueeze(-1).abs() == self.avail_pdgIds).float()  # [B, N, num_pdgIds]
        return one_hot
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, 8] = [pt, eta, phi, dxy, dxysig, is_pf, pdgId]
        Returns: [B, N, 8 + num_pdgIds] with normalized/scaled values and one-hot pdgId.
        - pt:        log(pt / sum_pt) per event
        - eta:       kept as is
        - phi:       kept as is
        - dxy:       tanh(dxy)
        - dxysig:    kept as is
        - charge:    derived from pdgId: +1/-1 for e, mu, pi; 0 else
        - is_pf:     kept as is (0/1)
        - pdgId:     one-hot encoded absolute pdgId from available list
        Padding (pt==0) rows are zeroed.
        Continuous features are batch-normalized.
        """

        pt_raw = x[..., 0]
        eta_raw = x[..., 1]
        phi_raw = x[..., 2]
        dxy_raw = x[..., 3]
        dxysig_raw = x[..., 4]
        is_pf_raw = x[..., 5]
        pdgId_raw = x[..., 6]

        valid = pt_raw > 0 # [B, N]
        
        if valid.any():
            pt = torch.where(valid, pt_raw, torch.zeros_like(pt_raw))
            pt = pt / (pt.sum(dim=-1, keepdim=True) + EPS)
            pt[valid] = torch.log(pt[valid]) 
            
            dxy = torch.where(valid, dxy_raw, torch.zeros_like(dxy_raw))
            dxy[valid] = torch.tanh(dxy_raw[valid])

            pos = (pdgId_raw == 11) | (pdgId_raw == 13) | (pdgId_raw == 211)
            neg = (pdgId_raw == -11) | (pdgId_raw == -13) | (pdgId_raw == -211)
            charge = torch.zeros_like(valid, dtype=torch.float)
            charge[valid & pos] = 1.0
            charge[valid & neg] = -1.0
            
            pdgId_onehot = self.pdgId_to_onehot(pdgId_raw)
            
            x_proc = torch.cat([
                pt.unsqueeze(-1),
                eta_raw.unsqueeze(-1),
                phi_raw.unsqueeze(-1),
                dxy.unsqueeze(-1),
                dxysig_raw.unsqueeze(-1),
                # Next ones NOT are not continuous, so NOT fed to batch norm layer.
                charge.unsqueeze(-1), # Computed
                is_pf_raw.unsqueeze(-1),
                pdgId_onehot
            ], dim=-1)

            # Masked batch norm on continuous ftrs
            x_cont = x_proc[..., :self.num_features_cont]  # [B, N, num_cont]
            B, N, C = x_cont.shape
            x_cont_flat = x_cont.reshape(B * N, C)
            valid_flat = valid.reshape(B * N)
            x_cont_flat[valid_flat] = self.batch_norm(x_cont_flat[valid_flat])
            x_proc[..., :self.num_features_cont] = x_cont_flat.reshape(B, N, C)
        else:
            x_proc = torch.zeros(*x.shape[:-1], self.num_features, device=x.device)
        
        return x_proc

class PUPPIPreProcessor(nn.Module):
    def __init__(self, norm_constants: dict):
        super().__init__()
        self.norm_constants = norm_constants
        self.num_features = 7

    def forward(self, x):
        """
        x: [P, 7] = [pt, eta, phi, dxy, btag, has_dxy, has_btag]
        Returns: [P, 7] with normalized/scaled values and flags kept as channels.
        - pt:   log1p + min–max (train-split stats)
        - eta:  min–max
        - phi:  wrapped to (-pi, pi] then scaled to [0,1]
        - dxy:  min–max where defined (has_dxy), else 0
        - btag: clamped to [0,1] where defined (has_btag), else 0
        - flags: kept as float channels
        Padding (pt==0) rows are zeroed.
        """

        pt_raw   = x[:, 0]
        eta_raw  = x[:, 1]
        phi_raw  = x[:, 2]
        dxy_raw  = x[:, 3]
        btag_raw = x[:, 4]
        has_dxy  = (x[:, 5] > 0.5)
        has_btag = (x[:, 6] > 0.5)

        valid = pt_raw > 0

        # pt: 
        pt = torch.zeros_like(pt_raw)
        if valid.any():
            pt_log = torch.log1p(pt_raw[valid])
            pt[valid] = (pt_log - self.norm_constants["pt_min"]) / (self.norm_constants["pt_max"] - self.norm_constants["pt_min"] + EPS)

        # eta: min–max on valid
        eta = torch.zeros_like(eta_raw)
        if valid.any():
            eta[valid] = (eta_raw[valid] - self.norm_constants["eta_min"]) / (self.norm_constants["eta_max"] - self.norm_constants["eta_min"] + EPS)

        # phi:
        phi = torch.zeros_like(phi_raw)
        if valid.any():
            phi[valid] = (phi_raw[valid] + torch.pi) / (2 * torch.pi)

        # dxy: min–max only where defined (valid & has_dxy)
        dxy = torch.zeros_like(dxy_raw)
        dv = valid & has_dxy
        if dv.any():
            dxy[dv] = (dxy_raw[dv] - self.norm_constants["dxy_min"]) / (self.norm_constants["dxy_max"] - self.norm_constants["dxy_min"] + EPS)

        # btag: (valid & has_btag)
        btag = torch.zeros_like(btag_raw)
        jb = valid & has_btag
        if jb.any():
            btag[jb] = btag_raw[jb]

        has_dxy_f  = has_dxy.float()
        has_btag_f = has_btag.float()
        has_dxy_f[~valid] = 0.0
        has_btag_f[~valid] = 0.0

        normed = torch.stack([pt, eta, phi, dxy, btag, has_dxy_f, has_btag_f], dim=-1)
        return normed

class _PFPreProcessorPtVariant(PFPreProcessor):
    """PFPreProcessor with a different pt normalisation. Everything else is identical.

    Why this exists (WP-B2). Stock ``PFPreProcessor`` encodes pt as
    ``log(pt_i / sum_pt)`` where ``sum_pt`` is summed over the *surviving* candidates
    of the event. A dead region removes candidates, so ``sum_pt`` falls and **every
    surviving candidate's** pt feature shifts by the same constant ``-log(f)``, where
    ``f`` is the surviving pt fraction. The whole event therefore moves in feature
    space even for candidates nowhere near the dead region. Measured on the stock
    checkpoint, roughly half of the latent displacement under degradation comes from
    this term alone (about 2.75 batch-norm units of shift at severity 0.8).

    Subclasses override :meth:`pt_feature` to choose a denominator that does not move
    when unrelated candidates disappear. ``num_features`` and the state dict layout are
    unchanged, so ``TransformerEncoder`` and the checkpoint format need no changes; the
    variant is selected purely through the ``preproc_type`` config key.
    """

    def pt_feature(self, pt_raw: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pt_raw = x[..., 0]
        eta_raw = x[..., 1]
        phi_raw = x[..., 2]
        dxy_raw = x[..., 3]
        dxysig_raw = x[..., 4]
        is_pf_raw = x[..., 5]
        pdgId_raw = x[..., 6]

        valid = pt_raw > 0  # [B, N]
        if not valid.any():
            return torch.zeros(*x.shape[:-1], self.num_features, device=x.device, dtype=x.dtype)

        zero = torch.zeros_like(pt_raw)
        pt = torch.where(valid, self.pt_feature(pt_raw, valid), zero)
        dxy = torch.where(valid, torch.tanh(dxy_raw), zero)

        pos = (pdgId_raw == 11) | (pdgId_raw == 13) | (pdgId_raw == 211)
        neg = (pdgId_raw == -11) | (pdgId_raw == -13) | (pdgId_raw == -211)
        charge = torch.where(valid & pos, torch.ones_like(zero), zero)
        charge = torch.where(valid & neg, -torch.ones_like(zero), charge)

        x_proc = torch.cat([
            pt.unsqueeze(-1),
            eta_raw.unsqueeze(-1),
            phi_raw.unsqueeze(-1),
            dxy.unsqueeze(-1),
            dxysig_raw.unsqueeze(-1),
            # Not continuous, so not fed to the batch norm layer.
            charge.unsqueeze(-1),
            is_pf_raw.unsqueeze(-1),
            self.pdgId_to_onehot(pdgId_raw),
        ], dim=-1)

        # Masked batch norm on the continuous features, exactly as PFPreProcessor does.
        x_cont = x_proc[..., :self.num_features_cont]
        B, N, C = x_cont.shape
        flat = x_cont.reshape(B * N, C).clone()
        valid_flat = valid.reshape(B * N)
        flat[valid_flat] = self.batch_norm(flat[valid_flat])
        x_proc = torch.cat([flat.reshape(B, N, C), x_proc[..., self.num_features_cont:]], dim=-1)

        # Keep zeroed candidates exactly zero: TransformerEncoder re-derives its
        # attention mask from all-zero rows.
        return torch.where(valid.unsqueeze(-1), x_proc, torch.zeros_like(x_proc))


class PFPreProcessorAbsPt(_PFPreProcessorPtVariant):
    """pt encoded as ``log(pt)``, with no per-event denominator at all.

    Fully local: a candidate's pt feature depends only on that candidate, so removing
    other candidates cannot move it. The batch norm supplies the scale. Trades away the
    "share of the event" information that the stock ratio carries.
    """

    def pt_feature(self, pt_raw, valid):
        return torch.log(pt_raw.clamp_min(EPS))


class PFPreProcessorMaxPt(_PFPreProcessorPtVariant):
    """pt encoded as ``log(pt_i / max_pt)``, the leading surviving candidate as reference.

    Keeps the relative-scale information of the stock ratio but uses an order statistic
    instead of a sum: the reference only moves if the *leading* candidate is itself
    dropped, rather than shifting a little for every candidate lost.
    """

    def pt_feature(self, pt_raw, valid):
        lead = torch.where(valid, pt_raw, torch.zeros_like(pt_raw)).amax(dim=-1, keepdim=True)
        return torch.log((pt_raw / lead.clamp_min(EPS)).clamp_min(EPS))


class PFPreProcessorMeanPt(_PFPreProcessorPtVariant):
    """pt encoded as ``log(pt_i / mean_pt)`` over the surviving candidates.

    Identically ``log(pt_i / sum_pt) + log(n_valid)``: the stock rule plus a count
    correction. This is the robust member of the family for a simple reason -- a dead
    region removes candidates and their pt together, so ``sum_pt`` and ``n_valid``
    shrink by roughly the same factor and their ratio barely moves, whereas ``sum_pt``
    alone scales directly with the loss.

    It also removes a train/eval mismatch that has nothing to do with degradation:
    training events carry 200 candidates and eval events 400, so ``sum_pt`` is about
    twice as large at eval time and the stock feature is offset by roughly log(2)
    before any dead region is applied. A per-candidate mean is insensitive to the
    number of candidates in the event.
    """

    def pt_feature(self, pt_raw, valid):
        ptv = torch.where(valid, pt_raw, torch.zeros_like(pt_raw))
        n = valid.sum(dim=-1, keepdim=True).clamp_min(1).to(pt_raw.dtype)
        mean = (ptv.sum(dim=-1, keepdim=True) / n).clamp_min(EPS)
        return torch.log((pt_raw / mean).clamp_min(EPS))
