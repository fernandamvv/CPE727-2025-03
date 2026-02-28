from __future__ import annotations

from pathlib import Path
import sys
import warnings
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Subset
import numpy as np
from sklearn.preprocessing import RobustScaler
import pandas as pd

from .ode_jump import ODEJump, TS_SPAN
from .tsdiffusion import TSDiffusion, cosine_beta_schedule

_SEQKAN_ARTIGO_PATH = Path("/home/ferna/seqKAN_artigo")

if _SEQKAN_ARTIGO_PATH.exists() and str(_SEQKAN_ARTIGO_PATH) not in sys.path:
    sys.path.insert(0, str(_SEQKAN_ARTIGO_PATH))

try:
    from seqkan.kan import KAN as ArticleKAN  # type: ignore
except Exception as exc:
    raise ImportError(
        "Não foi possível importar o seqKAN_artigo. "
        "Verifique se /home/ferna/seqKAN_artigo está disponível e importável."
    ) from exc


def _build_kan(width, params, device):
    grid = params.get("grid", 20)
    k = params.get("k", 3)
    grid_range = params.get("grid_range", (-10, 10))
    grid_eps = params.get("grid_eps", 0.02)
    sparse_init = params.get("sparse_init", False)
    return ArticleKAN(
        width=width,
        grid=grid,
        k=k,
        grid_eps=grid_eps,
        grid_range=list(grid_range),
        seed=params.get("seed", 42),
        sp_trainable=params.get("sp_trainable", False),
        sb_trainable=params.get("sb_trainable", False),
        affine_trainable=params.get("affine_trainable", False),
        symbolic_enabled=params.get("symbolic_enabled", False),
        sparse_init=sparse_init,
        auto_save=False,
        save_act=False,
        device=str(device),
    )


class SeqKANSeq(nn.Module):
    """
    Sequential KAN cell: h_t = KAN_cell([x_t, h_{t-1}]), y_hat = KAN_out(h_t).
    Supports feature-wise top-k on x, h and output heads.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        kan_params=None,
        device=None,
        concat_mask: bool = False,
    ):
        super().__init__()
        if kan_params is None:
            kan_params = {}
        self.kan_params = kan_params
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.concat_mask = bool(concat_mask)

        cell_params = kan_params.get("cell", kan_params.get("hidden", {}))
        out_params = kan_params.get("output", {})

        self.kan_cell = _build_kan(
            width=[self.input_size + self.hidden_size, self.hidden_size],
            params=cell_params,
            device=self.device,
        )
        self.kan_out = _build_kan(
            width=[self.hidden_size, self.output_size],
            params=out_params,
            device=self.device,
        )

        topk_cfg = kan_params.get("topk", {}) if isinstance(kan_params, dict) else {}
        self.topk_enabled = bool(topk_cfg.get("enabled", False))
        self.topk_warmup_epochs = int(topk_cfg.get("warmup_epochs", 0) or 0)
        self.topk_kind = str(topk_cfg.get("kind", "activation")).lower()
        self.topk_structural = self.topk_kind in ("structural", "connection", "edge", "edges")
        self.topk_structural_mode = str(topk_cfg.get("conn_mode", "per_out")).lower()
        self.topk_structural_score = str(topk_cfg.get("score", "coef_l2")).lower()
        self.topk_kx = topk_cfg.get("k_x", None)
        self.topk_kh = topk_cfg.get("k_h", None)
        self.topk_kout = topk_cfg.get("k_out", None)
        self.topk_kcell = topk_cfg.get("k_cell", None)
        self.topk_km = topk_cfg.get("k_m", None)
        self.topk_mask_size = topk_cfg.get("mask_size", None)
        self.last_topk_ratio = {}
        self.current_epoch = None
        self._last_structural_epoch = None

    def _edge_scores(self, layer):
        if hasattr(layer, "coef") and layer.coef is not None:
            coef = layer.coef
            if self.topk_structural_score in ("coef_l1", "coef_abs"):
                return coef.abs().mean(dim=-1)
            if self.topk_structural_score in ("coef_l2", "coef_norm"):
                return torch.sqrt((coef ** 2).mean(dim=-1) + 1e-12)
        if hasattr(layer, "scale_sp") and layer.scale_sp is not None:
            return layer.scale_sp.abs()
        raise ValueError("SeqKANSeq: cannot compute structural topk scores for layer")

    def _apply_structural_topk(self, kan, k, name):
        if k is None:
            return
        k = int(k)
        if k <= 0:
            return
        ratios = []
        for li, layer in enumerate(getattr(kan, "act_fun", [])):
            base_mask = layer.mask.detach()
            scores = self._edge_scores(layer).detach()
            if scores.shape != base_mask.shape:
                if scores.T.shape == base_mask.shape:
                    scores = scores.T
                else:
                    raise ValueError(
                        f"SeqKANSeq: structural topk score shape {tuple(scores.shape)} "
                        f"does not match mask shape {tuple(base_mask.shape)}"
                    )
            scores = scores * (base_mask > 0).float()
            in_dim, out_dim = scores.shape
            if name == "cell" and self.topk_kx is not None and self.topk_kh is not None:
                x_rows = self.input_size
                h_rows = self.hidden_size
                if in_dim != x_rows + h_rows:
                    raise ValueError(
                        f"SeqKANSeq: expected cell in_dim={x_rows + h_rows}, got {in_dim}"
                    )
                new_mask = torch.zeros_like(scores)
                kkx = min(int(self.topk_kx), x_rows)
                kkh = min(int(self.topk_kh), h_rows)
                if self.topk_km is not None:
                    if self.topk_kx is None:
                        raise ValueError("SeqKANSeq: k_m requires k_x to be set.")
                    if self.topk_mask_size is not None:
                        m_rows = int(self.topk_mask_size)
                    else:
                        if x_rows % 2 != 0:
                            raise ValueError(
                                "SeqKANSeq: cannot infer mask_size (input_size not even). "
                                "Set topk.mask_size explicitly."
                            )
                        m_rows = x_rows // 2
                    if m_rows <= 0 or m_rows >= x_rows:
                        raise ValueError(
                            f"SeqKANSeq: invalid mask_size={m_rows} for input_size={x_rows}"
                        )
                    x_feat_rows = x_rows - m_rows
                    kkm = min(int(self.topk_km), m_rows)
                    for j in range(out_dim):
                        col = scores[:, j]
                        x_col = col[:x_feat_rows]
                        m_col = col[x_feat_rows:x_rows]
                        h_col = col[x_rows:]
                        if kkx > 0:
                            _, idx_x = torch.topk(x_col, k=min(kkx, x_feat_rows), dim=0)
                            new_mask[idx_x, j] = 1.0
                        if kkm > 0:
                            _, idx_m = torch.topk(m_col, k=kkm, dim=0)
                            new_mask[x_feat_rows + idx_m, j] = 1.0
                        if kkh > 0:
                            _, idx_h = torch.topk(h_col, k=kkh, dim=0)
                            new_mask[x_rows + idx_h, j] = 1.0
                else:
                    for j in range(out_dim):
                        col = scores[:, j]
                        x_col = col[:x_rows]
                        h_col = col[x_rows:]
                        if kkx > 0:
                            _, idx_x = torch.topk(x_col, k=kkx, dim=0)
                            new_mask[idx_x, j] = 1.0
                        if kkh > 0:
                            _, idx_h = torch.topk(h_col, k=kkh, dim=0)
                            new_mask[x_rows + idx_h, j] = 1.0
            elif self.topk_structural_mode == "global":
                total = scores.numel()
                kk = min(k, total)
                flat_scores = scores.reshape(-1)
                _, idx = torch.topk(flat_scores, k=kk, dim=0)
                new_mask = torch.zeros_like(flat_scores)
                new_mask.scatter_(0, idx, 1.0)
                new_mask = new_mask.reshape_as(scores)
            else:
                kk = min(k, in_dim)
                new_mask = torch.zeros_like(scores)
                for j in range(out_dim):
                    col = scores[:, j]
                    if kk >= in_dim:
                        new_mask[:, j] = (col != 0).float()
                        continue
                    _, idx = torch.topk(col, k=kk, dim=0)
                    new_mask[idx, j] = 1.0
            layer.mask.data = new_mask * (base_mask > 0).float()
            ratios.append(float((layer.mask > 0).float().mean().item()))
        if ratios:
            self.last_topk_ratio[name] = float(sum(ratios) / len(ratios))
        else:
            self.last_topk_ratio[name] = 0.0

    def _maybe_apply_structural_topk(self):
        if not self.topk_enabled or not self.topk_structural:
            return
        if self.current_epoch is None or self.current_epoch <= self.topk_warmup_epochs:
            return
        if self._last_structural_epoch == self.current_epoch:
            return
        with torch.no_grad():
            self._apply_structural_topk(self.kan_cell, self.topk_kcell, "cell")
            if self.kan_out is not None:
                self._apply_structural_topk(self.kan_out, self.topk_kout, "out")
        self._last_structural_epoch = self.current_epoch

    def forward(self, x, return_last=False):
        if next(self.parameters()).device != x.device:
            self.to(x.device)
        self._maybe_apply_structural_topk()
        B, T, C = x.shape
        if C != self.input_size:
            raise ValueError("SeqKANSeq: input_size mismatch")
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            h_prev = h
            h_in = torch.cat([x_t, h_prev], dim=-1)
            h = self.kan_cell(h_in)
            y_t = self.kan_out(h)
            outputs.append(y_t)
        outputs = torch.stack(outputs, dim=1)
        if return_last:
            return outputs, h
        return outputs


class SeqKANCore(nn.Module):
    def __init__(self, hidden_dim, input_dim=None, output_dim=None, kan_params=None, variational_dropout=0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        input_dim = self.hidden_dim if input_dim is None else int(input_dim)
        output_dim = self.hidden_dim if output_dim is None else int(output_dim)
        self.variational_dropout = float(max(0.0, variational_dropout))
        self.seqkan = SeqKANSeq(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            output_size=output_dim,
            kan_params=kan_params,
        )

    def _apply_variational_dropout(self, x):
        if not self.training or self.variational_dropout <= 0:
            return x
        B, _, C = x.shape
        mask = x.new_ones(B, C)
        mask = F.dropout(mask, p=self.variational_dropout, training=True)
        return x * mask.unsqueeze(1)

    def forward(self, x, return_last=False):
        x = self._apply_variational_dropout(x)
        return self.seqkan(x, mask=None, return_last=return_last)


class SeqKANEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        kan_params=None,
        variational_dropout=0.0,
        use_layernorm: bool = True,
        mask_as_input: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.variational_dropout = float(max(0.0, variational_dropout))
        self.use_layernorm = use_layernorm
        self.mask_as_input = bool(mask_as_input)
        in_dim_eff = self.in_dim * 2 if self.mask_as_input else self.in_dim
        self.seqkan = SeqKANSeq(
            input_size=in_dim_eff,
            hidden_size=self.hidden_dim,
            output_size=self.hidden_dim,
            kan_params=kan_params,
        )
        self.norm_x = nn.LayerNorm(in_dim_eff) if use_layernorm else None
        self.norm_H = nn.LayerNorm(self.hidden_dim) if use_layernorm else None
        self.encoder = None

    def _apply_variational_dropout(self, x):
        if not self.training or self.variational_dropout <= 0:
            return x
        B, _, C = x.shape
        mask = x.new_ones(B, C)
        mask = F.dropout(mask, p=self.variational_dropout, training=True)
        return x * mask.unsqueeze(1)

    def forward(self, x, ts=None, only_gru=False, mask=None):  # noqa: ARG002
        x = self._apply_variational_dropout(x)
        if self.mask_as_input:
            if mask is None:
                mask = torch.ones_like(x)
            x = torch.cat([x, mask], dim=-1)
        if self.norm_x is not None:
            x = self.norm_x(x)
        outputs = self.seqkan(x, mask=None, return_last=False)
        if self.norm_H is not None:
            outputs = self.norm_H(outputs)
        return outputs



class TSDF_seqKANSeq(TSDiffusion):
    """
    TSDiffusion com backbone SeqKANSeq (difusao no latente, mesmo esquema do TSDiffusion).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 24,
        static_dim: int = 0,
        status_dim: int = 0,
        lam: list[float, float, float, float, float, float] = [0.9, 0.0, 0.0, 0.1, 0.0, 0.0],
        num_steps: int = 1000,
        cost_columns: list = None,
        bi_gru: bool = False,
        bi_method: str = "concat",
        bi_coupled: bool = False,
        log_likelihood: bool = False,
        variational_dropout: float = 0.0,
        use_layernorm: bool = True,
        sigma_temp: float = 0.7,
        kan_params: dict | None = None,
        direct_x: bool = True,
        x_min: float | None = None,
        x_max: float | None = None,
        clamp_in_forward: bool = True,
        noise_on_mask: bool = False,
    ):
        if bi_gru:
            raise ValueError("seqKAN core does not support bi_gru.")
        if not direct_x:
            raise ValueError("TSDF_seqKANSeq: direct_x=False not supported in KAN-only mode.")
        if static_dim > 0:
            raise ValueError("TSDF_seqKANSeq: static_dim>0 not supported in KAN-only mode.")
        if status_dim > 0:
            raise ValueError("TSDF_seqKANSeq: status_dim>0 not supported in KAN-only mode.")
        if lam[3] > 0.0:
            raise ValueError("TSDF_seqKANSeq: lam[3]>0 (miss head) not supported in KAN-only mode.")
        if lam[4] > 0.0 or lam[5] > 0.0:
            raise ValueError("TSDF_seqKANSeq: VAE components not supported in KAN-only mode.")
        if log_likelihood:
            raise ValueError("TSDF_seqKANSeq: log_likelihood not supported in KAN-only mode.")
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            static_dim=static_dim,
            lam=lam,
            num_steps=num_steps,
            cost_columns=cost_columns,
            status_dim=status_dim,
            log_likelihood=log_likelihood,
            sigma_temp=sigma_temp,
            x_max=x_max,
            x_min=x_min
        )
        self.direct_x = bool(direct_x)
        self.clamp_in_forward = bool(clamp_in_forward)
        self.noise_on_mask = bool(noise_on_mask)
        self._warned_mask_none = False
        self.encoder = None
        self.state_dim = in_channels * 2
        self.static_dim = 0
        if self.lam[0] > 0.0 or self.lam[4] > 0.0:
            # Reconstrução direta via KAN out (sem kan_out_rebuild e sem MLP decoder)
            self.seqKAN = SeqKANSeq(
                input_size=self.state_dim,
                hidden_size=hidden_dim,
                output_size=in_channels,
                kan_params=kan_params,
            )
            self.decoder = None

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool = False,
        return_x_hat: bool = False,
        mask: torch.Tensor = None,
        mask_ts: torch.Tensor = None,
        test: bool = True,
        only_gru: bool = False,
    ) -> torch.Tensor:
        noise = None
        noise_hat = None
        vae_x = None
        vae_mu = None
        vae_logvar = None
        vae_tmax = None
        vae_tmax_mu = None
        vae_tmax_logvar = None
        vae_tmax_logvar_obs = None
        vae_logvar_obs = None
        if self.clamp_in_forward:
            if self.x_min is not None:
                x = x.clamp(min=self.x_min)
            if self.x_max is not None:
                x = x.clamp(max=self.x_max)
        t = t if t is not None else torch.randint(0, self.num_steps, (x.size(0),), device=x.device)
        if mask_ts is None and mask is not None:
            mask_ts = mask.any(dim=2, keepdim=True).float()
        if self.direct_x and mask is None:
            if not self._warned_mask_none:
                warnings.warn(
                    "TSDF_seqKANSeq: mask=None with direct_x=True; using all-ones mask.",
                    stacklevel=2,
                )
                self._warned_mask_none = True
            mask = torch.ones_like(x)
        if mask_ts is None:
            mask_ts = mask.any(dim=2, keepdim=True).float()
        if not already_latent:
            h = torch.cat([x, mask], dim=-1)
            if not test and self.lam[1] > 0:
                if self.noise_on_mask:
                    noise = torch.randn_like(h)
                else:
                    noise = torch.zeros_like(h)
                    # gera ruído só nas features
                    noise_x = torch.randn_like(h[:, :, :self.in_channels])
                    if mask is not None:
                        noise_x = noise_x * mask  # feature-wise
                    noise[:, :, :self.in_channels] = noise_x
                ab = self.alpha_bar[t].view(-1, 1, 1)
                h = torch.sqrt(ab) * h + torch.sqrt(1 - ab) * noise
            else:
                t = torch.zeros((x.size(0),), device=x.device, dtype=torch.long)
                noise = None
        else:
            h = torch.cat([x, mask], dim=-1)
        if self.lam[0] > 0 or self.lam[4] > 0:
            outputs, h = self.seqKAN(h, return_last=True)
        ht = None
        tmax_hat = None

        if return_x_hat and self.lam[0] > 0:
            x_hat = outputs
        else:
            x_hat = None
        if x_hat is not None and self.clamp_in_forward:
            if self.x_min is not None:
                x_hat = x_hat.clamp(min=self.x_min)
            if self.x_max is not None:
                x_hat = x_hat.clamp(max=self.x_max)

        return (
            h,
            h,
            ht,
            x_hat,
            tmax_hat,
            noise,
            noise_hat,
            vae_x,
            vae_mu,
            vae_logvar,
            vae_tmax,
            vae_tmax_mu,
            vae_tmax_logvar,
            vae_logvar_obs if "vae_logvar_obs" in locals() else None,
            vae_tmax_logvar_obs if "vae_tmax_logvar_obs" in locals() else None,
        )
