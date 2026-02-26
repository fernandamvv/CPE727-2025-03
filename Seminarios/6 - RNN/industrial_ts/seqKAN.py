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

    def __init__(self, input_size, hidden_size, output_size, kan_params=None, device=None, output_size_rebuild=None):
        super().__init__()
        if kan_params is None:
            kan_params = {}
        self.kan_params = kan_params
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.output_size_rebuild = int(output_size_rebuild) if output_size_rebuild is not None else None
        self.device = torch.device("cpu") if device is None else torch.device(device)

        cell_params = kan_params.get("cell", kan_params.get("hidden", {}))
        out_params = kan_params.get("output", {})

        self.kan_cell = _build_kan(
            width=[self.input_size + self.hidden_size, self.hidden_size],
            params=cell_params,
            device=self.device,
        )
        self.kan_out = nn.ModuleList(
            [
                _build_kan(
                    width=[self.hidden_size, 1],
                    params=out_params,
                    device=self.device,
                )
                for _ in range(self.output_size)
            ]
        )
        if self.output_size_rebuild is not None:
            self.kan_out_rebuild = nn.ModuleList(
                [
                    _build_kan(
                        width=[self.hidden_size, 1],
                        params=out_params,
                        device=self.device,
                    )
                    for _ in range(self.output_size_rebuild)
                ]
            )
        else:
            self.kan_out_rebuild = None

        topk_cfg = kan_params.get("topk", {}) if isinstance(kan_params, dict) else {}
        self.topk_enabled = bool(topk_cfg.get("enabled", False))
        self.topk_warmup_epochs = int(topk_cfg.get("warmup_epochs", 0) or 0)
        self.topk_mode = str(topk_cfg.get("mode", "hard")).lower()
        self.topk_temp = float(topk_cfg.get("temp", 0.1))
        self.topk_kx = topk_cfg.get("k_x", None)
        self.topk_kh = topk_cfg.get("k_h", None)
        self.topk_kout = topk_cfg.get("k_out", None)
        self.last_topk_ratio = {}
        self.current_epoch = None

    def _apply_topk(self, x, k, name):
        if not self.topk_enabled:
            return x
        if self.current_epoch is None or self.current_epoch <= self.topk_warmup_epochs:
            return x
        if k is None:
            return x
        k = int(k)
        if k <= 0 or k >= x.size(-1):
            return x
        scores = x.abs()
        vals, idx = torch.topk(scores, k=k, dim=-1)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(dim=-1, index=idx, value=True)
        if self.topk_mode == "soft":
            kth = vals[..., -1, None]
            gate = torch.sigmoid((scores - kth) / max(self.topk_temp, 1e-6))
            out = x * gate
            ratio = gate.mean().detach().item()
        else:
            out = torch.where(mask, x, torch.zeros_like(x))
            ratio = mask.float().mean().detach().item()
        self.last_topk_ratio[name] = ratio
        return out

    def forward(self, x, mask=None, return_last=False):
        if next(self.parameters()).device != x.device:
            self.to(x.device)
        B, T, C = x.shape
        if C != self.input_size:
            raise ValueError("SeqKANSeq: input_size mismatch")
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        outputs_rebuild = [] if self.kan_out_rebuild is not None else None
        for t in range(T):
            x_t = x[:, t, :]
            if mask is not None:
                x_t = x_t * mask[:, t, :]
            x_t = self._apply_topk(x_t, self.topk_kx, "x")
            h_prev = self._apply_topk(h, self.topk_kh, "h")
            h_in = torch.cat([x_t, h_prev], dim=-1)
            h = self.kan_cell(h_in)
            y_list = []
            for i, head in enumerate(self.kan_out):
                h_out = self._apply_topk(h, self.topk_kout, f"out{i}")
                y_list.append(head(h_out))
            y_t = torch.cat(y_list, dim=-1)
            outputs.append(y_t)
            if self.kan_out_rebuild is not None:
                y_list_r = []
                for i, head in enumerate(self.kan_out_rebuild):
                    h_out = self._apply_topk(h, self.topk_kout, f"rebuild_out{i}")
                    y_list_r.append(head(h_out))
                y_t_r = torch.cat(y_list_r, dim=-1)
                outputs_rebuild.append(y_t_r)
        outputs = torch.stack(outputs, dim=1)
        last = outputs[:, -1, :]
        if self.kan_out_rebuild is None:
            return (outputs, last) if return_last else outputs
        outputs_rebuild = torch.stack(outputs_rebuild, dim=1)
        last_rebuild = outputs_rebuild[:, -1, :]
        if return_last:
            return (outputs, outputs_rebuild), (last, last_rebuild)
        return outputs, outputs_rebuild


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

    def forward(self, x, mask=None, return_last=False):
        x = self._apply_variational_dropout(x)
        return self.seqkan(x, mask=mask, return_last=return_last)


class SeqKANEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, kan_params=None, variational_dropout=0.0, use_layernorm: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.variational_dropout = float(max(0.0, variational_dropout))
        self.use_layernorm = use_layernorm
        self.seqkan = SeqKANSeq(
            input_size=self.in_dim,
            hidden_size=self.hidden_dim,
            output_size=self.hidden_dim,
            kan_params=kan_params,
        )
        self.norm_x = nn.LayerNorm(self.in_dim) if use_layernorm else None
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
        if self.norm_x is not None:
            x = self.norm_x(x)
        outputs = self.seqkan(x, mask=mask, return_last=False)
        if self.norm_H is not None:
            outputs = self.norm_H(outputs)
        return outputs


class TS_seqKANSeq(ODEJump):
    """
    SeqKANSeq com o mesmo pipeline/loss do TS_GRU (ODEJump).
    Apenas o backbone recursivo muda (GRU -> SeqKANSeq).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        static_dim: int = 0,
        denoised: bool = False,
        lam: list[float, float] | None = None,
        cost_columns: list | None = None,
        kan_params: dict | None = None,
    ):
        if lam is None:
            lam = [0.9, 0.1, 0.0, 0.0]
        # garante tamanho minimo para compatibilidade com ODEJump
        if len(lam) < 4:
            lam = list(lam) + [0.0] * (4 - len(lam))
        self.lam = lam
        super().__init__(in_channels, hidden_dim, static_dim, denoised, lam, cost_columns)
        self.val_loss = float("inf")
        self.model_dim = hidden_dim
        self.in_channels = in_channels
        self.static_dim = static_dim
        if kan_params is None:
            kan_params = {}

        self.encoder = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, in_channels),
        )
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU(),
            )

        # Backbone seqKANseq no espaço latente
        self.seq = SeqKANSeq(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            output_size=hidden_dim,
            kan_params=kan_params,
            device=None,
        )
        # (d) m_b  — probabilidade de observação (Bernoulli) para L4
        self.miss_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool = False,
        return_x_hat: bool = False,
        mask = None,
        x_denoised: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels) - dados ruidosos.
            t: (batch,) - passos de difusão.
            timestamps: (batch, seq_len) - colunas de tempo.
            static_feats: (batch, static_dim).
        """
        # Embedding de entrada
        if not already_latent:
            x = self._clip_x_tensor(x)
            if x_denoised is not None:
                x_denoised = self._clip_x_tensor(x_denoised)
            if x_denoised is not None:
                # aplique gate (treino=True quando model.training)
                x_fused, _ = self.denoise_gate(x, x_denoised, mask, train_mode=self.training)
                h_in = torch.cat([x_fused, mask], dim=-1)
            else:
                h_in = torch.cat([x, mask], dim=-1)

            h = self.encoder(h_in)  # (B,T,hidden_dim)
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        h = self.seq(h)
        state = h
        x_hat = self.decoder(state) if return_x_hat else None
        if x_hat is not None:
            x_hat = self._clip_x_tensor(x_hat)
        return x, state, x_hat


class TSDF_seqKANSeq(TSDiffusion):
    """
    TSDiffusion com backbone SeqKANSeq (difusao no latente, mesmo esquema do TSDiffusion).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
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
        feature_scale: bool = False,
    ):
        if bi_gru:
            raise ValueError("seqKAN core does not support bi_gru.")
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
        )
        self.direct_x = bool(direct_x)
        self.feature_scale = bool(feature_scale)
        if self.feature_scale:
            self.feature_scale_log = nn.Parameter(torch.zeros(in_channels))
        self._warned_mask_none = False
        if not self.direct_x:
            self.encoder = nn.Sequential(
                nn.Linear(in_channels * 2, hidden_dim),
                nn.ReLU(),
            )
            self.state_dim = hidden_dim
        else:
            self.encoder = None
            self.state_dim = in_channels

        self.static_dim = static_dim
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, in_channels if self.direct_x else hidden_dim),
                nn.ReLU(),
            )

        if status_dim > 0:
            self.tmax_head = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, status_dim),
            )
            self.encoder_ode_tmax = SeqKANSeq(
                input_size=self.state_dim,
                hidden_size=hidden_dim,
                output_size=self.state_dim,
                kan_params=kan_params,
            )
            if self.log_likelihood:
                self.lambda_tmax_head = nn.Sequential(
                    nn.Linear(self.state_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
        if self.lam[3] > 0.0:
            self.miss_head = nn.Linear(self.state_dim, 1)
        if self.lam[0] > 0.0 or self.lam[4] > 0.0:
            self.encoder_ode_x = SeqKANSeq(
                input_size=self.state_dim,
                hidden_size=hidden_dim,
                output_size=self.state_dim,
                kan_params=kan_params,
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, in_channels),
            )
            if log_likelihood:
                self.lambda_head = nn.Sequential(
                    nn.Linear(self.state_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
        if self.lam[4] > 0:
            self.vae_latent = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
            )
            self.vae_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, in_channels),
            )
            self.vae_sigma_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, in_channels),
            )
        if self.lam[5] > 0 and status_dim > 0:
            self.vae_tmax_latent = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
            )
            self.vae_tmax_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, status_dim),
            )
            self.vae_tmax_sigma_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, status_dim),
            )

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
        x = self._clip_x_tensor(x)
        t = t if t is not None else torch.randint(0, self.num_steps, (x.size(0),), device=x.device)
        if mask_ts is None and mask is not None:
            mask_ts = mask.any(dim=2, keepdim=True).float()
        if self.feature_scale and not already_latent:
            scale = self.feature_scale_log.exp().view(1, 1, -1)
            x = x * scale
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
            if self.direct_x:
                h = x
            else:
                h = self.encoder(torch.cat([x, mask], dim=-1))
            if not test and self.lam[1] > 0:
                noise = torch.randn_like(h) * mask_ts
                ab = self.alpha_bar[t].view(-1, 1, 1)
                h = torch.sqrt(ab) * h + torch.sqrt(1 - ab) * noise
            else:
                t = torch.zeros((x.size(0),), device=x.device, dtype=torch.long)
                noise = None
        else:
            h = x
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)
            h = h + se
        if self.lam[0] > 0 or self.lam[4] > 0:
            h = self.encoder_ode_x(h, mask=mask)
        if self.lam[2] > 0 or self.lam[5] > 0:
            ht = self.encoder_ode_tmax(h, mask=mask)
            tmax_hat = self.tmax_head(ht)
        else:
            ht = None
            tmax_hat = None
        if self.lam[4] > 0:
            mu_logvar = self.vae_latent(h)
            vae_mu, vae_logvar = torch.chunk(mu_logvar, 2, dim=-1)
            std = torch.exp(0.5 * vae_logvar)
            eps = torch.randn_like(std)
            z_vae = vae_mu + eps * std
            vae_x = self.vae_decoder(z_vae)
            vae_logvar_obs = self.vae_sigma_head(z_vae).clamp(min=-5.0, max=5.0)

        if self.lam[5] > 0 and ht is not None:
            mu_logvar_t = self.vae_tmax_latent(ht)
            vae_tmax_mu, vae_tmax_logvar = torch.chunk(mu_logvar_t, 2, dim=-1)
            std_t = torch.exp(0.5 * vae_tmax_logvar)
            eps_t = torch.randn_like(std_t)
            z_tmax = vae_tmax_mu + eps_t * std_t
            vae_tmax = self.vae_tmax_decoder(z_tmax)
            vae_tmax_logvar_obs = self.vae_tmax_sigma_head(z_tmax).clamp(min=-5.0, max=5.0)

        x_hat = self.decoder(h) if return_x_hat and self.lam[0] > 0 else None
        if x_hat is not None:
            x_hat = self._clip_x_tensor(x_hat)

        return (
            x,
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
