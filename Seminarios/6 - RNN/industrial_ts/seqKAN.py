from __future__ import annotations

from pathlib import Path
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ode_jump import ODEJump
from .tsdiffusion import TSDiffusion

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
    return ArticleKAN(
        width=width,
        grid=grid,
        k=k,
        grid_range=list(grid_range),
        seed=params.get("seed", 42),
        sp_trainable=params.get("sp_trainable", False),
        sb_trainable=params.get("sb_trainable", False),
        affine_trainable=params.get("affine_trainable", False),
        symbolic_enabled=params.get("symbolic_enabled", False),
        auto_save=False,
        save_act=False,
        device=str(device),
    )


class SeqKAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kan_params=None, device=None):
        super().__init__()
        if kan_params is None:
            kan_params = dict(
                hidden=dict(grid=20, k=3, grid_range=(-10, 10)),
                output=dict(grid=20, k=3, grid_range=(-10, 10)),
            )
        self.hidden_size = int(hidden_size)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.kan_x = _build_kan(
            width=[input_size, hidden_size],
            params=kan_params["hidden"],
            device=self.device,
        )
        self.kan_h = _build_kan(
            width=[hidden_size, hidden_size],
            params=kan_params["hidden"],
            device=self.device,
        )
        # z_t: sequência paralela
        self.kan_zx = _build_kan(
            width=[input_size, hidden_size],
            params=kan_params["hidden"],
            device=self.device,
        )
        self.kan_zz = _build_kan(
            width=[hidden_size, hidden_size],
            params=kan_params["hidden"],
            device=self.device,
        )
        self.kan_out = _build_kan(
            width=[hidden_size, output_size],
            params=kan_params["output"],
            device=self.device,
        )
        aux_params = kan_params.get(
            "aux",
            dict(
                grid=12,
                k=kan_params.get("output", {}).get("k", 3),
                grid_range=kan_params.get("output", {}).get("grid_range", (-10, 10)),
            ),
        )
        self.kan_out_aux = _build_kan(
            width=[input_size, output_size],
            params=aux_params,
            device=self.device,
        )
        self.aux_alpha = nn.Parameter(torch.tensor(0.1))
        self.aux_alpha_x = nn.Parameter(torch.tensor(1.0))
        self.aux_alpha_z = nn.Parameter(torch.tensor(1.0))
        self.aux_tau_z = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, mask=None, return_last=False):
        if next(self.parameters()).device != x.device:
            self.to(x.device)
        batch_size, seq_len, _ = x.shape
        hidden_state = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        z_state = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            s1 = self.kan_x(x_t)
            if mask is None:
                m = 1.0
            else:
                m_t = mask[:, t, :]
                if m_t.shape[1] != self.hidden_size:
                    m = m_t.any(dim=1, keepdim=True).float()
                else:
                    m = m_t.float()
            # z_t em paralelo (usa todo z_{t-1})
            z_prev = z_state
            z_state = self.kan_zx(x_t) + self.kan_zz(z_prev)
            # h_t usa z_{t-1}
            hidden_state = hidden_state + m * s1 + (1 - m) * z_prev
            output_t = self.kan_out(hidden_state)
            gate = torch.sigmoid(self.aux_alpha_x * (x_t.abs() - 2.0)).mean(dim=1, keepdim=True)
            aux_out = self.kan_out_aux(x_t)
            output_t = output_t + self.aux_alpha * gate * aux_out
            outputs.append(output_t)
        outputs = torch.stack(outputs, dim=1)
        last = outputs[:, -1, :]
        return (outputs, last) if return_last else outputs

    def aux_spline_penalty(self):
        if self.kan_out_aux is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        reg = 0.0
        for layer in self.kan_out_aux.act_fun:
            coef = layer.coef
            if coef.shape[-1] >= 3:
                d2 = torch.diff(coef, n=2, dim=-1)
                reg = reg + d2.abs().mean()
        if not isinstance(reg, torch.Tensor):
            reg = torch.tensor(reg, device=next(self.parameters()).device)
        return reg


class SeqKANCore(nn.Module):
    def __init__(self, hidden_dim, input_dim=None, output_dim=None, kan_params=None, variational_dropout=0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        input_dim = self.hidden_dim if input_dim is None else int(input_dim)
        output_dim = self.hidden_dim if output_dim is None else int(output_dim)
        self.variational_dropout = float(max(0.0, variational_dropout))
        self.seqkan = SeqKAN(
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

    def aux_spline_penalty(self):
        return self.seqkan.aux_spline_penalty()


class SeqKANEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, kan_params=None, variational_dropout=0.0, use_layernorm: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.variational_dropout = float(max(0.0, variational_dropout))
        self.use_layernorm = use_layernorm
        self.seqkan = SeqKAN(
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


class TS_seqKAN(ODEJump):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        static_dim: int = 0,
        denoised: bool = False,
        lam: list[float, float] = [0.9, 0.1],
        cost_columns: list = None,
        bi_gru: bool = False,
        bi_method: str = "concat",
        bi_coupled: bool = False,
        variational_dropout: float = 0.0,
        kan_params: dict | None = None,
        direct_x: bool = True,
    ):
        self.lam = lam
        super().__init__(in_channels, hidden_dim, static_dim, denoised, lam, cost_columns)
        self.val_loss = float("inf")
        self.model_dim = hidden_dim
        self.in_channels = in_channels
        self.direct_x = bool(direct_x)
        self._warned_mask_none = False
        if not self.direct_x:
            self.encoder = nn.Sequential(
                nn.Linear(in_channels * 2, hidden_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim if not (bi_gru and bi_method == "concat") else hidden_dim * 2, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, in_channels),
            )
        else:
            self.encoder = None
            self.decoder = None
        self.static_dim = static_dim
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, in_channels if self.direct_x else hidden_dim),
                nn.ReLU(),
            )
        if bi_gru:
            raise ValueError("seqKAN core does not support bi_gru.")
        self.seqkan = SeqKANCore(
            hidden_dim,
            input_dim=in_channels if self.direct_x else hidden_dim,
            output_dim=in_channels if self.direct_x else hidden_dim,
            kan_params=kan_params,
            variational_dropout=variational_dropout,
        )
        # (d) m_b — probabilidade de observação (Bernoulli) para L4
        # Se direct_x=True, o state tem dimensão in_channels.
        miss_in = in_channels if self.direct_x else hidden_dim
        if bi_gru and bi_method == "concat" and not self.direct_x:
            miss_in = hidden_dim * 2
        self.miss_head = nn.Linear(miss_in, 1)

    def aux_spline_penalty(self):
        return self.seqkan.aux_spline_penalty()

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool = False,
        return_x_hat: bool = False,
        mask=None,
        x_denoised: torch.Tensor = None,
    ) -> torch.Tensor:
        if not already_latent:
            if x_denoised is not None:
                x_fused, _ = self.denoise_gate(x, x_denoised, mask, train_mode=self.training)
            else:
                x_fused = x
            if self.direct_x:
                if mask is None:
                    if not self._warned_mask_none:
                        warnings.warn(
                            "TS_seqKAN: mask=None with direct_x=True; using all-ones mask.",
                            stacklevel=2,
                        )
                        self._warned_mask_none = True
                    mask = torch.ones_like(x_fused)
                h_base = x_fused
            else:
                h_in = torch.cat([x_fused, mask], dim=-1)
                h = self.encoder(h_in)
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)
            if self.direct_x:
                h_base = h_base + se
            else:
                h = h + se
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        if self.direct_x:
            h = self.seqkan(h_base, mask=mask)
        else:
            h = self.seqkan(h, mask=mask)
        state = h
        if return_x_hat:
            x_hat = state if self.direct_x else self.decoder(state)
        else:
            x_hat = None
        return state, x_hat


class TSDF_seqKAN(TSDiffusion):
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
            self.encoder_ode_tmax = SeqKANEncoder(
                self.state_dim,
                self.state_dim,
                kan_params=kan_params,
                variational_dropout=variational_dropout,
                use_layernorm=use_layernorm,
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
            self.encoder_ode_x = SeqKANEncoder(
                self.state_dim,
                self.state_dim,
                kan_params=kan_params,
                variational_dropout=variational_dropout,
                use_layernorm=use_layernorm,
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
        t = t if t is not None else torch.randint(0, self.num_steps, (x.size(0),), device=x.device)
        if mask_ts is None and mask is not None:
            mask_ts = mask.any(dim=2, keepdim=True).float()
        if self.feature_scale and not already_latent:
            scale = self.feature_scale_log.exp().view(1, 1, -1)
            x = x * scale
        if self.direct_x and mask is None:
            if not self._warned_mask_none:
                warnings.warn(
                    "TSDF_seqKAN: mask=None with direct_x=True; using all-ones mask.",
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
            if self.direct_x:
                h = self.encoder_ode_x(h, timestamps, only_gru, mask=mask)
            else:
                h = self.encoder_ode_x(h, timestamps, only_gru, mask=mask)
        if self.lam[2] > 0 or self.lam[5] > 0:
            if self.direct_x:
                ht = self.encoder_ode_tmax(h, timestamps, only_gru, mask=mask)
            else:
                ht = self.encoder_ode_tmax(h, timestamps, only_gru, mask=mask)
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
