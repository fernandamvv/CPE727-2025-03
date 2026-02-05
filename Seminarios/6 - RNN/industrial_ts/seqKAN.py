from .ode_jump import ODEJump
from .tsdiffusion import TSDiffusion
import torch.nn as nn
import torch
import torch.nn.functional as F


def extend_grid(grid, k_extend=0):
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
    for _ in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    return grid


def B_batch(x, grid, k=0):
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:, :, 0], grid=grid[0], k=k - 1)
        left = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)])
        right = (grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)])
        value = left * B_km1[:, :, :-1] + right * B_km1[:, :, 1:]
    return torch.nan_to_num(value)


def coef2curve(x_eval, grid, coef, k):
    b_splines = B_batch(x_eval, grid, k=k)
    return torch.einsum("ijk,jlk->ijl", b_splines, coef.to(b_splines.device))


class KANLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num=5,
        k=3,
        noise_scale=0.5,
        grid_range=(-4, 4),
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num = int(num)
        self.k = int(k)
        base_grid = torch.linspace(grid_range[0], grid_range[1], steps=self.num + 1)
        grid = base_grid[None, :].expand(self.in_dim, self.num + 1)
        grid = extend_grid(grid, k_extend=self.k)
        self.grid = nn.Parameter(grid, requires_grad=False)
        n_coef = self.grid.shape[1] - self.k - 1
        coef = (torch.rand(self.in_dim, self.out_dim, n_coef) - 0.5) * (noise_scale / max(self.num, 1))
        self.coef = nn.Parameter(coef)
        self.mask = nn.Parameter(torch.ones(self.in_dim, self.out_dim), requires_grad=False)

    def forward(self, x):
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)
        y = y * self.mask[None, :, :]
        return torch.sum(y, dim=1)


class KAN(nn.Module):
    def __init__(self, width, grid=3, k=3, grid_range=(-4, 4)):
        super().__init__()
        if len(width) != 2:
            raise ValueError("Minimal KAN only supports width=[in_dim, out_dim].")
        self.layer = KANLayer(width[0], width[1], num=grid, k=k, grid_range=grid_range)

    def forward(self, x):
        return self.layer(x)


class SeqKAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kan_params=None):
        super().__init__()
        if kan_params is None:
            kan_params = dict(
                hidden=dict(grid=5, k=3, grid_range=(-4, 4)),
                output=dict(grid=5, k=3, grid_range=(-4, 4)),
            )
        self.hidden_size = int(hidden_size)
        self.kan_hidden = KAN(
            width=[input_size + hidden_size, hidden_size],
            grid=kan_params["hidden"]["grid"],
            k=kan_params["hidden"]["k"],
            grid_range=kan_params["hidden"]["grid_range"],
        )
        self.kan_out = KAN(
            width=[hidden_size, output_size],
            grid=kan_params["output"]["grid"],
            k=kan_params["output"]["k"],
            grid_range=kan_params["output"]["grid_range"],
        )

    def forward(self, x, return_last=False):
        batch_size, seq_len, _ = x.shape
        hidden_state = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, hidden_state), dim=1)
            hidden_state = self.kan_hidden(combined)
            output_t = self.kan_out(hidden_state)
            outputs.append(output_t)
        outputs = torch.stack(outputs, dim=1)
        last = outputs[:, -1, :]
        return (outputs, last) if return_last else outputs


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

    def forward(self, x, return_last=False):
        x = self._apply_variational_dropout(x)
        return self.seqkan(x, return_last=return_last)


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
        if self.in_dim != self.hidden_dim:
            self.encoder = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            )
        else:
            self.encoder = None

    def _apply_variational_dropout(self, x):
        if not self.training or self.variational_dropout <= 0:
            return x
        B, _, C = x.shape
        mask = x.new_ones(B, C)
        mask = F.dropout(mask, p=self.variational_dropout, training=True)
        return x * mask.unsqueeze(1)

    def forward(self, x, ts=None, only_gru=False):  # noqa: ARG002
        x = self._apply_variational_dropout(x)
        if self.norm_x is not None:
            x = self.norm_x(x)
        outputs = self.seqkan(x, return_last=False)
        if self.norm_H is not None:
            outputs = self.norm_H(outputs)
        return outputs if self.encoder is None else self.encoder(outputs)



class TS_seqKAN(ODEJump):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        static_dim: int = 0,
        denoised: bool = False,
        lam: list[float,float] = [0.9, 0.1],
        cost_columns: list = None,
        bi_gru: bool = False,
        bi_method: str = 'concat',
        bi_coupled: bool = False,
        variational_dropout: float = 0.0,
        kan_params: dict | None = None,
        direct_x: bool = True
        
    ):
        self.lam = lam
        super().__init__(in_channels, hidden_dim, static_dim, denoised, lam, cost_columns)
        self.val_loss = float('inf')
        self.model_dim = hidden_dim
        self.in_channels = in_channels
        self.direct_x = bool(direct_x)
        if not self.direct_x:
            self.encoder = nn.Sequential(
                nn.Linear(in_channels*2, hidden_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim if not (bi_gru and bi_method=='concat') else hidden_dim * 2, hidden_dim // 2),
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
                nn.ReLU()
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
        # (d) m_b  — probabilidade de observação (Bernoulli) para L4
        self.miss_head = nn.Linear(hidden_dim if not (bi_gru and bi_method=='concat') else hidden_dim * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
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
            if x_denoised is not None:
                # aplique gate (treino=True quando model.training)
                x_fused, _ = self.denoise_gate(x, x_denoised, mask, train_mode=self.training)
                h_in = x_fused if self.direct_x else torch.cat([x_fused, mask], dim=-1)
            else:
                h_in = x if self.direct_x else torch.cat([x, mask], dim=-1)

            h = h_in if self.direct_x else self.encoder(h_in)  # (B,T,hidden_dim)
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        #tm_e = self.time_encoding(timestamps.to(h.dtype)).to(h.dtype)  # tempo contínuo
        #h = h + tm_e
        h = self.seqkan(h)
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
        lam: list[float,float,float,float,float,float] = [0.9, 0.0, 0.0, 0.1, 0.0, 0.0],
        num_steps: int = 1000,
        cost_columns: list = None,
        bi_gru: bool = False,
        bi_method: str = 'concat',
        bi_coupled: bool = False,
        log_likelihood: bool = False,
        variational_dropout: float = 0.0,
        use_layernorm: bool = True,
        sigma_temp: float = 0.7,
        kan_params: dict | None = None,
        direct_x: bool = True
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
            sigma_temp=sigma_temp
        )
        self.direct_x = bool(direct_x)
        if not self.direct_x:
            self.encoder = nn.Sequential(
                nn.Linear(in_channels*2, hidden_dim),
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
                nn.ReLU()
            )

        if status_dim > 0:
            self.tmax_head = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, status_dim)
            )
            self.encoder_ode_tmax = SeqKANEncoder(
                self.state_dim,
                self.state_dim,
                kan_params=kan_params,
                variational_dropout=variational_dropout,
            )
            if self.log_likelihood:
                self.lambda_tmax_head = nn.Sequential(
                    nn.Linear(self.state_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1)       # escalar
                )   
            # variância observacional para vae_tmax
        if self.lam[3] > 0.0:  
            self.miss_head = nn.Linear(self.state_dim, 1)
        if self.lam[0] > 0.0 or self.lam[4] > 0.0:
            self.encoder_ode_x = SeqKANEncoder(
                self.state_dim,
                self.state_dim,
                kan_params=kan_params,
                variational_dropout=variational_dropout,
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
                    nn.Linear(hidden_dim // 2, 1)       # escalar
                )
        if self.lam[4] > 0:
            self.vae_latent = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2)
            )
            self.vae_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, in_channels)
            )
            self.vae_sigma_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, in_channels)
            )
        if self.lam[5] > 0 and status_dim > 0:
            self.vae_tmax_latent = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2)
            )
            self.vae_tmax_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, status_dim)
            )
            self.vae_tmax_sigma_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, status_dim)
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask: torch.Tensor = None,
        mask_ts: torch.Tensor = None,
        test: bool=True,
        only_gru: bool = False
    ) -> torch.Tensor:
        only_gru = True
        """
        Args:
            x: (batch, seq_len, in_channels) - dados ruidosos.
            t: (batch,) - passos de difusão.
            timestamps: (batch, seq_len) - colunas de tempo.
            static_feats: (batch, static_dim).
        """
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
        # Embedding de entrada
        if not already_latent:
            if self.direct_x:
                h = x
            else:
                h = self.encoder(torch.cat([x, mask], dim=-1))
            if not test and self.lam[1]>0:
                noise = torch.randn_like(h) * mask_ts
                ab = self.alpha_bar[t].view(-1, 1, 1)
                h = torch.sqrt(ab) * h + torch.sqrt(1 - ab) * noise
            else:
                t = torch.zeros((x.size(0),), device=x.device, dtype=torch.long)
                noise = None
        else:
            h = x
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        if self.lam[0]>0 or self.lam[4]>0:
            h = self.encoder_ode_x(h, timestamps, only_gru)
        if self.lam[2]>0 or self.lam[5]>0:
            ht = self.encoder_ode_tmax(h, timestamps, only_gru)
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

        if return_x_hat and self.lam[0] > 0:
            x_hat = h if self.direct_x else self.decoder(h)
        else:
            x_hat = None

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
            vae_logvar_obs if 'vae_logvar_obs' in locals() else None,
            vae_tmax_logvar_obs if 'vae_tmax_logvar_obs' in locals() else None,
        )
