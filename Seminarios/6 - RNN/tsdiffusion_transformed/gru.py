from .ode_jump import ODEJump
from .tsdiffusion import TSDiffusion
import torch.nn as nn
import torch
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(
        self,
        hidden_dim,
        bi_gru,
        bi_method,
        bi_coupled,
        variational_dropout: float = 0.0
    ):
        super().__init__()
        self.bi_gru = bi_gru
        self.bi_method = bi_method
        self.bi_coupled = bi_coupled
        self.hidden_dim=hidden_dim
        self.variational_dropout = float(max(0.0, variational_dropout))
        self.gru = nn.GRUCell(hidden_dim,hidden_dim)
        if bi_gru:
            self.gru_bw = nn.GRUCell(hidden_dim,hidden_dim)
            if bi_method == 'gate':
                self.bi_gate = nn.Sequential(
                    nn.Linear(hidden_dim*2,hidden_dim),
                    nn.Sigmoid()
                )
            elif bi_method == 'gru':
                self.gru_fuser = nn.GRUCell(hidden_dim*2,hidden_dim)
    def _apply_variational_dropout(self, x):
        if not self.training or self.variational_dropout <= 0:
            return x
        B, _, C = x.shape
        mask = x.new_ones(B, C)
        mask = F.dropout(mask, p=self.variational_dropout, training=True)
        return x * mask.unsqueeze(1)

    def forward(self, x):
        """
        x  : (B, T, C)
        ts : (B, T)   segundos unix (normalizados ou não)
        """
        x = self._apply_variational_dropout(x)
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        states = []
        if self.bi_gru:

            states_bw = []
            for i in reversed(range(T)):
                h = self.gru_bw(x[:, i], h)                                 # jump
                states_bw.append(h)
            states_bw.reverse()
            if not self.bi_coupled:
                h = torch.zeros(B, self.hidden_dim, device=x.device)

        for i in range(T):
            h = self.gru(x[:, i], h)                                 # jump
            states.append(h)
        if self.bi_gru:
            states_concat = [torch.cat([f,b],dim=-1) for f,b in zip(states,states_bw)]
            if self.bi_method == 'concat':
                states = states_concat
            elif self.bi_method == 'gate':
                states_concat = torch.stack(states_concat, dim=1)
                states = torch.stack(states, dim=1)
                states_bw = torch.stack(states_bw, dim=1)
                sigma = self.bi_gate(states_concat)
                states = sigma * states + (1-sigma) * states_bw
            elif self.bi_method == 'gru':
                states_concat = torch.stack(states_concat, dim=1)
                h = torch.zeros(B, self.hidden_dim, device=x.device)
                states = []
                for i in range(T):
                    h = self.gru_fuser(states_concat[:,i],h)
                    states.append(h)
        if self.bi_method != 'gate':
            H = torch.stack(states, dim=1)   # (B, T, hidden_dim*2)
        else:
            H = states  # (B, T, hidden_dim)

        return H

class TS_GRU(ODEJump):
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
        variational_dropout: float = 0.0
        
    ):
        self.lam = lam
        super().__init__(in_channels, hidden_dim, static_dim, denoised, lam, cost_columns)
        self.val_loss = float('inf')
        self.model_dim = hidden_dim
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Linear(in_channels*2, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim if not (bi_gru and bi_method=='concat') else hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, in_channels),
        )
        self.static_dim = static_dim
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU()
            )
        self.gru = GRU(
            hidden_dim,
            bi_gru,
            bi_method,
            bi_coupled,
            variational_dropout=variational_dropout
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
                h_in = torch.cat([x_fused, mask], dim=-1)
            else:
                h_in = torch.cat([x, mask], dim=-1)

            h = self.encoder(h_in)  # (B,T,hidden_dim)
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        #tm_e = self.time_encoding(timestamps.to(h.dtype)).to(h.dtype)  # tempo contínuo
        #h = h + tm_e
        h = self.gru(h)
        state = h
        return state,self.decoder(state) if return_x_hat else None    
        
class GRUEncoder(nn.Module):
    """
    Self‑Attentive Jump‑ODE simplificado:
    - GRUCell executa o *jump* g_ψ na chegada de cada evento (x_i, t_i)
    - ODEFunc integra h(t) entre eventos.
    - Self‑attention usa máscara para faltantes (opcional).
    """
    def __init__(
        self,
        in_dim,
        hidden_dim,
        bi_gru,
        bi_method,
        bi_coupled,
        variational_dropout: float = 0.0,
        use_layernorm: bool = True
    ):
        super().__init__()
        self.bi_gru = bi_gru
        self.bi_method = bi_method
        self.bi_coupled = bi_coupled
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.variational_dropout = float(max(0.0, variational_dropout))
        self.use_layernorm = use_layernorm
        self.gru = nn.GRUCell(in_dim, in_dim)
        self.norm_x = nn.LayerNorm(in_dim) if use_layernorm else None
        out_dim = in_dim if not (bi_gru and bi_method == 'concat') else in_dim * 2
        self.norm_H = nn.LayerNorm(out_dim) if use_layernorm else None
        if bi_gru:
            self.gru_bw = nn.GRUCell(in_dim, in_dim)
            if bi_method == 'gate':
                self.bi_gate = nn.Sequential(
                    nn.Linear(hidden_dim*2,hidden_dim),
                    nn.Sigmoid()
                )
            elif bi_method == 'gru':
                self.gru_fuser = nn.GRUCell(hidden_dim*2,hidden_dim)

        if in_dim != hidden_dim:
            self.encoder=nn.Sequential(
                nn.Linear(in_dim,hidden_dim*4),
                nn.GELU(),
                nn.Linear(hidden_dim*4,hidden_dim)
            )
 

    def forward(self, x, ts, only_gru=False):
        if self.training and self.variational_dropout > 0:
            B, _, C = x.shape
            mask = x.new_ones(B, C)
            mask = F.dropout(mask, p=self.variational_dropout, training=True)
            x = x * mask.unsqueeze(1)
        B, T, C = x.shape
        #eps = 1e-6
        h = torch.zeros(B, self.in_dim, device=x.device)
        states = []
        if self.norm_x is not None:
            x = self.norm_x(x)
        if self.bi_gru:

            states_bw = []
            for i in reversed(range(T)):
                h = self.gru_bw(x[:, i], h)                                 # jump
                states_bw.append(h)
            states_bw.reverse()
            if not self.bi_coupled:
                h = torch.zeros(B, self.hidden_dim, device=x.device)
        for i in range(T):

            # JUMP no evento i (como no esquema original)
            h = self.gru(x[:, i], h)
            states.append(h)
        if self.bi_gru:
            states_concat = [torch.cat([f,b],dim=-1) for f,b in zip(states,states_bw)]
            if self.bi_method == 'concat':
                states = states_concat
            elif self.bi_method == 'gate':
                states_concat = torch.stack(states_concat, dim=1)
                states = torch.stack(states, dim=1)
                states_bw = torch.stack(states_bw, dim=1)
                sigma = self.bi_gate(states_concat)
                states = sigma * states + (1-sigma) * states_bw
            elif self.bi_method == 'gru':
                states_concat = torch.stack(states_concat, dim=1)
                h = torch.zeros(B, self.hidden_dim, device=x.device)
                states = []
                for i in range(T):
                    h = self.gru_fuser(states_concat[:,i],h)
                    states.append(h)

        if self.bi_method != 'gate':
            H = torch.stack(states, dim=1)   # (B, T, hidden_dim*2)
        else:
            H = states  # (B, T, hidden_dim)
        if self.norm_H is not None:
            H = self.norm_H(H)
        return H if self.in_dim == self.hidden_dim else self.encoder(H)

class TSDF_GRU(TSDiffusion):
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
        mog_components: int = 16,
        mog_components_tmax: int | None = None,
        vamp_init_scale: float = 0.1,
        vamp_logvar_min: float = -8.0,
        vamp_logvar_max: float = 8.0,
        ):
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
            mog_components=mog_components,
            mog_components_tmax=mog_components_tmax,
            vamp_init_scale=vamp_init_scale,
            vamp_logvar_min=vamp_logvar_min,
            vamp_logvar_max=vamp_logvar_max,
        )
        self.encoder = nn.Sequential(
            nn.Linear(in_channels*2, hidden_dim),
            nn.ReLU(),
        )
        self.state_dim = hidden_dim if not (bi_gru and bi_method == 'concat') else hidden_dim * 2

        self.static_dim = static_dim
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU()
            )

        if status_dim > 0:
            self.tmax_head = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, status_dim)
            )
            self.encoder_ode_tmax = GRUEncoder(
                hidden_dim,
                hidden_dim,
                bi_gru,
                bi_method,
                bi_coupled,
                variational_dropout=variational_dropout,
                use_layernorm=use_layernorm
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
            self.encoder_ode_x = GRUEncoder(
                hidden_dim,
                hidden_dim,
                bi_gru,
                bi_method,
                bi_coupled,
                variational_dropout=variational_dropout,
                use_layernorm=use_layernorm
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
            self.mog_pi_z = nn.Linear(self.state_dim, self.mog_components)
            self.mog_mu_z = nn.Linear(self.state_dim, self.mog_components * self.state_dim)
            self.mog_logvar_z = nn.Linear(self.state_dim, self.mog_components * self.state_dim)
            self.vamp_log_weights = nn.Parameter(torch.zeros(self.mog_components))
            self.vamp_pseudo_inputs = nn.Parameter(
                torch.randn(self.mog_components, 1, self.state_dim) * self.vamp_init_scale
            )
            self.vae_decoder = nn.Sequential(
                nn.Linear(self.state_dim, self.state_dim),
                nn.GELU(),
                nn.Linear(self.state_dim, in_channels)
            )
            self.vae_sigma_head = nn.Sequential(
                nn.Linear(self.state_dim, self.state_dim),
                nn.GELU(),
                nn.Linear(self.state_dim, in_channels)
            )
        if self.lam[5] > 0 and status_dim > 0:
            self.mog_pi_zt = nn.Linear(self.state_dim, self.mog_components_tmax)
            self.mog_mu_zt = nn.Linear(self.state_dim, self.mog_components_tmax * self.state_dim)
            self.mog_logvar_zt = nn.Linear(self.state_dim, self.mog_components_tmax * self.state_dim)
            self.vamp_log_weights_tmax = nn.Parameter(torch.zeros(self.mog_components_tmax))
            self.vamp_pseudo_inputs_tmax = nn.Parameter(
                torch.randn(self.mog_components_tmax, 1, self.state_dim) * self.vamp_init_scale
            )
            self.vae_tmax_decoder = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, status_dim)
            )
            self.vae_tmax_sigma_head = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim // 2),
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
        pi_z_logits = None
        mu_z = None
        logvar_z = None
        prior_pi_z = None
        prior_mu_z = None
        prior_logvar_z = None
        pi_zt_logits = None
        mu_zt = None
        logvar_zt = None
        prior_pi_zt = None
        prior_mu_zt = None
        prior_logvar_zt = None
        t = t if t is not None else torch.randint(0, self.num_steps, (x.size(0),), device=x.device)
        if mask_ts is None and mask is not None:
            mask_ts = mask.any(dim=2, keepdim=True).float()
        # Embedding de entrada
        if not already_latent:
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
            pi_z_logits = self.mog_pi_z(h)
            mu_z = self.mog_mu_z(h).view(h.size(0), h.size(1), self.mog_components, -1)
            logvar_z = self.mog_logvar_z(h).view(h.size(0), h.size(1), self.mog_components, -1).clamp(min=self.vamp_logvar_min, max=self.vamp_logvar_max)
            # amostra componente com Gumbel-Softmax (straight-through) para preservar multimodalidade
            '''comp_onehot = F.gumbel_softmax(pi_z_logits, tau=1.0, hard=True, dim=-1)
            mu_sel = (comp_onehot.unsqueeze(-1) * mu_z).sum(dim=2)
            logvar_sel = (comp_onehot.unsqueeze(-1) * logvar_z).sum(dim=2)
            z_vae = mu_sel + torch.randn_like(mu_sel) * torch.exp(0.5 * logvar_sel)'''
            resp = F.softmax(pi_z_logits, dim=-1)  # (B,T,K)
            z_vae = (torch.sqrt(logvar_z.exp()) * torch.randn_like(logvar_z) + mu_z)  # (B,T,K,C
            z_vae = (resp.unsqueeze(-1) * z_vae).sum(dim=2)  # (B,T,C)
            mu_sel = (resp.unsqueeze(-1) * mu_z).sum(dim=2)
            var_second_moment = (resp.unsqueeze(-1) * (logvar_z.exp() + mu_z ** 2)).sum(dim=2)
            var_sel = var_second_moment - mu_sel ** 2
            logvar_sel = var_sel.clamp(min=1e-6).log().clamp(min=self.vamp_logvar_min, max=self.vamp_logvar_max)
            prior_pi_z_logits = self.mog_pi_z(self.vamp_pseudo_inputs).squeeze(1)
            prior_mu_z_all = self.mog_mu_z(self.vamp_pseudo_inputs).view(self.mog_components,1,self.mog_components,-1).squeeze(1)
            prior_logvar_z_all = self.mog_logvar_z(self.vamp_pseudo_inputs).view(self.mog_components,1,self.mog_components,-1).squeeze(1).clamp(min=self.vamp_logvar_min, max=self.vamp_logvar_max)
            # combina mistura dos pseudo-inputs em vez de usar apenas a diagonal/weights fixos
            prior_pi_z_probs_all = torch.softmax(prior_pi_z_logits, dim=-1)  # (K,K)
            prior_pi_z_raw = prior_pi_z_probs_all.sum(dim=0, keepdim=True)   # (1,K)
            prior_pi_z_probs = prior_pi_z_raw / prior_pi_z_raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            prior_mu_z = (
                (prior_pi_z_probs_all.unsqueeze(-1) * prior_mu_z_all).sum(dim=0, keepdim=True)
                / prior_pi_z_raw.unsqueeze(-1).clamp(min=1e-8)
            )
            prior_var_z = (
                (prior_pi_z_probs_all.unsqueeze(-1) * (prior_logvar_z_all.exp() + prior_mu_z_all ** 2)).sum(dim=0, keepdim=True)
                / prior_pi_z_raw.unsqueeze(-1).clamp(min=1e-8)
            ) - prior_mu_z ** 2
            prior_logvar_z = prior_var_z.clamp(min=1e-6).log().clamp(min=self.vamp_logvar_min, max=self.vamp_logvar_max)
            prior_pi_z = torch.log(prior_pi_z_probs.clamp(min=1e-8)).unsqueeze(1)             # (1,1,K)
            prior_mu_z = prior_mu_z.unsqueeze(1)                                              # (1,1,K,C)
            prior_logvar_z = prior_logvar_z.unsqueeze(1)                                      # (1,1,K,C)
            vae_mu = mu_sel
            vae_logvar = logvar_sel
            vae_x = self.vae_decoder(z_vae)
            vae_logvar_obs = self.vae_sigma_head(z_vae).clamp(min=-5.0, max=5.0)

        if self.lam[5] > 0 and ht is not None:
            pi_zt_logits = self.mog_pi_zt(ht)
            mu_zt = self.mog_mu_zt(ht).view(ht.size(0), ht.size(1), self.mog_components_tmax, -1)
            logvar_zt = self.mog_logvar_zt(ht).view(ht.size(0), ht.size(1), self.mog_components_tmax, -1).clamp(min=self.vamp_logvar_min, max=self.vamp_logvar_max)
            comp_onehot_t = F.gumbel_softmax(pi_zt_logits, tau=1.0, hard=True, dim=-1)
            mu_sel_t = (comp_onehot_t.unsqueeze(-1) * mu_zt).sum(dim=2)
            logvar_sel_t = (comp_onehot_t.unsqueeze(-1) * logvar_zt).sum(dim=2)
            z_tmax_lat = mu_sel_t + torch.randn_like(mu_sel_t) * torch.exp(0.5 * logvar_sel_t)
            prior_pi_zt_logits = self.mog_pi_zt(self.vamp_pseudo_inputs_tmax).squeeze(1)
            prior_mu_zt_all = self.mog_mu_zt(self.vamp_pseudo_inputs_tmax).view(self.mog_components_tmax,1,self.mog_components_tmax,-1).squeeze(1)
            prior_logvar_zt_all = self.mog_logvar_zt(self.vamp_pseudo_inputs_tmax).view(self.mog_components_tmax,1,self.mog_components_tmax,-1).squeeze(1).clamp(min=self.vamp_logvar_min, max=self.vamp_logvar_max)
            prior_pi_zt_probs_all = torch.softmax(prior_pi_zt_logits, dim=-1)  # (K_t,K_t)
            prior_pi_zt_raw = prior_pi_zt_probs_all.sum(dim=0, keepdim=True)   # (1,K_t)
            prior_pi_zt_probs = prior_pi_zt_raw / prior_pi_zt_raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            prior_mu_zt = (
                (prior_pi_zt_probs_all.unsqueeze(-1) * prior_mu_zt_all).sum(dim=0, keepdim=True)
                / prior_pi_zt_raw.unsqueeze(-1).clamp(min=1e-8)
            )
            prior_var_zt = (
                (prior_pi_zt_probs_all.unsqueeze(-1) * (prior_logvar_zt_all.exp() + prior_mu_zt_all ** 2)).sum(dim=0, keepdim=True)
                / prior_pi_zt_raw.unsqueeze(-1).clamp(min=1e-8)
            ) - prior_mu_zt ** 2
            prior_logvar_zt = prior_var_zt.clamp(min=1e-6).log().clamp(min=self.vamp_logvar_min, max=self.vamp_logvar_max)
            prior_pi_zt = torch.log(prior_pi_zt_probs.clamp(min=1e-8)).unsqueeze(1)           # (1,1,K_t)
            prior_mu_zt = prior_mu_zt.unsqueeze(1)                                            # (1,1,K_t,C)
            prior_logvar_zt = prior_logvar_zt.unsqueeze(1)                                    # (1,1,K_t,C)
            vae_tmax_mu = mu_sel_t
            vae_tmax_logvar = logvar_sel_t
            vae_tmax = self.vae_tmax_decoder(z_tmax_lat)
            vae_tmax_logvar_obs = self.vae_tmax_sigma_head(z_tmax_lat).clamp(min=-5.0, max=5.0)

        x_hat = self.decoder(h) if return_x_hat and self.lam[0]>0 else None

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
            vae_logvar_obs,
            vae_tmax_logvar_obs,
            pi_z_logits if 'pi_z_logits' in locals() else None,
            mu_z,
            logvar_z,
            prior_pi_z if 'prior_pi_z' in locals() else None,
            prior_mu_z if 'prior_mu_z' in locals() else None,
            prior_logvar_z if 'prior_logvar_z' in locals() else None,
            pi_zt_logits if 'pi_zt_logits' in locals() else None,
            mu_zt if 'mu_zt' in locals() else None,
            logvar_zt if 'logvar_zt' in locals() else None,
            prior_pi_zt if 'prior_pi_zt' in locals() else None,
            prior_mu_zt if 'prior_mu_zt' in locals() else None,
            prior_logvar_zt if 'prior_logvar_zt' in locals() else None,
        )
