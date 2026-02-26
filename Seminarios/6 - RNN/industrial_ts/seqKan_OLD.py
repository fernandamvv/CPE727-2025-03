from __future__ import annotations

from pathlib import Path
import sys
import warnings
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Subset
import numpy as np
from sklearn.preprocessing import RobustScaler
import pandas as pd

from .ode_jump import ODEJump, TS_SPAN
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

        # Top-k sparsity (activation-level) config
        topk_cfg = kan_params.get("topk", {}) if isinstance(kan_params, dict) else {}
        self.topk_enabled = bool(topk_cfg.get("enabled", False))
        self.topk_k_by_target = topk_cfg.get("k_by_target", {})
        self.topk_warmup_epochs = int(topk_cfg.get("warmup_epochs", 0) or 0)
        self.topk_targets = set(topk_cfg.get("targets", ["s1", "h_rec"]))
        self.topk_mode = str(topk_cfg.get("mode", "hard")).lower()
        self.topk_temp = float(topk_cfg.get("temp", 0.1))
        self.last_topk_ratio = {}
        self.current_epoch = None

    def _apply_topk(self, x, name: str | None = None):
        if not self.topk_enabled:
            return x
        if self.current_epoch is None or self.current_epoch <= self.topk_warmup_epochs:
            return x
        k_cfg = self.topk_k_by_target.get(name, None) if name is not None else None
        if k_cfg is None:
            return x
        k = int(k_cfg)
        if k <= 0 or k >= x.size(-1):
            return x
        scores = x.abs()
        vals, idx = torch.topk(scores, k=k, dim=-1)
        # hard mask (top-k by abs)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(dim=-1, index=idx, value=True)
        if self.topk_mode == "soft":
            # soft gate around k-th threshold
            kth = vals[..., -1, None]
            gate = torch.sigmoid((scores - kth) / max(self.topk_temp, 1e-6))
            out = x * gate
            ratio = gate.mean().detach().item()
        else:
            out = torch.where(mask, x, torch.zeros_like(x))
            ratio = mask.float().mean().detach().item()
        if name is not None:
            self.last_topk_ratio[name] = ratio
        return out

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
            if "s1" in self.topk_targets:
                s1 = self._apply_topk(s1, name="s1")
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
            if "z" in self.topk_targets:
                z_state = self._apply_topk(z_state, name="z")
            # h_t com recorrência explícita via KAN_h
            h_prev = hidden_state
            h_rec = self.kan_h(h_prev)
            if "h_rec" in self.topk_targets:
                h_rec = self._apply_topk(h_rec, name="h_rec")
            hidden_state = h_rec + m * s1 + (1 - m) * z_prev
            output_t = self.kan_out(hidden_state)
            outputs.append(output_t)
        outputs = torch.stack(outputs, dim=1)
        last = outputs[:, -1, :]
        return (outputs, last) if return_last else outputs


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


class TS_seqKANSeq(nn.Module):
    """
    Wrapper com train_cognite para SeqKANSeq (previsao).
    Replica o fluxo do notebook: janelas + horizonte, loss no futuro.
    """
    def __init__(self, in_channels, hidden_dim, output_dim, kan_params=None, device=None, lam=None):
        super().__init__()
        self.in_channels = int(in_channels)
        self.output_dim = int(output_dim)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        if lam is None:
            lam = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.lam = list(lam)
        self.seq = SeqKANSeq(
            input_size=in_channels,
            hidden_size=hidden_dim,
            output_size=output_dim,
            kan_params=kan_params,
            device=self.device,
        )

    def forward(self, x, mask=None):
        return self.seq(x, mask=mask)

    def train_cognite(
        self,
        df,
        feature_cols,
        static_features_cols=None,
        timestamp_col="index",
        states_col=None,
        batch_size=256,
        window_size=10,
        window_step=1,
        epochs=50,
        validate=True,
        patience=10,
        train_fraction=0.6,
        seed_split: int = 42,
        fixed_test_idx: np.ndarray | None = None,
        use_group_sampler: bool = True,
        horizon=1,
        y_cols=None,
        rebuild=False,
        cost_columns: list | None = None,
        df_denoised: pd.DataFrame | None = None,
        reconstruction_test=False,
        optimizer_name="adam",
        optimizer_params=None,
        lam=None,
        x_min=None,
        x_max=None,
        use_robust_scaler=True,
        save_best_ckpt=True,
        ckpt_path="ts_seqkanseq.pt",
        log_every=100,
        debug_batch_stats=False,
        debug_batch_stats_names=None,
        **kwargs,
    ):
        if reconstruction_test:
            raise ValueError("TS_seqKANSeq e apenas previsao; reconstruction_test deve ser False.")
        if optimizer_params is None:
            optimizer_params = {"lr": 2e-4}
        if lam is None:
            # compatibilidade: usa rebuild flag para definir lam default
            lam = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] if rebuild else [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        lam = list(lam)
        if len(lam) < 2:
            raise ValueError("lam deve ter ao menos dois termos (rebuild e forecast).")
        use_rebuild = lam[0] > 0
        use_forecast = lam[1] > 0
        if not use_rebuild and not use_forecast:
            raise ValueError("lam requer ao menos um termo > 0 (rebuild ou forecast).")
        kan_params = getattr(self.seq, "kan_params", None)
        feature_cols = list(feature_cols)
        if y_cols is None:
            y_cols = feature_cols[: self.output_dim]
        y_cols = list(y_cols)
        if use_forecast and len(y_cols) != self.output_dim:
            raise ValueError("len(y_cols) deve bater com output_dim.")
        if use_rebuild and self.output_dim != self.in_channels and getattr(self.seq, "kan_out_rebuild", None) is None:
            # cria cabeca de rebuild (frame inteiro) mantendo a cabeca principal (y_cols)
            self.seq = SeqKANSeq(
                input_size=self.in_channels,
                hidden_size=self.seq.hidden_size,
                output_size=self.output_dim,
                kan_params=kan_params,
                device=self.device,
                output_size_rebuild=self.in_channels,
            )

        y_idx = [feature_cols.index(c) for c in y_cols] if use_forecast else list(range(self.in_channels))
        cost_mask = None
        if cost_columns is not None:
            cost_mask = np.array([1.0 if c in cost_columns else 0.0 for c in feature_cols], dtype=float)
        # --- prepara dados numéricos e trata NaN/Inf ---
        vals = df[feature_cols].to_numpy(dtype=float)
        vals[~np.isfinite(vals)] = np.nan
        # preenche NaN por interpolação + mediana (pipeline antigo)
        # usa pandas via df para preservar ordem
        df_feat = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        df_feat = df_feat.interpolate(limit_direction='both')
        df_feat = df_feat.fillna(df_feat.median())
        vals = df_feat.to_numpy(dtype=float)

        if timestamp_col != "index":
            ts_raw = pd.to_datetime(df[timestamp_col]).astype("int64") / 1e9
        else:
            ts_raw = pd.to_datetime(df.index).astype("int64") / 1e9
        t0 = ts_raw.iloc[0] if hasattr(ts_raw, "iloc") else ts_raw[0]
        ts_rel = ((ts_raw - t0) / TS_SPAN).to_numpy(dtype=np.float32)

        static = None
        if static_features_cols:
            static = df[static_features_cols].to_numpy(dtype=float)

        vals_denoised = None
        if df_denoised is not None:
            df_den = df_denoised[feature_cols].apply(pd.to_numeric, errors='coerce')
            df_den = df_den.replace([np.inf, -np.inf], np.nan)
            df_den = df_den.interpolate(limit_direction='both')
            df_den = df_den.fillna(df_den.median())
            vals_denoised = df_den.to_numpy(dtype=float)

        T, C = vals.shape
        xs, ms, ys, groups = [], [], [], []
        ts_seqs, stat_seqs, den_seqs = [], [], []
        if rebuild:
            for t in range(window_size - 1, T, window_step):
                seq = vals[t - window_size + 1 : t + 1].copy()
                mask = np.ones_like(seq)
                xs.append(seq)
                ms.append(mask)
                if not rebuild:
                    # alvo para previsao: ultimo timestep, y_cols
                    ys.append(vals[t : t + 1, y_idx])
                else:
                    # rebuild-only: target dummy (não usado), mantém shape compatível
                    ys.append(seq)
                ts_seqs.append(ts_rel[t - window_size + 1 : t + 1].copy())
                if static is not None:
                    stat_seqs.append(static[0].copy())
                if vals_denoised is not None:
                    den_seqs.append(vals_denoised[t - window_size + 1 : t + 1].copy())
                if states_col is not None and states_col in df.columns:
                    groups.append(int(df[states_col].iloc[t]))
        else:
            for t in range(window_size - 1, T - horizon, window_step):
                seq = vals[t - window_size + 1 : t + 1 + horizon].copy()
                mask = np.ones_like(seq)
                mask[-horizon:, :] = 0.0
                mask[-horizon:, y_idx] = 1.0
                seq[-horizon:, :] *= mask[-horizon:, :]
                xs.append(seq)
                ms.append(mask)
                ys.append(vals[t + 1 : t + 1 + horizon, y_idx])
                ts_seqs.append(ts_rel[t - window_size + 1 : t + 1 + horizon].copy())
                if static is not None:
                    stat_seqs.append(static[0].copy())
                if vals_denoised is not None:
                    den_seqs.append(vals_denoised[t - window_size + 1 : t + 1 + horizon].copy())
                if states_col is not None and states_col in df.columns:
                    groups.append(int(df[states_col].iloc[t]))
        x_seq = np.stack(xs)
        m_seq = np.stack(ms)
        y_future = np.stack(ys)
        ts_seq = np.stack(ts_seqs)
        stat_seq = np.stack(stat_seqs) if stat_seqs else None
        den_seq = np.stack(den_seqs) if den_seqs else None
        groups = np.asarray(groups) if groups else None

        N = x_seq.shape[0]
        indices = np.arange(N)
        if groups is not None:
            val_frac = (1.0 - train_fraction) / 2
            if fixed_test_idx is not None:
                test_idx = np.asarray(fixed_test_idx, dtype=int)
                remain_mask = np.ones(N, dtype=bool)
                remain_mask[test_idx] = False
                tr_idx_rel, va_idx_rel, _ = ODEJump._split_by_group_proportions(
                    groups[remain_mask],
                    validate=validate,
                    train_frac=train_fraction,
                    val_frac=val_frac,
                    test_frac=val_frac,
                    seed=seed_split,
                )
                base = np.where(remain_mask)[0]
                train_idx = base[tr_idx_rel]
                val_idx = base[va_idx_rel]
            else:
                train_idx, val_idx, test_idx = ODEJump._split_by_group_proportions(
                    groups,
                    validate=validate,
                    train_frac=train_fraction,
                    val_frac=val_frac,
                    test_frac=val_frac,
                    seed=seed_split,
                )
        else:
            rng = np.random.default_rng(seed_split)
            rng.shuffle(indices)
            n_tr = int(train_fraction * N)
            n_va = int(((1.0 - train_fraction) / 2) * N) if validate else 0
            train_idx = indices[:n_tr]
            val_idx = indices[n_tr:n_tr + n_va]
            test_idx = indices[n_tr + n_va:]

        if groups is not None:
            def _count_groups(idx):
                out = {}
                for g in np.unique(groups):
                    out[int(g)] = int(np.sum(groups[idx] == g))
                return out
            print(f"GRUPOS (total): {_count_groups(np.arange(len(groups)))}")
            print(f"GRUPOS (train): {_count_groups(train_idx)}")
            print(f"GRUPOS (valid): {_count_groups(val_idx) if validate else {}}")
            print(f"GRUPOS (test):  {_count_groups(test_idx)}")

        # --- RobustScaler (igual ODEJump/TSDiffusion): fit no treino e aplica em tudo ---
        if use_robust_scaler:
            x_seq_t = torch.tensor(x_seq, dtype=torch.float32)
            x_seq = ODEJump.scale_tensor(x_seq_t[train_idx], x_seq_t).numpy()
            if den_seq is not None:
                den_seq_t = torch.tensor(den_seq, dtype=torch.float32)
                den_seq = ODEJump.scale_tensor(den_seq_t[train_idx], den_seq_t).numpy()

        # clamp opcional após scaling
        if x_min is not None or x_max is not None:
            vmin = -np.inf if x_min is None else float(x_min)
            vmax = np.inf if x_max is None else float(x_max)
            x_seq = np.clip(x_seq, vmin, vmax)
            y_future = np.clip(y_future, vmin, vmax)
            if den_seq is not None:
                den_seq = np.clip(den_seq, vmin, vmax)

        x_seq = torch.tensor(x_seq, dtype=torch.float32)
        m_seq = torch.tensor(m_seq, dtype=torch.float32)
        y_future = torch.tensor(y_future, dtype=torch.float32)
        ts_seq = torch.tensor(ts_seq, dtype=torch.float32)
        stat_seq = torch.tensor(stat_seq, dtype=torch.float32) if stat_seq is not None else None
        den_seq = torch.tensor(den_seq, dtype=torch.float32) if den_seq is not None else None
        cost_mask_t = torch.tensor(cost_mask, dtype=torch.float32).view(1, 1, -1) if cost_mask is not None else None

        ds_tensors = [x_seq, ts_seq, m_seq]
        if stat_seq is not None:
            ds_tensors.append(stat_seq)
        if den_seq is not None:
            ds_tensors.append(den_seq)
        full_ds = TensorDataset(*ds_tensors, y_future)
        train_ds = Subset(full_ds, train_idx.tolist())
        val_ds = Subset(full_ds, val_idx.tolist()) if validate else None
        test_ds = Subset(full_ds, test_idx.tolist())

        train_sampler = None
        if groups is not None and use_group_sampler:
            train_sampler = ODEJump._make_weighted_sampler_from_classes(groups[train_idx])

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler if train_sampler is not None else None,
            pin_memory=True,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True) if validate else None
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
        print(f"train/val/test batches: {len(train_loader)}/{len(val_loader) if val_loader is not None else 0}/{len(test_loader)}")

        if debug_batch_stats:
            names = debug_batch_stats_names if debug_batch_stats_names is not None else feature_cols
            x_all = vals
            pct2 = float(np.mean(np.abs(x_all) > 2) * 100.0)
            pct3 = float(np.mean(np.abs(x_all) > 3) * 100.0)
            pct4 = float(np.mean(np.abs(x_all) > 4) * 100.0)
            p1, p50, p99 = np.percentile(x_all, [1, 50, 99])
            min_feat_idx = int(np.argmin(x_all.min(axis=0)))
            max_feat_idx = int(np.argmax(x_all.max(axis=0)))
            worst_min_feature = names[min_feat_idx] if min_feat_idx < len(names) else str(min_feat_idx)
            worst_max_feature = names[max_feat_idx] if max_feat_idx < len(names) else str(max_feat_idx)

            def _print_stats():
                print(f">% |x| > 2 : {pct2:.2f}%")
                print(f">% |x| > 3 : {pct3:.2f}%")
                print(f">% |x| > 4 : {pct4:.2f}%")
                print(f"min: {x_all.min():.1f} max: {x_all.max():.1f}")
                print(f"p1/p50/p99: [{p1:.6f} {p50:.6f} {p99:.6f}]")
                print(f"worst_min_feature: {worst_min_feature} {x_all[:, min_feat_idx].min():.1f}")
                print(f"worst_max_feature: {worst_max_feature} {x_all[:, max_feat_idx].max():.1f}")
        else:
            def _print_stats():
                return

        optim_name = optimizer_name.lower()
        if optim_name == "adamw":
            optimizer = torch.optim.AdamW(self.seq.parameters(), **optimizer_params)
        elif optim_name == "radam":
            optimizer = torch.optim.RAdam(self.seq.parameters(), **optimizer_params)
        else:
            optimizer = torch.optim.Adam(self.seq.parameters(), **optimizer_params)

        def _masked_l1_stats(pred, target, mask, mask_train):
            mask_err = mask * (1 - mask_train)
            err = (pred - target) ** 2
            sse = (err * mask_err).sum()
            nobs = mask_err.sum().clamp(min=1.0)
            return sse, nobs

        def _make_random_mask(xb, ts_batch, m, device, num_steps: int = 1000, max_drop: float = 0.3):
            # mesma lógica do TSDiffusion._make_random_mask
            t_mask = torch.randint(0, num_steps, (xb.size(0),), device=device)
            t_mask_ts = torch.randint(0, num_steps, (xb.size(0),), device=device)
            p_drop_t = (t_mask.float() / (num_steps - 1)) * max_drop
            p_drop_t = p_drop_t.view(-1, 1, 1)
            p_drop_ts = (t_mask_ts.float() / (num_steps - 1)) * max_drop
            p_drop_ts = p_drop_ts.view(-1, 1)
            rand_mask = (torch.rand(m.shape, device=device) > p_drop_t).float()
            rand_mask_ts = (torch.rand(ts_batch.shape, device=device) > p_drop_ts).unsqueeze(-1).float()
            rand_mask_ts[:, -1, 0] = 0
            return m * rand_mask * rand_mask_ts

        def eval_loader(loader):
            self.seq.eval()
            losses = []
            horizon_losses = []
            mse_samples = []
            with torch.no_grad():
                for batch in loader:
                    xb, tsb, mb = batch[0], batch[1], batch[2]
                    idx = 3
                    stat = batch[idx] if static_features_cols else None
                    if static_features_cols:
                        idx += 1
                    den = batch[idx] if df_denoised is not None else None
                    if df_denoised is not None:
                        idx += 1
                    yf = batch[idx]
                    xb = xb.to(self.device)
                    tsb = tsb.to(self.device)
                    mb = mb.to(self.device)
                    yf = yf.to(self.device)
                    if stat is not None:
                        stat = stat.to(self.device)
                    if den is not None:
                        den = den.to(self.device)
                    cm = cost_mask_t.to(self.device) if cost_mask_t is not None else None
                    if use_rebuild:
                        m_train = _make_random_mask(xb, tsb, mb, xb.device)
                    else:
                        m_train = mb.clone()
                        m_train[:, -1:, :] = 0
                    pred_out = self.seq(xb, mask=m_train)
                    if isinstance(pred_out, tuple):
                        pred_main, pred_rebuild = pred_out
                    else:
                        pred_main, pred_rebuild = pred_out, None

                    loss_val = 0.0
                    primary_pred = None
                    primary_target = None
                    primary_mask = None
                    primary_mtrain = None

                    if use_rebuild:
                        pred_r = pred_rebuild if pred_rebuild is not None else pred_main
                        m_r = mb * cm if cm is not None else mb
                        mtr_r = m_train * cm if cm is not None else m_train
                        sse_r, nobs_r = _masked_l1_stats(pred_r, xb, m_r, mtr_r)
                        loss_val += lam[0] * float(sse_r / nobs_r)
                        primary_pred = pred_r
                        primary_target = xb
                        primary_mask = m_r
                        primary_mtrain = mtr_r

                    if use_forecast:
                        pred_f = pred_main
                        target_f = xb[:, :, y_idx]
                        mask_f = mb[:, :, y_idx]
                        if cm is not None:
                            mask_f = mask_f * cm[:, :, y_idx]
                        mtrain_f = mask_f.clone()
                        mtrain_f[:, -1:, :] = 0
                        sse_f, nobs_f = _masked_l1_stats(pred_f, target_f, mask_f, mtrain_f)
                        loss_val += lam[1] * float(sse_f / nobs_f)
                        if primary_pred is None:
                            primary_pred = pred_f
                            primary_target = target_f
                            primary_mask = mask_f
                            primary_mtrain = mtrain_f

                    losses.append((float(loss_val), 1))
                    err = (primary_pred - primary_target) ** 2
                    step_sse = (err * (primary_mask * (1 - primary_mtrain))).sum(dim=2)
                    step_nobs = (primary_mask * (1 - primary_mtrain)).sum(dim=2).clamp(min=1.0)
                    step_mse = (step_sse / step_nobs).mean(dim=0).detach().cpu().numpy()
                    horizon_losses.append(step_mse)
                    sample_sse = step_sse.sum(dim=1)
                    sample_nobs = step_nobs.sum(dim=1).clamp(min=1.0)
                    mse = (sample_sse / sample_nobs).detach().cpu().numpy()
                    mse_samples.append(mse)
            h = np.stack(horizon_losses).mean(axis=0) if horizon_losses else None
            mse_samples = np.concatenate(mse_samples) if mse_samples else np.array([])
            if losses:
                sse = sum(x[0] for x in losses)
                nobs = sum(x[1] for x in losses)
                return float(sse / max(nobs, 1)), h, mse_samples
            return float("nan"), h, mse_samples

        def _group_metrics(mse_per_sample, groups):
            groups = np.asarray(groups)
            per_group = {}
            per_group_se = {}
            per_group_counts = {}
            for g in np.unique(groups):
                vals = mse_per_sample[groups == g]
                per_group[int(g)] = float(np.mean(vals))
                per_group_counts[int(g)] = int(len(vals))
                if len(vals) >= 2:
                    per_group_se[int(g)] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                else:
                    per_group_se[int(g)] = float("nan")
            macro = float(np.mean(list(per_group.values()))) if per_group else float("nan")
            micro = float(np.mean(mse_per_sample)) if len(mse_per_sample) else float("nan")
            micro_se = float(np.std(mse_per_sample, ddof=1) / np.sqrt(len(mse_per_sample))) if len(mse_per_sample) >= 2 else float("nan")
            return macro, micro, micro_se, per_group, per_group_se, per_group_counts

        def _macro_se_from_groups(per_group_vals):
            vals = [v for v in per_group_vals.values() if np.isfinite(v)]
            if len(vals) < 2:
                return float("nan")
            return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

        def _format_per_group(per_group, per_group_se, per_group_counts):
            if not per_group:
                return {}
            out = {}
            for g, v in per_group.items():
                se = per_group_se.get(g, float("nan"))
                cnt = per_group_counts.get(g, 0)
                out[g] = f"{v:.6f} ± {se:.6f} (n={cnt})"
            return out

        best_val = float("inf")
        best_epoch = None
        wait = patience
        for ep in range(1, epochs + 1):
            self.seq.current_epoch = ep
            self.seq.train()
            losses = []
            _print_stats()
            for step, batch in enumerate(train_loader, 1):
                xb, tsb, mb = batch[0], batch[1], batch[2]
                idx = 3
                stat = batch[idx] if static_features_cols else None
                if static_features_cols:
                    idx += 1
                den = batch[idx] if df_denoised is not None else None
                if df_denoised is not None:
                    idx += 1
                yf = batch[idx]
                xb = xb.to(self.device)
                tsb = tsb.to(self.device)
                mb = mb.to(self.device)
                yf = yf.to(self.device)
                if stat is not None:
                    stat = stat.to(self.device)
                if den is not None:
                    den = den.to(self.device)
                cm = cost_mask_t.to(self.device) if cost_mask_t is not None else None
                if use_rebuild:
                    m_train = _make_random_mask(xb, tsb, mb, xb.device)
                else:
                    m_train = mb.clone()
                    m_train[:, -1:, :] = 0
                pred_out = self.seq(xb, mask=m_train)
                if isinstance(pred_out, tuple):
                    pred_main, pred_rebuild = pred_out
                else:
                    pred_main, pred_rebuild = pred_out, None

                loss = torch.tensor(0.0, device=xb.device, dtype=xb.dtype)
                if use_rebuild:
                    pred_r = pred_rebuild if pred_rebuild is not None else pred_main
                    m_r = mb * cm if cm is not None else mb
                    mtr_r = m_train * cm if cm is not None else m_train
                    sse_r, nobs_r = _masked_l1_stats(pred_r, xb, m_r, mtr_r)
                    loss = loss + lam[0] * (sse_r / nobs_r)
                if use_forecast:
                    pred_f = pred_main
                    target_f = xb[:, :, y_idx]
                    mask_f = mb[:, :, y_idx]
                    if cm is not None:
                        mask_f = mask_f * cm[:, :, y_idx]
                    mtrain_f = mask_f.clone()
                    mtrain_f[:, -1:, :] = 0
                    sse_f, nobs_f = _masked_l1_stats(pred_f, target_f, mask_f, mtrain_f)
                    loss = loss + lam[1] * (sse_f / nobs_f)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.seq.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append((float(loss.detach().item()), 1))
                if log_every and step % int(log_every) == 0:
                    sse_sum = sum(x[0] for x in losses)
                    nobs_sum = sum(x[1] for x in losses)
                    mean_loss = float(sse_sum / max(nobs_sum, 1))
                    print(f"Epoch {ep}/{epochs} | step {step}/{len(train_loader)} | train {mean_loss:.6f}")
            if losses:
                sse_sum = sum(x[0] for x in losses)
                nobs_sum = sum(x[1] for x in losses)
                train_loss = float(sse_sum / max(nobs_sum, 1))
            else:
                train_loss = float("nan")

            if val_loader is not None:
                val_loss, val_h, val_mse_samples = eval_loader(val_loader)
            else:
                val_loss, val_h, val_mse_samples = float("nan"), None, np.array([])

            test_loss, test_h, mse_samples = eval_loader(test_loader)
            if groups is not None:
                macro, micro, micro_se, per_group, per_group_se, per_group_counts = _group_metrics(mse_samples, groups[test_idx])
                macro_se = _macro_se_from_groups(per_group)
            else:
                macro = float("nan")
                micro = float(np.mean(mse_samples)) if len(mse_samples) else float("nan")
                micro_se = float(np.std(mse_samples, ddof=1) / np.sqrt(len(mse_samples))) if len(mse_samples) >= 2 else float("nan")
                macro_se = float("nan")
                per_group = {}
                per_group_se = {}
                per_group_counts = {}

            if val_loader is not None and len(val_mse_samples):
                if groups is not None:
                    val_macro, val_micro, val_micro_se, val_pg, _, _ = _group_metrics(val_mse_samples, groups[val_idx])
                    val_macro_se = _macro_se_from_groups(val_pg)
                else:
                    val_macro = float("nan")
                    val_micro = float(np.mean(val_mse_samples))
                    val_micro_se = float(np.std(val_mse_samples, ddof=1) / np.sqrt(len(val_mse_samples))) if len(val_mse_samples) >= 2 else float("nan")
                    val_macro_se = float("nan")
            else:
                val_macro = float("nan")
                val_micro = float("nan")
                val_micro_se = float("nan")
                val_macro_se = float("nan")

            print(
                f"Epoch {ep}/{epochs} | Train(sampled) L1:{train_loss:.6f} L2:0.000000 L3:0.000000  L4:0.000000 L5:0.000000 L6:0.000000 | "
                f"Val macro:{val_macro:.6f} ± {val_macro_se:.6f} | Val micro:{val_micro:.6f} ± {val_micro_se:.6f}"
            )
            print(
                f"          >> Test macro:{macro:.6f} ± {macro_se:.6f} | micro:{micro:.6f} ± {micro_se:.6f}"
            )
            if groups is not None:
                print(f"          >> per_group (weighted SE): {_format_per_group(per_group, per_group_se, per_group_counts)}")

            yield {
                "epoch": ep,
                "train_loss": train_loss,
                "train_L1": train_loss,
                "val_loss": val_loss,
                "val_h": val_h,
                "test_loss": test_loss,
                "test_h": test_h,
                "micro_mse": micro,
                "micro_se": micro_se,
                "macro_mse": macro,
                "macro_se": macro_se,
                "per_group_mse": per_group,
                "per_group_se_w": per_group_se,
                "per_group_counts": per_group_counts,
                "val_micro": val_micro,
                "val_macro": val_macro,
            }

            # early stopping by val_micro (mesmo criterio do zstate)
            score = val_micro if val_loader is not None else micro
            if score < best_val:
                best_val = score
                best_epoch = ep
                wait = patience
                if save_best_ckpt:
                    torch.save(self.seq.state_dict(), ckpt_path)
            else:
                wait -= 1
                if wait <= 0:
                    break

        yield None

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
