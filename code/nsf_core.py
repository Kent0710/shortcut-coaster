"""
nsf_core — Reusable PyTorch implementation of Neural Statistical Functions
(Xu et al., NeurIPS 2026, arXiv:2605.11327v1, Section 4.1 & Appendix C).

Open Artifact providing:
  - Exact 11-parameter 2D dynamical system (`sample_designs`, `instantaneous_energy_np`,
    `instantaneous_energy_torch`, `trajectory_polar_np`, `analytic_interval_mean`)
  - Prefix statistical network (`NeuralStatisticalMLP`, `PrefixStatisticWrapper`)
  - Stop-gradient domain-clamped finite-difference training (`train_nsf`)
  - Reference baselines (`mc_estimate`, `midpoint_estimate`, `gauss_legendre_estimate`)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

PARAM_NAMES = (
    "a",
    "b",
    "omega0",
    "alpha",
    "phi_r",
    "phi_w",
    "phi_h",
    "mu_r",
    "mu_w",
    "gamma_r",
    "gamma_w",
)

PAPER_TABLE11_NSF = np.array([0.0220, 0.0217, 0.0214, 0.0212, 0.0210, 0.0209, 0.0208, 0.0206])
PAPER_TABLE10_NSF = np.array([0.0298, 0.0279, 0.0281, 0.0282, 0.0289, 0.0292, 0.0283, 0.0281])
PAPER_TABLE11_MID40 = np.array([0.0194, 0.0215, 0.0213, 0.0241, 0.0307, 0.0395, 0.0879, 0.1398])
PAPER_TABLE11_MC400 = np.array([0.0302, 0.0301, 0.0301, 0.0302, 0.0302, 0.0295, 0.0312, 0.0302])
PAPER_FIG4A_R2 = 0.9976


# ---------------------------------------------------------------------------
# 1. Exact 11-Parameter 2D Dynamical System (Eqs. 23-24)
# ---------------------------------------------------------------------------
def sample_designs(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample `n` independent 11-parameter 2D dynamical system designs."""
    if n < 1:
        raise ValueError(f"Expected n >= 1, got {n}")
    if rng is None:
        rng = np.random.default_rng()
    a = rng.uniform(0.25, 1.0, size=n)
    b = rng.uniform(0.25, 1.5, size=n)
    omega0 = rng.uniform(2.0 * np.pi, 6.0 * np.pi, size=n)
    alpha = rng.uniform(0.0, 0.35 * omega0 / (2.0 * np.pi))
    phi_r = rng.uniform(0.0, 2.0 * np.pi, size=n)
    phi_w = rng.uniform(0.0, 2.0 * np.pi, size=n)
    phi_h = rng.uniform(0.0, 2.0 * np.pi, size=n)
    mu_r = rng.uniform(0.08, 0.92, size=n)
    mu_w = rng.uniform(0.08, 0.92, size=n)
    gamma_r = rng.uniform(0.65, 1.35, size=n)
    gamma_w = rng.uniform(0.65, 1.35, size=n)
    return np.stack(
        [a, b, omega0, alpha, phi_r, phi_w, phi_h, mu_r, mu_w, gamma_r, gamma_w],
        axis=-1,
    ).astype(np.float64)


def instantaneous_energy_np(params: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Exact instantaneous kinetic energy h(x, s) = r'(s)^2 + (r(s) * eta'(s))^2."""
    a, b, omega0, alpha = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
    phi_r, phi_w, phi_h = params[..., 4], params[..., 5], params[..., 6]
    mu_r, mu_w = params[..., 7], params[..., 8]
    gamma_r, gamma_w = params[..., 9], params[..., 10]

    sigma = 0.008
    exp_r = np.exp(-0.5 * ((s - mu_r) / sigma) ** 2)
    r = a + b * s + 0.1 * np.sin(2.0 * np.pi * s + phi_r) + 0.25 * gamma_r * exp_r
    r_dot = (
        b
        + 0.2 * np.pi * np.cos(2.0 * np.pi * s + phi_r)
        - 0.25 * gamma_r * ((s - mu_r) / (sigma**2)) * exp_r
    )

    exp_w = np.exp(-0.5 * ((s - mu_w) / sigma) ** 2)
    eta_dot = (
        omega0
        + 2.0 * np.pi * alpha * np.cos(2.0 * np.pi * s + phi_w)
        + 1.5 * gamma_w * np.sin(80.0 * np.pi * s + phi_h)
        + 8.0 * gamma_w * exp_w
    )
    return r_dot**2 + (r * eta_dot) ** 2


def trajectory_polar_np(
    params_1d: np.ndarray, s_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D Cartesian spiral path (x(s), y(s)), radius r(s), angular velocity eta'(s),
    and instantaneous energy h(x, s) for a single 11-parameter design over sorted s_grid.
    """
    p = np.asarray(params_1d, dtype=np.float64).reshape(11)
    s = np.asarray(s_grid, dtype=np.float64)
    a, b, omega0, alpha, phi_r, phi_w, phi_h, mu_r, mu_w, gamma_r, gamma_w = p

    sigma = 0.008
    exp_r = np.exp(-0.5 * ((s - mu_r) / sigma) ** 2)
    r = a + b * s + 0.1 * np.sin(2.0 * np.pi * s + phi_r) + 0.25 * gamma_r * exp_r
    r_dot = (
        b
        + 0.2 * np.pi * np.cos(2.0 * np.pi * s + phi_r)
        - 0.25 * gamma_r * ((s - mu_r) / (sigma**2)) * exp_r
    )
    exp_w = np.exp(-0.5 * ((s - mu_w) / sigma) ** 2)
    eta_dot = (
        omega0
        + 2.0 * np.pi * alpha * np.cos(2.0 * np.pi * s + phi_w)
        + 1.5 * gamma_w * np.sin(80.0 * np.pi * s + phi_h)
        + 8.0 * gamma_w * exp_w
    )
    energy = r_dot**2 + (r * eta_dot) ** 2

    # Cumulative trapezoidal integration of eta'(s) to get angle eta(s) with eta(0) = 0
    ds = np.diff(s)
    eta = np.zeros_like(s)
    eta[1:] = np.cumsum(0.5 * (eta_dot[:-1] + eta_dot[1:]) * ds)
    x = r * np.cos(eta)
    y = r * np.sin(eta)
    return x, y, r, eta_dot, energy


def instantaneous_energy_torch(params: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Vectorized exact instantaneous kinetic energy in PyTorch (params: [B, 11], s: [B])."""
    a, b, omega0, alpha = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    phi_r, phi_w, phi_h = params[:, 4], params[:, 5], params[:, 6]
    mu_r, mu_w = params[:, 7], params[:, 8]
    gamma_r, gamma_w = params[:, 9], params[:, 10]

    sigma = 0.008
    two_pi = 2.0 * math.pi
    exp_r = torch.exp(-0.5 * ((s - mu_r) / sigma) ** 2)
    r = a + b * s + 0.1 * torch.sin(two_pi * s + phi_r) + 0.25 * gamma_r * exp_r
    r_dot = (
        b
        + 0.2 * math.pi * torch.cos(two_pi * s + phi_r)
        - 0.25 * gamma_r * ((s - mu_r) / (sigma**2)) * exp_r
    )

    exp_w = torch.exp(-0.5 * ((s - mu_w) / sigma) ** 2)
    eta_dot = (
        omega0
        + two_pi * alpha * torch.cos(two_pi * s + phi_w)
        + 1.5 * gamma_w * torch.sin(80.0 * math.pi * s + phi_h)
        + 8.0 * gamma_w * exp_w
    )
    return r_dot**2 + (r * eta_dot) ** 2


def analytic_interval_mean(
    params: np.ndarray, s0: float, s1: float, n_panels: int = 512
) -> np.ndarray:
    """High-precision composite 8-point Gauss-Legendre interval mean (~1e-12 accuracy)."""
    if not (0.0 <= s0 < s1 <= 1.0):
        raise ValueError(f"Expected 0 <= s0 < s1 <= 1, got [{s0}, {s1}]")
    if n_panels < 1:
        raise ValueError(f"Expected n_panels >= 1, got {n_panels}")
    params_2d = np.atleast_2d(np.asarray(params, dtype=np.float64))
    gl_x, gl_w = np.polynomial.legendre.leggauss(8)
    edges = np.linspace(s0, s1, n_panels + 1)
    left, right = edges[:-1, None], edges[1:, None]
    pts = (0.5 * (right - left) * gl_x[None, :] + 0.5 * (right + left)).reshape(-1)
    wts = (0.5 * (right - left) * gl_w[None, :] / (s1 - s0)).reshape(-1)
    vals = instantaneous_energy_np(params_2d[:, None, :], pts[None, :])
    return vals @ wts


def mc_estimate(
    params: np.ndarray,
    s0: float,
    s1: float,
    k: int = 50,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IID Monte Carlo interval mean estimate using `k` uniform samples on [s0, s1].
    Returns (means, sample_s, sample_energies).
    """
    if not (0.0 <= s0 < s1 <= 1.0):
        raise ValueError(f"Expected 0 <= s0 < s1 <= 1, got [{s0}, {s1}]")
    if k < 1:
        raise ValueError(f"Expected k >= 1, got {k}")
    if rng is None:
        rng = np.random.default_rng()
    params_2d = np.atleast_2d(np.asarray(params, dtype=np.float64))
    sample_s = rng.uniform(s0, s1, size=(len(params_2d), k))
    sample_e = instantaneous_energy_np(params_2d[:, None, :], sample_s)
    return sample_e.mean(axis=1), sample_s, sample_e


def midpoint_estimate(
    params: np.ndarray, s0: float, s1: float, n_points: int = 40
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite Midpoint quadrature rule with `n_points` equispaced nodes on [s0, s1]."""
    if not (0.0 <= s0 < s1 <= 1.0):
        raise ValueError(f"Expected 0 <= s0 < s1 <= 1, got [{s0}, {s1}]")
    if n_points < 1:
        raise ValueError(f"Expected n_points >= 1, got {n_points}")
    params_2d = np.atleast_2d(np.asarray(params, dtype=np.float64))
    nodes = s0 + (s1 - s0) * (np.arange(n_points, dtype=np.float64) + 0.5) / float(n_points)
    vals = instantaneous_energy_np(params_2d[:, None, :], nodes[None, :])
    return vals.mean(axis=1), nodes, vals


def gauss_legendre_estimate(
    params: np.ndarray, s0: float, s1: float, n_points: int = 40
) -> np.ndarray:
    """Gauss-Legendre quadrature rule with `n_points` nodes on [s0, s1]."""
    if not (0.0 <= s0 < s1 <= 1.0):
        raise ValueError(f"Expected 0 <= s0 < s1 <= 1, got [{s0}, {s1}]")
    params_2d = np.atleast_2d(np.asarray(params, dtype=np.float64))
    gl_x, gl_w = np.polynomial.legendre.leggauss(n_points)
    nodes = 0.5 * (s1 - s0) * gl_x + 0.5 * (s1 + s0)
    vals = instantaneous_energy_np(params_2d[:, None, :], nodes[None, :])
    return vals @ (0.5 * gl_w)


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """Global relative L2 error ||pred - ref||_2 / ||ref||_2."""
    pred_a = np.asarray(pred, dtype=np.float64)
    ref_a = np.asarray(ref, dtype=np.float64)
    ref_norm = float(np.linalg.norm(ref_a))
    if not (math.isfinite(ref_norm) and ref_norm > 0.0):
        raise ValueError(f"Reference must have finite non-zero norm, got {ref_norm}")
    return float(np.linalg.norm(pred_a - ref_a) / ref_norm)


# ---------------------------------------------------------------------------
# 2. Neural Statistical Function Architecture & Wrapper
# ---------------------------------------------------------------------------
class NeuralStatisticalMLP(nn.Module):
    """4-hidden-layer width-512 SiLU MLP mapping (x_norm, s) -> normalized prefix mean."""

    def __init__(self, in_dim: int = 12, width: int = 512, depth: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.width = width
        self.depth = depth
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_feat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_feat, s.unsqueeze(-1)], dim=-1)
        return self.net(inp).squeeze(-1)


def clamped_central_fd_s(
    model: nn.Module, x_feat: torch.Tensor, s: torch.Tensor, delta_s: float = 1e-3
) -> torch.Tensor:
    """Domain-clamped finite-difference estimate of d/ds h_tilde(x, s) on [0, 1]."""
    s_plus = (s + delta_s).clamp(0.0, 1.0)
    s_minus = (s - delta_s).clamp(0.0, 1.0)
    return (model(x_feat, s_plus) - model(x_feat, s_minus)) / (s_plus - s_minus).clamp(min=1e-12)


class PrefixStatisticWrapper:
    """
    Inference wrapper around a trained `NeuralStatisticalMLP` storing input/target
    normalization statistics and implementing exact 2-evaluation interval subtraction.
    """

    def __init__(
        self,
        model: NeuralStatisticalMLP,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
        y_mean: float,
        y_std: float,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.feat_mean = feat_mean.to(dtype=torch.float32, device=self.device)
        self.feat_std = feat_std.to(dtype=torch.float32, device=self.device)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

    @torch.no_grad()
    def predict_prefix(self, params_np: np.ndarray, s_vals: np.ndarray | float) -> np.ndarray:
        """Evaluate the learned prefix statistic h_tilde(x, s) at query points s in [0, 1]."""
        params_2d = np.atleast_2d(np.asarray(params_np, dtype=np.float32))
        if params_2d.ndim != 2 or params_2d.shape[1] != 11:
            raise ValueError(f"Expected params_np with shape (N, 11), got {params_2d.shape}")
        n = len(params_2d)
        if np.ndim(s_vals) == 0:
            s_arr = np.full(n, float(s_vals), dtype=np.float32)
        else:
            s_arr = np.asarray(s_vals, dtype=np.float32)
            if s_arr.shape != (n,):
                raise ValueError(f"Expected s_vals shape ({n},), got {s_arr.shape}")
        x_t = torch.from_numpy(params_2d).to(self.device)
        feat = (x_t - self.feat_mean) / self.feat_std
        s_t = torch.from_numpy(s_arr).to(self.device)
        out = self.model(feat, s_t).cpu().numpy().astype(np.float64)
        return out * self.y_std + self.y_mean

    @torch.no_grad()
    def predict_interval(self, params_np: np.ndarray, s0: float, s1: float) -> np.ndarray:
        """
        Evaluate the interval mean over [s0, s1] in strictly 2 prefix evaluations
        (1 evaluation when s0 == 0.0 via the exact boundary condition s0 * h_tilde(x, s0) = 0).
        """
        if not (0.0 <= s0 < s1 <= 1.0):
            raise ValueError(f"Expected 0 <= s0 < s1 <= 1, got [{s0}, {s1}]")
        h1 = self.predict_prefix(params_np, float(s1))
        if s0 == 0.0:
            return h1
        h0 = self.predict_prefix(params_np, float(s0))
        return (s1 * h1 - s0 * h0) / (s1 - s0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "in_dim": self.model.in_dim,
                "width": self.model.width,
                "depth": self.model.depth,
                "feat_mean": self.feat_mean.cpu(),
                "feat_std": self.feat_std.cpu(),
                "y_mean": self.y_mean,
                "y_std": self.y_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "PrefixStatisticWrapper":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = NeuralStatisticalMLP(
            in_dim=int(ckpt.get("in_dim", 12)),
            width=int(ckpt.get("width", 512)),
            depth=int(ckpt.get("depth", 4)),
        )
        model.load_state_dict(ckpt["state_dict"])
        return cls(
            model=model,
            feat_mean=ckpt["feat_mean"],
            feat_std=ckpt["feat_std"],
            y_mean=float(ckpt["y_mean"]),
            y_std=float(ckpt["y_std"]),
            device=device,
        )


# ---------------------------------------------------------------------------
# 3. Training (Algorithm 1: Stop-Gradient Clamped Finite-Difference)
# ---------------------------------------------------------------------------
def train_nsf(
    train_params_np: np.ndarray,
    epochs: int = 2000,
    batch_size: int = 1024,
    width: int = 512,
    depth: int = 4,
    max_lr: float = 1e-3,
    weight_decay: float = 1e-2,
    pct_start: float = 0.3,
    delta_s: float = 1e-3,
    seed: int = 0,
    device: torch.device | str = "cpu",
    log_fn: Callable[[str], None] | None = None,
) -> PrefixStatisticWrapper:
    """Train a Neural Statistical Function using Algorithm 1 (arXiv:2605.11327v1)."""
    dev = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_params = torch.tensor(train_params_np, dtype=torch.float32, device=dev)
    feat_mean = train_params.mean(dim=0, keepdim=True)
    feat_std = train_params.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_feat = (train_params - feat_mean) / feat_std

    with torch.no_grad():
        s_grid = torch.linspace(0.0, 1.0, 64, device=dev, dtype=torch.float32)
        h_grid = instantaneous_energy_torch(
            train_params.unsqueeze(1).expand(-1, 64, -1).reshape(-1, 11),
            s_grid.unsqueeze(0).expand(len(train_params), -1).reshape(-1),
        )
        y_mean = float(h_grid.mean().item())
        y_std = float(max(h_grid.std().item(), 1e-6))

    model = NeuralStatisticalMLP(in_dim=12, width=width, depth=depth).to(dev)
    n_train = len(train_params)
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = epochs * steps_per_epoch

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
    )

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            p_b = train_params[idx]
            f_b = train_feat[idx]
            s_b = torch.rand(len(idx), device=dev).clamp(min=1e-4, max=1.0)
            h_true = instantaneous_energy_torch(p_b, s_b)
            psi_norm = (h_true - y_mean) / y_std
            h_tilde = model(f_b, s_b)
            with torch.no_grad():
                dh_ds = clamped_central_fd_s(model, f_b, s_b, delta_s)
                h_tgt = psi_norm - s_b * dh_ds
            loss = torch.mean((h_tilde - h_tgt) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if log_fn is not None and (epoch == 1 or epoch % 500 == 0 or epoch == epochs):
            log_fn(f"Epoch {epoch:4d}/{epochs} | loss={epoch_loss / n_batches:.6f}")

    return PrefixStatisticWrapper(
        model=model,
        feat_mean=feat_mean,
        feat_std=feat_std,
        y_mean=y_mean,
        y_std=y_std,
        device=dev,
    )
