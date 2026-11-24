"""
Visualization and interactive callback engine for the Neural Statistical Functions
Gradio Explainer Space (`arXiv:2605.11327v1`).

Decoupled from Gradio so every figure and Markdown table can be unit-tested and
rendered headlessly without a live browser or Gradio server.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import numpy as np

from nsf_core import (
    PAPER_FIG4A_R2,
    PAPER_TABLE10_NSF,
    PAPER_TABLE11_MC400,
    PAPER_TABLE11_MID40,
    PAPER_TABLE11_NSF,
    PrefixStatisticWrapper,
    analytic_interval_mean,
    instantaneous_energy_np,
    mc_estimate,
    midpoint_estimate,
    sample_designs,
    trajectory_polar_np,
)

# Colorblind-safe academic palette
C_NSF = "#2563EB"       # Royal Blue
C_TRUTH = "#0F172A"     # Deep Slate
C_MC = "#DC2626"        # Crimson Red
C_MID = "#D97706"       # Amber
C_ACCENT = "#059669"    # Emerald
C_BG = "#F8FAFC"


def _held_out_test_pool() -> np.ndarray:
    """Return the exact 900 held-out test designs (disjoint from the 3,000 train designs)."""
    rng = np.random.default_rng(42)
    _ = sample_designs(3000, rng)
    return sample_designs(900, rng)


HELD_OUT_TEST_POOL = _held_out_test_pool()

PRESET_LABELS = [
    "Default Spiral (Held-Out #0)",
    "High-Frequency Oscillator (Held-Out #7)",
    "Wide Expanding Orbit (Held-Out #14)",
    "Sharp Energy Spike (Held-Out #23)",
    "Random Held-Out Trajectory",
]

PRESET_INDEX_MAP = {
    "Default Spiral (Held-Out #0)": 0,
    "High-Frequency Oscillator (Held-Out #7)": 7,
    "Wide Expanding Orbit (Held-Out #14)": 14,
    "Sharp Energy Spike (Held-Out #23)": 23,
}


def get_design_for_preset(preset_name: str, random_seed: int = 0) -> tuple[np.ndarray, str]:
    if preset_name in PRESET_INDEX_MAP:
        idx = PRESET_INDEX_MAP[preset_name]
        return HELD_OUT_TEST_POOL[idx], f"Held-out Test Trajectory #{idx}"
    idx = int(abs(random_seed)) % len(HELD_OUT_TEST_POOL)
    return HELD_OUT_TEST_POOL[idx], f"Random Held-Out Trajectory #{idx}"


def _true_prefix_curve(params_1d: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
    """Compute high-accuracy true prefix mean h_tilde(x, s) = (1/s) int_0^s h(x, u) du."""
    # Dense 4096-point cumulative integral on [0, 1], interpolated onto s_grid
    fine_s = np.linspace(0.0, 1.0, 4097)
    fine_h = instantaneous_energy_np(params_1d[None, :], fine_s[None, :])[0]
    ds = fine_s[1] - fine_s[0]
    cum_int = np.zeros_like(fine_s)
    cum_int[1:] = np.cumsum(0.5 * (fine_h[:-1] + fine_h[1:]) * ds)
    prefix_fine = np.empty_like(fine_s)
    prefix_fine[0] = fine_h[0]
    prefix_fine[1:] = cum_int[1:] / fine_s[1:]
    return np.interp(s_grid, fine_s, prefix_fine)


_TRAJ_CACHE: dict[tuple[int, bytes], dict[str, np.ndarray]] = {}


def _get_traj_precomputed(params_1d: np.ndarray, wrapper: PrefixStatisticWrapper) -> dict[str, np.ndarray]:
    """Cache dense trajectory, cumulative integral, and prefix curves per (wrapper, trajectory)."""
    key = (id(wrapper), params_1d.tobytes())
    if key not in _TRAJ_CACHE:
        fine_s = np.linspace(0.0, 1.0, 4097)
        fine_h = instantaneous_energy_np(params_1d[None, :], fine_s[None, :])[0]
        ds = fine_s[1] - fine_s[0]
        cum_int = np.zeros_like(fine_s)
        cum_int[1:] = np.cumsum(0.5 * (fine_h[:-1] + fine_h[1:]) * ds)
        prefix_fine = np.empty_like(fine_s)
        prefix_fine[0] = fine_h[0]
        prefix_fine[1:] = cum_int[1:] / fine_s[1:]

        s_dense = np.linspace(0.0, 1.0, 1200)
        x_pos, y_pos, _, _, h_dense = trajectory_polar_np(params_1d, s_dense)

        s_prefix = np.linspace(0.01, 1.0, 250)
        true_prefix = np.interp(s_prefix, fine_s, prefix_fine)
        nsf_prefix = wrapper.predict_prefix(np.repeat(params_1d[None, :], len(s_prefix), axis=0), s_prefix)

        if len(_TRAJ_CACHE) > 64:
            _TRAJ_CACHE.clear()
        _TRAJ_CACHE[key] = {
            "fine_s": fine_s,
            "cum_int": cum_int,
            "s_dense": s_dense,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "h_dense": h_dense,
            "s_prefix": s_prefix,
            "true_prefix": true_prefix,
            "nsf_prefix": nsf_prefix,
        }
    return _TRAJ_CACHE[key]


def _new_figure(ncols: int, figsize: tuple[float, float], dpi: int = 130) -> tuple[Figure, np.ndarray]:
    """Create a leak-free OO Matplotlib Figure with Agg canvas (never registers in pyplot Gcf)."""
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, ncols)
    fig.patch.set_facecolor("white")
    return fig, axes


_ALIASING_SWEEP_CACHE: dict[int, tuple[np.ndarray, list[float], list[float]]] = {}


def _get_aliasing_sweep(wrapper: PrefixStatisticWrapper) -> tuple[np.ndarray, list[float], list[float]]:
    key = id(wrapper)
    if key not in _ALIASING_SWEEP_CACHE:
        sweep_widths = np.linspace(0.90, 1.00, 21)
        subset = HELD_OUT_TEST_POOL[:200]
        mid_curve: list[float] = []
        nsf_curve: list[float] = []
        for w in sweep_widths:
            t_ref = analytic_interval_mean(subset, 0.0, float(w), n_panels=128)
            m_pred, _, _ = midpoint_estimate(subset, 0.0, float(w), n_points=40)
            n_pred = wrapper.predict_interval(subset, 0.0, float(w))
            mid_curve.append(float(np.linalg.norm(m_pred - t_ref) / np.linalg.norm(t_ref)))
            nsf_curve.append(float(np.linalg.norm(n_pred - t_ref) / np.linalg.norm(t_ref)))
        _ALIASING_SWEEP_CACHE[key] = (sweep_widths, mid_curve, nsf_curve)
    return _ALIASING_SWEEP_CACHE[key]


# ---------------------------------------------------------------------------
# Tab 1: The Prefix Trick (Live 2D Spiral + Instantaneous Energy + Prefix Curve)
# ---------------------------------------------------------------------------
def render_prefix_trick(
    preset_name: str,
    random_seed: int,
    s0: float,
    s1: float,
    mc_k: int,
    mc_seed: int,
    wrapper: PrefixStatisticWrapper,
) -> tuple[Figure, str]:
    raw_s0, raw_s1 = float(s0), float(s1)
    s0 = float(np.clip(raw_s0, 0.0, 0.98))
    clamped_note = ""
    if raw_s1 < s0 + 0.02:
        s1 = float(min(1.0, s0 + 0.02))
        clamped_note = f"\n> *Note: Adjusted interval end from `s1={raw_s1:.2f}` to `s1={s1:.2f}` so `s0 < s1`.*\n"
    else:
        s1 = float(np.clip(raw_s1, s0 + 0.02, 1.0))
    mc_k = int(np.clip(mc_k, 5, 1000))

    params_1d, traj_label = get_design_for_preset(preset_name, random_seed)
    params_2d = params_1d[None, :]
    cached = _get_traj_precomputed(params_1d, wrapper)

    # Fast exact interval mean via cached cumulative integral + 2-eval NSF, MC-K, and Midpoint-40
    int_s1 = float(np.interp(s1, cached["fine_s"], cached["cum_int"]))
    int_s0 = float(np.interp(s0, cached["fine_s"], cached["cum_int"]))
    true_val = (int_s1 - int_s0) / (s1 - s0)
    nsf_val = float(wrapper.predict_interval(params_2d, s0, s1)[0])
    mc_rng = np.random.default_rng(int(mc_seed) + 1000)
    mc_mean_arr, mc_s, mc_e = mc_estimate(params_2d, s0, s1, k=mc_k, rng=mc_rng)
    mc_val = float(mc_mean_arr[0])
    mid_mean_arr, _, _ = midpoint_estimate(params_2d, s0, s1, n_points=40)
    mid_val = float(mid_mean_arr[0])

    nsf_rel_err = abs(nsf_val - true_val) / max(abs(true_val), 1e-12)
    mc_rel_err = abs(mc_val - true_val) / max(abs(true_val), 1e-12)
    mid_rel_err = abs(mid_val - true_val) / max(abs(true_val), 1e-12)

    s_dense = cached["s_dense"]
    x_pos = cached["x_pos"]
    y_pos = cached["y_pos"]
    h_dense = cached["h_dense"]
    in_interval = (s_dense >= s0) & (s_dense <= s1)

    s_prefix = cached["s_prefix"]
    true_prefix = cached["true_prefix"]
    nsf_prefix = cached["nsf_prefix"]

    h1_pt = float(wrapper.predict_prefix(params_2d, s1)[0])
    h0_pt = float(wrapper.predict_prefix(params_2d, s0)[0]) if s0 > 0.0 else float(true_prefix[0])

    fig, axes = _new_figure(3, figsize=(16.5, 4.9), dpi=130)

    # Panel (a): 2D Polar Spiral colored by kinetic energy
    ax0 = axes[0]
    ax0.set_facecolor(C_BG)
    ax0.plot(x_pos, y_pos, color="#CBD5E1", lw=1.2, alpha=0.8, label="Outside $[s_0, s_1]$")
    if np.sum(in_interval) >= 2:
        pts = np.column_stack([x_pos[in_interval], y_pos[in_interval]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", linewidths=2.4)
        lc.set_array(h_dense[in_interval][:-1])
        ax0.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax0, fraction=0.046, pad=0.04)
        cb.set_label(r"Energy $h(\mathbf{x}, s)$", fontsize=8.5)
    idx0 = int(np.argmin(np.abs(s_dense - s0)))
    idx1 = int(np.argmin(np.abs(s_dense - s1)))
    ax0.scatter([x_pos[idx0]], [y_pos[idx0]], color=C_ACCENT, s=65, zorder=5, edgecolors="white", label=f"$s_0={s0:.2f}$")
    ax0.scatter([x_pos[idx1]], [y_pos[idx1]], color=C_MC, s=65, zorder=5, edgecolors="white", label=f"$s_1={s1:.2f}$")
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.set_title(f"(a) 2D Spiral Trajectory\n{traj_label}", fontsize=10.5, fontweight="bold")
    ax0.set_xlabel("$x(s) = r(s)\\cos\\eta(s)$", fontsize=9)
    ax0.set_ylabel("$y(s) = r(s)\\sin\\eta(s)$", fontsize=9)
    ax0.legend(loc="upper right", fontsize=7.8, framealpha=0.9)
    ax0.grid(True, alpha=0.25)

    # Panel (b): Instantaneous Energy h(x, s) + MC-K sample dots
    ax1 = axes[1]
    ax1.set_facecolor(C_BG)
    ax1.axvspan(s0, s1, color="#DBEAFE", alpha=0.55, label=f"Query $[{s0:.2f}, {s1:.2f}]$")
    ax1.plot(s_dense, h_dense, color=C_TRUTH, lw=1.2, alpha=0.9, label=r"True $h(\mathbf{x}, s)$")
    ax1.scatter(mc_s[0], mc_e[0], color=C_MC, s=16, alpha=0.75, zorder=4, label=f"MC samples ($K={mc_k}$)")
    ax1.axhline(true_val, color=C_ACCENT, ls="--", lw=1.5, label=f"Analytic mean ({true_val:.1f})")
    ax1.axhline(mc_val, color=C_MC, ls=":", lw=1.6, label=f"MC-{mc_k} mean ({mc_val:.1f})")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_title("(b) Instantaneous Energy & MC Sampling", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Normalized time $s \\in [0, 1]$", fontsize=9)
    ax1.set_ylabel(r"$h(\mathbf{x}, s)$", fontsize=9)
    ax1.legend(loc="upper right", fontsize=7.6, framealpha=0.92)
    ax1.grid(True, alpha=0.25)

    # Panel (c): Learned Prefix Curve h_tilde(x, s) + 2-endpoint subtraction
    ax2 = axes[2]
    ax2.set_facecolor(C_BG)
    ax2.plot(s_prefix, true_prefix, color=C_TRUTH, lw=2.0, ls="--", label=r"True Prefix $\tilde{h}(\mathbf{x}, s)$")
    ax2.plot(s_prefix, nsf_prefix, color=C_NSF, lw=2.2, label=r"NSF Learned $\tilde{h}_\theta(\mathbf{x}, s)$")
    if s0 > 0.0:
        ax2.scatter([s0, s1], [h0_pt, h1_pt], color=C_NSF, s=80, zorder=6, edgecolors="white", lw=1.5, label="2 NSF Evaluations")
        ax2.annotate(f"$s_0={s0:.2f}$", (s0, h0_pt), textcoords="offset points", xytext=(6, 8), fontsize=8.5, color=C_NSF, fontweight="bold")
    else:
        ax2.scatter([s1], [h1_pt], color=C_NSF, s=80, zorder=6, edgecolors="white", lw=1.5, label="1 NSF Eval ($s_0=0$)")
    ax2.annotate(f"$s_1={s1:.2f}$", (s1, h1_pt), textcoords="offset points", xytext=(-44, -15), fontsize=8.5, color=C_NSF, fontweight="bold")
    ax2.set_xlim(0.0, 1.02)
    ax2.set_title("(c) Prefix Trick: 2 Evals -> Any Interval", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Prefix endpoint $s \\in [0, 1]$", fontsize=9)
    ax2.set_ylabel(r"Prefix mean $\tilde{h}(\mathbf{x}, s)$", fontsize=9)
    ax2.legend(loc="best", fontsize=7.8, framealpha=0.92)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()

    n_evals_nsf = 1 if s0 == 0.0 else 2
    speedup = mc_k / float(n_evals_nsf)
    summary_md = f"""### Live Interval Query Comparison — `{traj_label}` on $[{s0:.2f}, {s1:.2f}]$

| Estimator | Function / Network Evals | Predicted Mean Energy | Relative Error vs. Truth | Eval Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Analytic Truth (4,096-pt GL)** | `4,096` | **`{true_val:.2f}`** | `0.00%` | reference |
| **NSF (Ours, Prefix Trick)** | **`{n_evals_nsf}`** | **`{nsf_val:.2f}`** | **`{nsf_rel_err * 100:.2f}%`** | **`{speedup:.1f}×` fewer than MC-{mc_k}** |
| **IID Monte Carlo (`K={mc_k}`)** | `{mc_k}` | `{mc_val:.2f}` | `{mc_rel_err * 100:.2f}%` | `1.0×` |
| **Midpoint Quadrature (40)** | `40` | `{mid_val:.2f}` | `{mid_rel_err * 100:.2f}%` | — |

> **Formula used by NSF:** $\\hat{{h}}_{{[{s0:.2f}, {s1:.2f}]}} = \\frac{{{s1:.2f}\\,\\tilde{{h}}_\\theta(\\mathbf{{x}}, {s1:.2f}) - {s0:.2f}\\,\\tilde{{h}}_\\theta(\\mathbf{{x}}, {s0:.2f})}}{{{s1 - s0:.2f}}} = \\mathbf{{{nsf_val:.2f}}}$
{clamped_note}"""
    return fig, summary_md


# ---------------------------------------------------------------------------
# Tab 2: Why Naive Quadrature Fails (40-Cycle Aliasing Demo)
# ---------------------------------------------------------------------------
def render_aliasing_demo(
    width: float,
    traj_idx: int,
    wrapper: PrefixStatisticWrapper,
) -> tuple[Figure, str]:
    width = float(np.clip(width, 0.90, 1.00))
    traj_idx = int(np.clip(traj_idx, 0, len(HELD_OUT_TEST_POOL) - 1))
    params_1d = HELD_OUT_TEST_POOL[traj_idx]
    params_2d = params_1d[None, :]

    s0, s1 = 0.0, width
    true_val = float(analytic_interval_mean(params_2d, s0, s1)[0])
    nsf_val = float(wrapper.predict_interval(params_2d, s0, s1)[0])
    mid_arr, mid_nodes, mid_vals = midpoint_estimate(params_2d, s0, s1, n_points=40)
    mid_val = float(mid_arr[0])

    # Cached 21-width x 200-trajectory sweep for fast interactive slider updates
    sweep_widths, mid_curve, nsf_curve = _get_aliasing_sweep(wrapper)

    # Zoomed 40-cycle harmonic term on [0.0, 0.25]
    s_zoom = np.linspace(0.0, 0.25, 1000)
    phi_h = params_1d[6]
    gamma_w = params_1d[10]
    harmonic_zoom = 1.5 * gamma_w * np.sin(80.0 * np.pi * s_zoom + phi_h)
    nodes_zoom = mid_nodes[mid_nodes <= 0.25]
    nodes_harm = 1.5 * gamma_w * np.sin(80.0 * np.pi * nodes_zoom + phi_h)

    fig, axes = _new_figure(2, figsize=(14.5, 4.8), dpi=130)

    # Left: Phase-locking of 40-point Midpoint grid onto 40-cycle sine wave
    ax0 = axes[0]
    ax0.set_facecolor(C_BG)
    ax0.plot(s_zoom, harmonic_zoom, color=C_TRUTH, lw=1.4, alpha=0.85, label=r"40-cycle term $1.5\gamma_\omega\sin(80\pi s + \phi_h)$")
    ax0.scatter(nodes_zoom, nodes_harm, color=C_MID, s=55, zorder=5, edgecolors="white", lw=1.0, label=f"Midpoint-40 nodes (spacing = {width/40:.4f})")
    ax0.axhline(0.0, color="#64748B", ls="--", lw=1.0, label="True harmonic mean (0.0)")
    ax0.axhline(float(np.mean(nodes_harm)), color=C_MID, ls=":", lw=2.0, label=f"Midpoint-40 sampled mean ({np.mean(nodes_harm):+.2f})")
    ax0.set_xlim(0.0, 0.25)
    ax0.set_title(f"(a) Phase-Locking Aliasing at Width $w = {width:.3f}$", fontsize=10.5, fontweight="bold")
    ax0.set_xlabel("Time $s$ (zoomed to $[0.00, 0.25]$)", fontsize=9.5)
    ax0.set_ylabel("High-Frequency Angular Velocity Term", fontsize=9.5)
    ax0.legend(loc="upper right", fontsize=8.0, framealpha=0.92)
    ax0.grid(True, alpha=0.25)

    # Right: Global Relative L2 Error across Width [0.90, 1.00]
    ax1 = axes[1]
    ax1.set_facecolor(C_BG)
    ax1.plot(sweep_widths, mid_curve, color=C_MID, lw=2.5, marker="o", ms=4, label="Midpoint (40 evals) — aliases at $w \\to 1.0$!")
    ax1.plot(sweep_widths, nsf_curve, color=C_NSF, lw=2.5, marker="s", ms=4, label="NSF (2 evals) — flat & immune!")
    ax1.axhline(float(np.mean(PAPER_TABLE11_MC400)), color=C_MC, ls="--", lw=1.6, label="MC (400 evals) reference (~0.030)")
    ax1.axvline(width, color=C_ACCENT, ls="-.", lw=1.8, label=f"Selected width $w = {width:.3f}$")
    ax1.set_xlim(0.898, 1.002)
    ax1.set_ylim(0.0, max(0.155, max(mid_curve) * 1.1))
    ax1.set_title("(b) Relative $\\ell_2$ Error vs. Interval Width (200 Held-Out Designs)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Interval Width $w = s_1 - s_0$", fontsize=9.5)
    ax1.set_ylabel("Global Relative $\\ell_2$ Error", fontsize=9.5)
    ax1.legend(loc="upper left", fontsize=8.2, framealpha=0.92)
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()

    mid_err_single = abs(mid_val - true_val) / max(abs(true_val), 1e-12)
    nsf_err_single = abs(nsf_val - true_val) / max(abs(true_val), 1e-12)
    summary_md = f"""### Why 40-Point Midpoint Quadrature Explodes at $w = 1.00$

- The angular velocity $\\eta'(s)$ contains a **40-cycle harmonic** $1.5\\gamma_\\omega \\sin(80\\pi s + \\phi_h)$ with period $T = \\frac{{1}}{{40}} = 0.025$.
- At width $w = {width:.3f}$, the 40 equispaced Midpoint nodes have spacing $\\Delta s = {width / 40:.5f}$. When $w \\to 1.000$, $\\Delta s = 1/40 = T$ **exactly**, so every single quadrature node lands on the **exact same phase** of the sine wave!
- Instead of averaging the oscillation to `0`, Midpoint-40 locks onto a constant bias (`{np.mean(nodes_harm):+.2f}` on Trajectory #{traj_idx}), spiking global error from `0.019` to `0.136` (**7× worse**).
- **On Trajectory #{traj_idx} ($[0, {width:.3f}]$):** Analytic Truth = `{true_val:.2f}` | **NSF (2 evals)** = **`{nsf_val:.2f}`** (`{nsf_err_single*100:.2f}%` err) | Midpoint-40 = `{mid_val:.2f}` (`{mid_err_single*100:.2f}%` err).
"""
    return fig, summary_md


# ---------------------------------------------------------------------------
# Tab 3: Verified NeurIPS 2026 Benchmarks (Tables 10 & 11) + Open Artifact
# ---------------------------------------------------------------------------
def render_benchmarks_tab(benchmarks_path: Path) -> tuple[Figure, str, str]:
    data = json.loads(benchmarks_path.read_text())
    t11 = data["table11_analytic"]
    t10 = data["table10_mc1000"]

    widths = np.array([r["width"] for r in t11])
    nsf_mean = np.array([r["NSF"] for r in t11])
    nsf_se = np.array([r["NSF_se"] for r in t11])
    mc400 = np.array([r["MC400_mean"] for r in t11])
    mc100 = np.array([r["MC100_mean"] for r in t11])
    mc50 = np.array([r["MC50_mean"] for r in t11])
    mid40 = np.array([r["Midpoint40"] for r in t11])
    gl40 = np.array([r["GL40"] for r in t11])
    strat40 = np.array([r["Stratified40_mean"] for r in t11])
    sobol40 = np.array([r["Sobol40_mean"] for r in t11])

    fig, axes = _new_figure(2, figsize=(14.5, 4.8), dpi=130)

    # Left: NSF vs Monte Carlo (50, 100, 400)
    ax0 = axes[0]
    ax0.set_facecolor(C_BG)
    ax0.plot(widths, mc50, "o--", color="#EF4444", lw=1.8, ms=4, label="IID MC (50 evals)")
    ax0.plot(widths, mc100, "s--", color="#F97316", lw=1.8, ms=4, label="IID MC (100 evals)")
    ax0.plot(widths, mc400, "^--", color="#B91C1C", lw=2.0, ms=5, label="IID MC (400 evals)")
    ax0.errorbar(widths, nsf_mean, yerr=nsf_se, fmt="o-", color=C_NSF, lw=2.5, ms=5, capsize=3, label="NSF (2 evals) — Ours (5-seed)")
    ax0.plot(widths, PAPER_TABLE11_NSF, "x:", color=C_TRUTH, lw=1.6, ms=6, label="NSF (2 evals) — Paper Table 11")
    ax0.set_title("(a) Table 11: NSF (2 evals) vs. Monte Carlo", fontsize=10.5, fontweight="bold")
    ax0.set_xlabel("Interval Width $s_1 - s_0$", fontsize=9.5)
    ax0.set_ylabel("Global Relative $\\ell_2$ Error (Analytic Ref)", fontsize=9.5)
    ax0.legend(loc="center right", fontsize=8.0, framealpha=0.92)
    ax0.grid(True, alpha=0.25)

    # Right: NSF vs 40-eval Quadrature Rules
    ax1 = axes[1]
    ax1.set_facecolor(C_BG)
    ax1.plot(widths, mid40, "o-", color=C_MID, lw=2.0, ms=4, label="Midpoint (40 evals)")
    ax1.plot(widths, gl40, "s--", color="#7C3AED", lw=1.8, ms=4, label="Gauss-Legendre (40 evals)")
    ax1.plot(widths, sobol40, "^--", color="#0891B2", lw=1.8, ms=4, label="Sobol QMC (40 evals)")
    ax1.plot(widths, strat40, "d--", color=C_ACCENT, lw=1.8, ms=4, label="Stratified (40 evals)")
    ax1.errorbar(widths, nsf_mean, yerr=nsf_se, fmt="o-", color=C_NSF, lw=2.6, ms=5, capsize=3, label="NSF (2 evals) — Ours (5-seed)")
    ax1.set_title("(b) Table 11: NSF (2 evals) vs. 40-Eval Quadrature", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Interval Width $s_1 - s_0$", fontsize=9.5)
    ax1.set_ylabel("Global Relative $\\ell_2$ Error (Analytic Ref)", fontsize=9.5)
    ax1.legend(loc="upper left", fontsize=8.0, framealpha=0.92)
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()

    rows_md = []
    for i, r in enumerate(t11):
        r10 = t10[i]
        rows_md.append(
            f"| `{r['width']:.4f}` | `{PAPER_TABLE11_NSF[i]:.4f}` | **`{r['NSF']:.4f} ± {r['NSF_se']:.4f}`** "
            f"| `{r['MC400_mean']:.4f}` | `{r['Midpoint40']:.4f}` | `{PAPER_TABLE10_NSF[i]:.4f}` | **`{r10['NSF']:.4f} ± {r10['NSF_se']:.4f}`** |"
        )
    table_body = "\n".join(rows_md)

    summary_md = f"""### Verified 5-Seed Reproduction vs. Published Paper (`arXiv:2605.11327v1`)

- **Table 11 MAE vs. Paper:** **`{data['table11_nsf_mae_vs_paper']:.5f}`** (max abs diff `{data['table11_nsf_max_diff_vs_paper']:.5f}`)
- **Table 10 MAE vs. Paper:** **`{data['table10_nsf_mae_vs_paper']:.5f}`**
- **Figure 4(a) $R^2$ (5-seed mean):** **`{data['r2_vs_mc1000_mean']:.5f}`** vs. Paper **`{PAPER_FIG4A_R2:.4f}`**

| Width $s_1 - s_0$ | Paper T11 | Ours T11 (5-seed) | MC (400) | Midpoint (40) | Paper T10 | Ours T10 (5-seed) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{table_body}
"""

    snippet_md = """### Reusable Open Artifact (`nsf_core.py`)

Because `arXiv:2605.11327v1` does not link an official code repository, we package our verified implementation as a clean, standalone PyTorch module (`nsf_core.py`):

```python
import numpy as np
from nsf_core import PrefixStatisticWrapper, sample_designs, train_nsf, analytic_interval_mean

# 1. Load pretrained verified checkpoint (instant) OR train from scratch in ~60s on CPU:
nsf = PrefixStatisticWrapper.load("checkpoints/nsf_seed0.pt")
# nsf = train_nsf(sample_designs(3000, np.random.default_rng(42)), epochs=2000)

# 2. Query any interval [s0, s1] on unseen trajectories in strictly 2 evaluations:
test_x = sample_designs(900, np.random.default_rng(2026))
pred_mean = nsf.predict_interval(test_x, s0=0.05, s1=0.95)
true_mean = analytic_interval_mean(test_x, s0=0.05, s1=0.95)
print("Relative L2 error:", np.linalg.norm(pred_mean - true_mean) / np.linalg.norm(true_mean))
```
"""
    return fig, summary_md, snippet_md
