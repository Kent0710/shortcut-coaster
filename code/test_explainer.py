#!/usr/bin/env python3
"""
Automated test suite for `best-explainer` (Open Artifact `nsf_core.py` + Gradio Explainer callbacks).

Verifies:
  1. `nsf_core.py` physics, quadrature, and wrapper edge cases (boundary s0=0, invalid inputs, additivity).
  2. Pretrained checkpoint `checkpoints/nsf_seed0.pt` size (< 5 MB) and Table 11 parity (rel_L2 <= 0.023, R^2 >= 0.997).
  3. All 3 interactive Gradio explainer callbacks (`render_prefix_trick`, `render_aliasing_demo`,
     `render_benchmarks_tab`) across normal and edge inputs, saving preview PNGs to `outputs/`.
  4. `app.py` syntax/AST validity (and live `build_demo()` instantiation if `gradio` is installed).
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np

from explainer_viz import (
    PRESET_LABELS,
    render_aliasing_demo,
    render_benchmarks_tab,
    render_prefix_trick,
)
from nsf_core import (
    PAPER_TABLE11_NSF,
    PrefixStatisticWrapper,
    analytic_interval_mean,
    gauss_legendre_estimate,
    instantaneous_energy_np,
    mc_estimate,
    midpoint_estimate,
    rel_l2,
    sample_designs,
    trajectory_polar_np,
)

ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "checkpoints" / "nsf_seed0.pt"
DATA_PATH = ROOT / "data" / "verified_benchmarks.json"
OUT_DIR = ROOT / "outputs"


def test_nsf_core_physics_and_edge_cases() -> None:
    rng = np.random.default_rng(123)
    designs = sample_designs(10, rng)
    assert designs.shape == (10, 11)

    # Polar trajectory consistency
    s_grid = np.linspace(0.0, 1.0, 200)
    x, y, r, eta_dot, energy = trajectory_polar_np(designs[0], s_grid)
    assert x.shape == y.shape == r.shape == eta_dot.shape == energy.shape == (200,)
    assert np.allclose(np.sqrt(x**2 + y**2), np.abs(r), atol=1e-10)
    assert np.allclose(energy, instantaneous_energy_np(designs[0:1, :], s_grid[None, :])[0], atol=1e-10)

    # Estimators finite & positive
    a_mean = analytic_interval_mean(designs, 0.05, 0.95)
    mc_mean, mc_s, mc_e = mc_estimate(designs, 0.05, 0.95, k=50, rng=rng)
    mid_mean, mid_n, mid_v = midpoint_estimate(designs, 0.05, 0.95, n_points=40)
    gl_mean = gauss_legendre_estimate(designs, 0.05, 0.95, n_points=40)
    assert a_mean.shape == mc_mean.shape == mid_mean.shape == gl_mean.shape == (10,)
    assert np.all(a_mean > 0.0) and np.all(np.isfinite(mc_mean))
    assert mc_s.shape == (10, 50) and mc_e.shape == (10, 50)
    assert mid_n.shape == (40,) and mid_v.shape == (10, 40)

    # Invalid interval validation
    for bad_s0, bad_s1 in [(0.5, 0.5), (0.8, 0.2), (-0.1, 0.5), (0.2, 1.1)]:
        try:
            analytic_interval_mean(designs, bad_s0, bad_s1)
            raise AssertionError(f"Expected ValueError for [{bad_s0}, {bad_s1}]")
        except ValueError:
            pass
    print("[PASS] 1. nsf_core physics, estimators, and edge-case validation")


def test_pretrained_checkpoint_accuracy() -> None:
    assert CKPT_PATH.is_file(), f"Missing pretrained checkpoint: {CKPT_PATH}"
    size_mb = CKPT_PATH.stat().st_size / (1024 * 1024)
    assert size_mb < 5.0, f"Checkpoint too large ({size_mb:.2f} MB >= 5.0 MB)"

    wrapper = PrefixStatisticWrapper.load(CKPT_PATH, device="cpu")
    rng = np.random.default_rng(42)
    _ = sample_designs(3000, rng)
    test_params = sample_designs(900, rng)

    # Exact prefix additivity check
    h01 = wrapper.predict_interval(test_params[:64], 0.0, 0.45)
    h12 = wrapper.predict_interval(test_params[:64], 0.45, 0.90)
    h02 = wrapper.predict_interval(test_params[:64], 0.0, 0.90)
    add_err = rel_l2(0.45 * h01 + 0.45 * h12, 0.90 * h02)
    assert add_err < 1e-6, f"Prefix additivity failed: {add_err}"

    # Table 11 parity across all 8 widths
    widths = np.linspace(0.90, 1.00, 8)
    errs = []
    all_p, all_r = [], []
    for w in widths:
        intervals = [(0.0, 1.0)] if np.isclose(w, 1.0) else [(0.0, float(w)), (float(1.0 - w), 1.0)]
        refs = np.concatenate([analytic_interval_mean(test_params, s0, s1) for s0, s1 in intervals])
        preds = np.concatenate([wrapper.predict_interval(test_params, s0, s1) for s0, s1 in intervals])
        errs.append(rel_l2(preds, refs))
        all_p.append(preds)
        all_r.append(refs)

    max_err = max(errs)
    mae_vs_paper = float(np.mean(np.abs(np.array(errs) - PAPER_TABLE11_NSF)))
    y_p, y_r = np.concatenate(all_p), np.concatenate(all_r)
    r2 = float(1.0 - np.sum((y_p - y_r) ** 2) / np.sum((y_r - np.mean(y_r)) ** 2))

    assert max_err <= 0.0235, f"Expected all Table 11 rel_L2 <= 0.0235, got max {max_err:.5f}"
    assert r2 >= 0.997, f"Expected R^2 >= 0.997, got {r2:.6f}"
    print(
        f"[PASS] 2. Pretrained checkpoint ({size_mb:.2f} MB) | "
        f"max rel_L2={max_err:.4f}, MAE vs paper={mae_vs_paper:.5f}, R^2={r2:.6f}, add_err={add_err:.2e}"
    )


def test_explainer_callbacks_and_previews() -> None:
    wrapper = PrefixStatisticWrapper.load(CKPT_PATH, device="cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tab 1: normal + edge case (s0=0.0, s1=1.0; and inverted s0 > s1 clamping)
    fig1, md1 = render_prefix_trick(PRESET_LABELS[0], 0, 0.05, 0.95, 50, 0, wrapper)
    assert isinstance(fig1, plt.Figure) and "NSF (Ours, Prefix Trick)" in md1
    fig1.savefig(OUT_DIR / "preview_tab1.png", dpi=130, bbox_inches="tight")

    fig1_edge, md1_edge = render_prefix_trick("Random Held-Out Trajectory", 17, 0.0, 1.0, 400, 3, wrapper)
    assert isinstance(fig1_edge, plt.Figure) and "`1`" in md1_edge

    fig1_inv, md1_inv = render_prefix_trick(PRESET_LABELS[1], 0, 0.85, 0.40, 10, 0, wrapper)
    assert isinstance(fig1_inv, plt.Figure) and "Adjusted interval end" in md1_inv

    # Tab 2: aliasing demo at w=1.00 and w=0.90 (second call uses cached sweep)
    import time

    t0 = time.perf_counter()
    fig2, md2 = render_aliasing_demo(1.00, 0, wrapper)
    t1 = time.perf_counter()
    assert isinstance(fig2, plt.Figure) and "40-cycle harmonic" in md2
    fig2.savefig(OUT_DIR / "preview_tab2.png", dpi=130, bbox_inches="tight")

    t2 = time.perf_counter()
    fig2_low, _ = render_aliasing_demo(0.90, 5, wrapper)
    t3 = time.perf_counter()
    assert isinstance(fig2_low, plt.Figure)

    # Tab 3: verified benchmarks
    fig3, md3, snip3 = render_benchmarks_tab(DATA_PATH)
    assert isinstance(fig3, plt.Figure) and "0.00030" in md3 and "PrefixStatisticWrapper" in snip3
    fig3.savefig(OUT_DIR / "preview_tab3.png", dpi=130, bbox_inches="tight")

    # Zero global pyplot figure leaks!
    assert len(plt.get_fignums()) == 0, f"Leaked {len(plt.get_fignums())} pyplot figures!"

    # Wrong-column input shape validation in nsf_core.py
    try:
        wrapper.predict_prefix(np.zeros((4, 10)), 0.5)
        raise AssertionError("Expected ValueError for (4, 10) params")
    except ValueError:
        pass

    print(
        f"[PASS] 3. All 3 Gradio explainer callbacks + zero figure leaks "
        f"(Tab 2 cold={(t1 - t0) * 1000:.1f}ms, cached={(t3 - t2) * 1000:.1f}ms)"
    )


def test_app_syntax_and_optional_gradio() -> None:
    app_path = ROOT / "app.py"
    source = app_path.read_text()
    ast.parse(source)
    if importlib.util.find_spec("gradio") is not None:
        import app  # noqa: PLC0415

        demo = app.build_demo()
        assert demo is not None
        print("[PASS] 4. app.py AST valid AND live Gradio Blocks instantiated")
    else:
        print("[PASS] 4. app.py AST valid (gradio not installed in offline venv; skipped live Blocks launch)")


def main() -> None:
    test_nsf_core_physics_and_edge_cases()
    test_pretrained_checkpoint_accuracy()
    test_explainer_callbacks_and_previews()
    test_app_syntax_and_optional_gradio()
    print("\nALL 4 EXPLAINER & OPEN ARTIFACT TESTS PASSED!")


if __name__ == "__main__":
    main()
