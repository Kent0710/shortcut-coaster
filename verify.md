# Verification: Interactive Gradio Explainer & Open Artifact (`best-explainer`)

- **Command:** `python test_explainer.py` (4/4 PASS)
- **Pretrained Checkpoint:** `code/checkpoints/nsf_seed0.pt` (`3.04 MB`, trained in `147.6s` on CPU via `train_checkpoint.py`)

## Automated Verification Results (4/4 PASS)
1. **`nsf_core.py` Physics, Estimators & Edge Cases:** Verified polar trajectory reconstruction ($x^2 + y^2 = r^2$ to `1e-10`), exact instantaneous energy parity, MC / Midpoint / Gauss-Legendre estimator shapes & finiteness, and `ValueError` validation on invalid intervals.
2. **Pretrained Checkpoint Parity (`checkpoints/nsf_seed0.pt`):**
   - Size: **`3.04 MB`** (`< 5.0 MB` target for instant HF Space CPU startup)
   - Prefix additivity discrepancy: **`5.30e-17`** (exact machine precision)
   - Table 11 max relative $\ell_2$ across 8 widths: **`0.0227`** (all `<= 0.023`)
   - Table 11 MAE vs. Paper: **`0.00041`** (exact match to verified `exp/exact-repro` seed-0)
   - $R^2$ vs. Analytic Truth: **`0.998772`**
3. **Interactive Explainer Callbacks & Visual Inspection:**
   - Exercised `render_prefix_trick` (default, random trajectory, boundary `s0=0.0, s1=1.0`, inverted `s0 > s1` safe clamping with UX note), `render_aliasing_demo` (`w=1.00` and `w=0.90`, cached sweep `259.4 ms` cold → `54.5 ms` cached), and `render_benchmarks_tab`.
   - Verified zero global `pyplot` figure leaks (`len(plt.get_fignums()) == 0`) via OO `Figure` + `FigureCanvasAgg` API and `ValueError` guard on invalid `(N, 10)` design shapes in `nsf_core.py`.
   - Visually inspected `outputs/preview_tab1.png`, `outputs/preview_tab2.png`, and `outputs/preview_tab3.png`: clean layout, legible legends, zero overlapping labels.
4. **`app.py` Syntax:** AST-verified cleanly. *(Note: live `gradio` server startup was not executed in the offline sandbox because `gradio` is not installed in the offline venv; all underlying callbacks and Matplotlib figures are verified.)*
