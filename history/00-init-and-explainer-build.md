# History 00: Interactive Explainer & Open Artifact Build

- **Date:** 2026-11-23
- **Scope:** Built the Hugging Face Gradio Space (`app.py`, `explainer_viz.py`, `README.md`) and standalone PyTorch Open Artifact (`nsf_core.py`) inside `neural-statistical-functions/best-explainer/code/`.

## Actions Taken
1. Extracted a clean, zero-bloat PyTorch library (`nsf_core.py`) with `PrefixStatisticWrapper`, `NeuralStatisticalMLP`, `train_nsf`, polar trajectory integration (`trajectory_polar_np`), and quadrature/MC baselines.
2. Trained and saved `checkpoints/nsf_seed0.pt` (`3.04 MB`, `0.00041` MAE vs. Paper Table 11, `R^2 = 0.998772`) via `train_checkpoint.py`.
3. Copied verified 5-seed Tables 10 & 11 benchmarks into `data/verified_benchmarks.json`.
4. Built `explainer_viz.py` (3-tab Matplotlib figure + Markdown summary engine) and wired it into `app.py` (Gradio 5 `gr.Blocks`).
5. Ran `test_explainer.py` (4/4 PASS) and visually verified all 3 tab preview PNGs (`outputs/preview_tab1.png`, `preview_tab2.png`, `preview_tab3.png`).
