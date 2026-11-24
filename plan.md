# Plan: Interactive Gradio Explainer & Open Artifact (`best-explainer`)

- **Tracks:** 📊 Best Explainer + ✨ Open Artifact + 🗳️ Community Choice — NeurIPS 2026 ML Intern Reproductions
- **Paper:** *Neural Statistical Functions* (Xu et al., NeurIPS 2026, `arXiv:2605.11327v1`)
- **Format:** Hugging Face Gradio 5 Space (`app.py`) backed by a clean, reusable PyTorch library (`nsf_core.py`) and a small pretrained checkpoint (`checkpoints/nsf_seed0.pt`, `3.04 MB`) for instant CPU startup.

## Objective
1. **Best Explainer:** Make the Prefix Statistics trick visually intuitive in 30 seconds for both ML researchers and general audiences via 3 interactive tabs:
   - **Tab 1 — The Prefix Trick (Live 2D Spiral):** Interactive 2D spiral colored by kinetic energy, instantaneous energy with $K$-sample Monte Carlo dots, and the learned prefix curve showing 2-endpoint subtraction.
   - **Tab 2 — Why Naive Quadrature Fails (40-Cycle Aliasing):** Live width slider (`0.90 -> 1.00`) showing 40-point Midpoint quadrature phase-locking onto the 40-cycle sine wave while 2-eval NSF stays flat.
   - **Tab 3 — Verified NeurIPS 2026 Benchmarks & Open Artifact:** Verified 5-seed reproduction of Tables 10 & 11 (`0.00030` MAE vs. paper) + copy-ready Python snippet for `nsf_core.py`.
2. **Open Artifact:** Because `arXiv:2605.11327v1` links no official GitHub repository, package `nsf_core.py` with clean APIs (`sample_designs`, `instantaneous_energy_np`, `trajectory_polar_np`, `analytic_interval_mean`, `NeuralStatisticalMLP`, `PrefixStatisticWrapper`, `train_nsf`, `mc_estimate`, `midpoint_estimate`, `gauss_legendre_estimate`) and verified weights.
