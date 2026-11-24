---
title: Neural Statistical Functions — Interactive Explainer
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
license: mit
short_description: Interactive NeurIPS 2026 explainer & PyTorch Open Artifact for Neural Statistical Functions
---

# Neural Statistical Functions — Interactive Explainer & Open Artifact

An interactive **Gradio Space** and standalone PyTorch library (`nsf_core.py`) built on top of our verified reproduction of **Neural Statistical Functions** (*Xu et al., NeurIPS 2026, `arXiv:2605.11327v1`*).

## What's Inside
1. **Tab 1 — The Prefix Trick (Live 2D Spiral):** Drag `[s0, s1]` and Monte Carlo sample count `K` on held-out 2D trajectories to see how 2 prefix network evaluations beat 400-sample Monte Carlo.
2. **Tab 2 — Why Naive Quadrature Fails (40-Cycle Aliasing):** Drag interval width `w` toward `1.00` to watch 40-point Midpoint quadrature phase-lock onto the 40-cycle harmonic and spike error 7× while 2-eval NSF stays flat.
3. **Tab 3 — Verified NeurIPS 2026 Benchmarks & Open Artifact:** Complete 5-seed reproduction of Tables 10 & 11 (`0.00030` MAE vs. paper, $R^2 = 0.99783 \pm 0.00006$ vs. `0.9976`) plus copy-ready usage for `nsf_core.py`.

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```

## Use `nsf_core.py` as a Reusable PyTorch Artifact
```python
import numpy as np
from nsf_core import PrefixStatisticWrapper, sample_designs, analytic_interval_mean

nsf = PrefixStatisticWrapper.load("checkpoints/nsf_seed0.pt")
test_x = sample_designs(900, np.random.default_rng(2026))
pred = nsf.predict_interval(test_x, s0=0.05, s1=0.95)
truth = analytic_interval_mean(test_x, s0=0.05, s1=0.95)
```
