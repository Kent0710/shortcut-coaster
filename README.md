---
title: The Shortcut Coaster — Neural Statistical Functions
emoji: 🎢
colorFrom: cyan
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# The Shortcut Coaster — Neural Statistical Functions Explainer & Open Artifact

- **Competition:** NeurIPS 2026 ML Intern Reproductions (Gradio & Hugging Face)
- **Paper:** *Neural Statistical Functions* (Xu et al., NeurIPS 2026, `arXiv:2605.11327v1`)
- **Tracks:** Best Explainer · Open Artifact · Community Choice
- **GitHub:** [github.com/Kent0710/shortcut-coaster](https://github.com/Kent0710/shortcut-coaster)

## What Is Inside

1. **`vortex-rollercoaster-layman/` — The Shortcut Coaster (Plain-English Story + 3D Ride):**
   - 13-step zero-jargon interactive story explaining how Neural Statistical Functions measure any segment of a wild curve in **2 checks** instead of hundreds of random samples.
   - Seamless handoff into a 60 FPS Three.js 3D rollercoaster ride with a live **Paper in Action** tracker comparing **AI (2 checks)** vs. **20 Random Guesses** on real held-out model predictions.
2. **`03-vortex-rollercoaster/` — Technical 3D Edition:**
   - Full research-terminology 3D tunnel ride with live $\tilde{h}_\theta(s)$ prefix meter, $[s_0, s_1]$ interval gates, `0.008`-width impulse spike detection, and `MC-20` variance radar.
3. **`code/` — Open Artifact & Verified Reproduction (`nsf_core.py` + `checkpoints/nsf_seed0.pt`):**
   - Clean, standalone PyTorch implementation reproducing Paper Table 11 (**5-seed MAE `0.00030`**, $R^2 = 0.9978$ vs. paper `0.9976`).
   - Full-domain 2-eval NSF mean relative error across the 20 exported held-out trajectories: **`1.99%`**.

## Run Locally

```bash
# 1. Play the standalone 3D web app directly (no server needed)
open vortex-rollercoaster-layman/index.html

# 2. Or launch the unified Gradio 5 Space
uv run python app.py

# 3. Run automated verification
python test_3d_games.py
```
