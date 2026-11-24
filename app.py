"""Unified Hugging Face Space entry point for The Shortcut Coaster (Best Explainer + Open Artifact)."""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent


def _embed_game(subdir: str, bg: str) -> str:
    gdir = ROOT / subdir
    index_html = (gdir / "index.html").read_text(encoding="utf-8")
    game_data_js = (gdir / "game_data.js").read_text(encoding="utf-8")
    inlined = index_html.replace(
        '<script src="game_data.js"></script>',
        f"<script>{game_data_js}</script>",
    )
    escaped = html.escape(inlined, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:86vh;min-height:680px;border:none;border-radius:14px;background:{bg};" '
        'allow="accelerometer; fullscreen"></iframe>'
    )


ARTIFACT_MD = """
## Verified Reproduction & Open Artifact — *Neural Statistical Functions* (`arXiv:2605.11327v1`)

- **Verdict:** `verified`
- **Reproduced Table 11 (5-Seed 2D Dynamical Systems):** MAE = `0.00030`, $R^2 = 0.9978$ (Paper reported $R^2 = 0.9976$)
- **Exported 20 Held-Out Trajectories (`checkpoints/nsf_seed0.pt`):**
  - Full-domain `[0.0, 1.0]` 2-eval NSF mean relative error: **`1.99%`**
  - Sub-interval `[0.20, 0.80]` 2-eval NSF mean relative error: **`4.28%`** (`95.72%` accuracy) vs. `MC-20` **`11.30%`** (using 10x more evaluations)
- **GitHub Repository:** [github.com/Kent0710/shortcut-coaster](https://github.com/Kent0710/shortcut-coaster)

### Quick Python Usage (`code/nsf_core.py`)
```python
from code.nsf_core import PrefixStatisticWrapper, sample_designs, analytic_interval_mean
import numpy as np

wrapper = PrefixStatisticWrapper.load("code/checkpoints/nsf_seed0.pt", device="cpu")
p = sample_designs(1, np.random.default_rng(42))

# 2-evaluation interval mean on [0.20, 0.80]
nsf_est = wrapper.predict_interval_mean(p, 0.20, 0.80)[0]
true_val = analytic_interval_mean(p, 0.20, 0.80)[0]
print(f"NSF (2 evals): {nsf_est:.2f} | Analytic Truth: {true_val:.2f}")
```
"""


def create_app() -> gr.Blocks:
    with gr.Blocks(title="The Shortcut Coaster — Neural Statistical Functions") as demo:
        with gr.Tabs():
            with gr.Tab("The Shortcut Coaster (Interactive Story & 3D Ride)"):
                gr.HTML(_embed_game("vortex-rollercoaster-layman", "#050509"))
            with gr.Tab("Technical 3D Mode"):
                gr.HTML(_embed_game("03-vortex-rollercoaster", "#06040f"))
            with gr.Tab("Verified Reproduction & Open Artifact"):
                gr.Markdown(ARTIFACT_MD)
    return demo


if __name__ == "__main__":
    create_app().launch(server_name="0.0.0.0", server_port=7860)
