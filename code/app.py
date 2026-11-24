#!/usr/bin/env python3
"""
Interactive Hugging Face Gradio Explainer Space for:
  Neural Statistical Functions (Xu et al., NeurIPS 2026, arXiv:2605.11327v1)

3 interactive tabs:
  1. The Prefix Trick (Live 2D Spiral): compare 2-eval NSF vs. K-sample Monte Carlo
  2. Why Naive Quadrature Fails (40-Cycle Aliasing): watch Midpoint-40 phase-lock at w -> 1.0
  3. Verified NeurIPS 2026 Benchmarks (Tables 10 & 11) + Reusable Open Artifact (`nsf_core.py`)
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from explainer_viz import (
    PRESET_LABELS,
    render_aliasing_demo,
    render_benchmarks_tab,
    render_prefix_trick,
)
from nsf_core import PrefixStatisticWrapper

ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "checkpoints" / "nsf_seed0.pt"
BENCHMARKS_PATH = ROOT / "data" / "verified_benchmarks.json"

_WRAPPER: PrefixStatisticWrapper | None = None


def get_wrapper() -> PrefixStatisticWrapper:
    global _WRAPPER
    if _WRAPPER is None:
        if not CKPT_PATH.is_file():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found at {CKPT_PATH}. Run `python train_checkpoint.py` first."
            )
        _WRAPPER = PrefixStatisticWrapper.load(CKPT_PATH, device="cpu")
    return _WRAPPER


def build_demo() -> gr.Blocks:
    wrapper = get_wrapper()
    bench_fig, bench_md, snippet_md = render_benchmarks_tab(BENCHMARKS_PATH)

    with gr.Blocks(
        title="Neural Statistical Functions — Interactive Explainer (NeurIPS 2026)",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    ) as demo:
        gr.Markdown(
            """
            # Neural Statistical Functions — Interactive Explainer
            **Paper:** *Neural Statistical Functions* (Xu et al., NeurIPS 2026, `arXiv:2605.11327v1`) ·
            **Verified Reproduction:** `0.00030` MAE vs. Table 11 (`5-seed`), $R^2 = 0.9978$ vs. `0.9976`

            Instead of evaluating hundreds of Monte Carlo samples for every new query interval $[s_0, s_1]$,
            a **Neural Statistical Function** learns a single prefix mean network $\\tilde{h}_\\theta(\\mathbf{x}, s)$
            and computes **any** interval mean in strictly **2 network evaluations** via
            $\\hat{h}_{[s_0, s_1]} = \\frac{s_1 \\tilde{h}_\\theta(\\mathbf{x}, s_1) - s_0 \\tilde{h}_\\theta(\\mathbf{x}, s_0)}{s_1 - s_0}$.
            """
        )

        with gr.Tabs():
            # Tab 1: The Prefix Trick
            with gr.Tab("1. The Prefix Trick (Live 2D Spiral)"):
                with gr.Row():
                    preset_dd = gr.Dropdown(
                        choices=PRESET_LABELS,
                        value=PRESET_LABELS[0],
                        label="Held-Out 2D Trajectory",
                        scale=2,
                    )
                    rand_seed = gr.Number(value=0, precision=0, label="Random Trajectory Seed", scale=1)
                    mc_k_slider = gr.Slider(10, 400, value=50, step=10, label="Monte Carlo Samples (K)", scale=2)
                    mc_seed_slider = gr.Slider(0, 20, value=0, step=1, label="MC Seed", scale=1)
                with gr.Row():
                    s0_slider = gr.Slider(0.0, 0.90, value=0.05, step=0.01, label="Interval Start (s0)")
                    s1_slider = gr.Slider(0.10, 1.00, value=0.95, step=0.01, label="Interval End (s1)")

                t1_plot = gr.Plot(label="Synchronized 2D Spiral, Instantaneous Energy, and Prefix Curve")
                t1_md = gr.Markdown()

                def _cb_tab1(preset: str, rseed: float, s0: float, s1: float, k: int, mseed: int):
                    return render_prefix_trick(preset, int(rseed or 0), s0, s1, int(k), int(mseed), wrapper)

                t1_inputs = [preset_dd, rand_seed, s0_slider, s1_slider, mc_k_slider, mc_seed_slider]
                for comp in t1_inputs:
                    comp.change(_cb_tab1, inputs=t1_inputs, outputs=[t1_plot, t1_md], show_progress="hidden")
                demo.load(_cb_tab1, inputs=t1_inputs, outputs=[t1_plot, t1_md], show_progress="hidden")

            # Tab 2: Why Naive Quadrature Fails
            with gr.Tab("2. Why Naive Quadrature Fails (40-Cycle Aliasing)"):
                with gr.Row():
                    w_slider = gr.Slider(
                        0.90,
                        1.00,
                        value=1.00,
                        step=0.005,
                        label="Interval Width (w = s1 - s0) — drag toward 1.00 to trigger aliasing!",
                        scale=3,
                    )
                    traj_slider = gr.Slider(0, 19, value=0, step=1, label="Held-Out Trajectory Index", scale=1)

                t2_plot = gr.Plot(label="40-Cycle Phase-Locking Aliasing vs. 2-Eval NSF")
                t2_md = gr.Markdown()

                def _cb_tab2(w: float, idx: int):
                    return render_aliasing_demo(w, int(idx), wrapper)

                t2_inputs = [w_slider, traj_slider]
                for comp in t2_inputs:
                    comp.change(_cb_tab2, inputs=t2_inputs, outputs=[t2_plot, t2_md], show_progress="hidden")
                demo.load(_cb_tab2, inputs=t2_inputs, outputs=[t2_plot, t2_md], show_progress="hidden")

            # Tab 3: Verified Benchmarks & Open Artifact
            with gr.Tab("3. Verified Benchmarks & Open Artifact"):
                gr.Plot(value=bench_fig, label="Verified 5-Seed Reproduction of Paper Table 11")
                gr.Markdown(bench_md)
                gr.Markdown(snippet_md)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
