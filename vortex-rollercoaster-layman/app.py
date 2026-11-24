"""Gradio Space wrapper for vortex-rollercoaster-layman (Plain-English 3D Ride)."""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

DIR = Path(__file__).resolve().parent


def build_embedded_html() -> str:
    index_html = (DIR / "index.html").read_text(encoding="utf-8")
    game_data_js = (DIR / "game_data.js").read_text(encoding="utf-8")
    inlined = index_html.replace(
        '<script src="game_data.js"></script>',
        f"<script>{game_data_js}</script>",
    )
    escaped = html.escape(inlined, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        'style="width:100%;height:88vh;min-height:680px;border:none;border-radius:14px;background:#06040f;" '
        'allow="accelerometer; fullscreen"></iframe>'
    )


def create_app() -> gr.Blocks:
    with gr.Blocks(title="The Shortcut Coaster — Plain-English 3D Explainer") as demo:
        gr.HTML(build_embedded_html())
    return demo


if __name__ == "__main__":
    create_app().launch(server_name="0.0.0.0", server_port=7864)
