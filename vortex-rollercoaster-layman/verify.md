# Verification — Vortex Rollercoaster (Plain-English Story Edition)

## Automated Checks (`../test_3d_games.py`)
- **Status:** PASS (2026-11-24)
- **Data Integrity:** `game_data.js` (436.8 KB) parsed cleanly — 20 real tracks × 601 points from verified checkpoint `checkpoints/nsf_seed0.pt`.
- **Accuracy:** Full-track mean relative error = `0.0199`; `[20%, 85%]` 2-check average error = `0.0378`.
- **Jargon-Free Check:** PASS — zero technical terms (`Monte Carlo`, `MC-20`, `NSF`, `h̃`, `s₀`, `s₁`, `0.008`, `Prefix`, `Held-Out`, `kinetic energy`) in visible UI.
- **Emoji/Icon-Free Check:** PASS — zero Unicode emojis/dingbats anywhere in `index.html`.
- **5-Step Story Flow Check:** PASS — `story-layer`, `btn-next`, `btn-back`, `btn-skip`, `btn-replay`, and `sim-ui` verified.
- **JS Syntax & Gradio Wrapper:** `node --check` PASS; `app.py` AST PASS.

## How to Test Locally
- **Open directly in browser:** `open index.html`
- **Gradio Space mode:** `uv run python app.py` (port `7864`)
