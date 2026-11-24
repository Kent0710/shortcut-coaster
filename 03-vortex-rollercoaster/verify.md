# Verification — 03: Vortex Rollercoaster

## Automated Checks (`../test_3d_games.py`)
- **Status:** PASS (2026-11-24)
- **Data Integrity:** `game_data.js` (436.8 KB) parsed cleanly — 20 real held-out trajectories × 601 points from verified checkpoint `checkpoints/nsf_seed0.pt`.
- **NSF Accuracy:** Full-domain mean relative error = `0.0199`; `[0.20, 0.85]` 2-eval interval mean relative error = `0.0378`.
- **JS Syntax:** Verified via `node --check` (0 syntax errors).
- **Gradio Wrapper:** `app.py` AST verified (`build_embedded_html`, `create_app`).
- **UX Guarantee:** Pure client-side 60fps Three.js WebGL — zero loading overlay when scrubbing position or moving `[s0, s1]` portals.

## How to Play Locally
- **Fastest (double-click or `open`):** `open index.html`
- **Local HTTP server:** `python3 -m http.server 8083` → visit `http://localhost:8083`
- **Gradio Space mode:** `uv run python app.py` → visit `http://localhost:7863`
