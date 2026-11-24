# Plan: 03 — Vortex Rollercoaster (First-Person 3D Tunnel Ride)

## Concept
Place the user **inside** a neon 3D tunnel (`THREE.TubeGeometry`) shaped along the real 11-parameter spiral trajectory. Tunnel wall rings glow hotter (cyan → orange → blinding crimson) and the cockpit camera shakes proportionally to instantaneous kinetic energy `h(x, s)` as you blast through the 40-cycle oscillations and `0.008`-width Gaussian spikes.

## Architecture
- `game_data.json`: Real 20 held-out trajectories from `nsf_core.py` + `checkpoints/nsf_seed0.pt`.
- `index.html`: Standalone 60fps Three.js first-person cockpit + chase-cam toggle, Play/Pause/Scrub controls, `[s0, s1]` gate markers inside the tunnel, live windshield **Cumulative Prefix Meter $\tilde{h}_\theta(s)$** vs. noisy **MC Radar**, and instant 2-eval interval subtraction HUD.
- `app.py`: Hugging Face Gradio Space wrapper.
