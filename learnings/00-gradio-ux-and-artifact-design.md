# Learnings 00: Decoupled Gradio Architecture & Visual Storytelling

1. **Decoupling Plotting from Gradio (`explainer_viz.py` vs. `app.py`):**
   Separating pure Matplotlib/NumPy/PyTorch callback functions into `explainer_viz.py` lets us unit-test every tab, edge case (like `s0 >= s1` slider crossing or `s0 == 0.0` boundary condition), and visually inspect rendered PNGs even in headless or offline CI environments without needing a live Gradio server.
2. **Pretrained Checkpoint Size & Startup UX:**
   A 4-hidden-layer width-512 SiLU MLP saves to just `3.04 MB` in float32, loading in `< 50 ms` on free HF Space CPUs so visitors never wait 60–150 seconds for training before interacting with sliders.
3. **Showing Aliasing Physically Beats Just Showing a Table:**
   Plotting the zoomed 40-cycle harmonic $1.5\gamma_\omega\sin(80\pi s + \phi_h)$ alongside the 40 equispaced Midpoint nodes makes the `0.019 -> 0.136` error spike immediately obvious at a single glance: at `w = 1.00`, every single node lands on the exact same phase of the wave.
4. **Always Use Held-Out Trajectories in Interactive Demos:**
   By sourcing all interactive presets and random trajectories from the exact 900 held-out test designs (`rng(42)` after skipping the 3,000 training designs), every live prediction in the explainer is genuinely out-of-sample.
