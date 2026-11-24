# Todo: Interactive Gradio Explainer & Open Artifact (`best-explainer`)

- [x] Scaffold `best-explainer/` with `plan.md`, `todo.md`, `verify.md`, `history/`, `learnings/`, and `code/`.
- [x] Build reusable PyTorch Open Artifact (`code/nsf_core.py`) with exact parity to `exp/exact-repro`.
- [x] Train & export lightweight pretrained checkpoint (`code/checkpoints/nsf_seed0.pt`, `3.04 MB`) via `train_checkpoint.py`.
- [x] Bundle verified 5-seed Tables 10 & 11 results (`code/data/verified_benchmarks.json`).
- [x] Build headless-testable visualization engine (`code/explainer_viz.py`) for all 3 tabs.
- [x] Build Gradio 5 Space entrypoint (`code/app.py`), HF Space `README.md` YAML header, `pyproject.toml`, and `requirements.txt`.
- [x] Run automated test suite (`code/test_explainer.py`) & visually inspect rendered preview PNGs for all 3 tabs.
- [x] Initialize local git repo in `best-explainer/code/` and commit clean milestone.
- [ ] Optional: `pip install gradio` in an online environment and push `best-explainer/code/` to Hugging Face Spaces once Nov 30 submission opens.
