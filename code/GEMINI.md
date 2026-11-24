# Agent Guide: Gradio Explainer & Open Artifact (`best-explainer`)

## Mission
Interactive Hugging Face Gradio Space + reusable PyTorch Open Artifact (`nsf_core.py`) for the **Best Explainer**, **Open Artifact**, and **Community Choice** tracks of the NeurIPS 2026 ML Intern Reproductions hackathon.

## Layout
- `nsf_core.py`: Clean, documented, reusable PyTorch library (Open Artifact)
- `train_checkpoint.py`: Reproducible training script exporting `checkpoints/nsf_seed0.pt` (~3.2 MB)
- `explainer_viz.py`: Headless-testable Matplotlib figure & Markdown table callbacks for all 3 tabs
- `app.py`: Gradio 5 Space entrypoint (`gr.Blocks`)
- `data/verified_benchmarks.json`: Verified 5-seed Tables 10 & 11 results from `exp/exact-repro`
- `test_explainer.py`: Comprehensive unit, parity, checkpoint accuracy, and callback tests

## Commands
```bash
python train_checkpoint.py
python test_explainer.py
python app.py
```
