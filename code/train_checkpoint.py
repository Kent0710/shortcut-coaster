#!/usr/bin/env python3
"""
Train and export the pretrained `checkpoints/nsf_seed0.pt` checkpoint (~3.2 MB)
used by the Gradio Explainer Space for instant CPU startup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from nsf_core import (
    PAPER_TABLE11_NSF,
    analytic_interval_mean,
    rel_l2,
    sample_designs,
    train_nsf,
)


def main() -> None:
    root = Path(__file__).resolve().parent
    ckpt_path = root / "checkpoints" / "nsf_seed0.pt"

    rng = np.random.default_rng(42)
    train_params = sample_designs(3000, rng)
    test_params = sample_designs(900, rng)

    t0 = time.time()
    wrapper = train_nsf(
        train_params,
        epochs=2000,
        batch_size=1024,
        width=512,
        depth=4,
        max_lr=1e-3,
        weight_decay=1e-2,
        pct_start=0.3,
        delta_s=1e-3,
        seed=0,
        device="cpu",
        log_fn=print,
    )
    train_s = time.time() - t0
    wrapper.save(ckpt_path)

    # Verify Table 11 parity on held-out 900 test designs
    widths = np.linspace(0.90, 1.00, 8)
    errs = []
    all_preds, all_refs = [], []
    for w in widths:
        intervals = [(0.0, 1.0)] if np.isclose(w, 1.0) else [(0.0, float(w)), (float(1.0 - w), 1.0)]
        refs = np.concatenate([analytic_interval_mean(test_params, s0, s1) for s0, s1 in intervals])
        preds = np.concatenate([wrapper.predict_interval(test_params, s0, s1) for s0, s1 in intervals])
        errs.append(rel_l2(preds, refs))
        all_preds.append(preds)
        all_refs.append(refs)

    y_p = np.concatenate(all_preds)
    y_r = np.concatenate(all_refs)
    r2_analytic = float(1.0 - np.sum((y_p - y_r) ** 2) / np.sum((y_r - np.mean(y_r)) ** 2))
    mae_vs_paper = float(np.mean(np.abs(np.array(errs) - PAPER_TABLE11_NSF)))
    size_mb = ckpt_path.stat().st_size / (1024 * 1024)

    summary = {
        "checkpoint": str(ckpt_path.name),
        "size_mb": round(size_mb, 3),
        "train_seconds": round(train_s, 2),
        "table11_rel_l2": errs,
        "table11_mae_vs_paper": mae_vs_paper,
        "r2_vs_analytic": r2_analytic,
    }
    (root / "checkpoints" / "nsf_seed0_meta.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Saved {ckpt_path} ({size_mb:.2f} MB) in {train_s:.1f}s | "
        f"Table 11 MAE vs paper = {mae_vs_paper:.5f} | R^2 = {r2_analytic:.6f}"
    )


if __name__ == "__main__":
    main()
