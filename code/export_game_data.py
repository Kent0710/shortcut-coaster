"""Export 20 real held-out trajectories + verified NSF prefix curves to game_data.js for all 3 3D games."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nsf_core import (
    PrefixStatisticWrapper,
    analytic_interval_mean,
    instantaneous_energy_np,
    sample_designs,
    trajectory_polar_np,
)

CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
CKPT_PATH = CODE_DIR / "checkpoints" / "nsf_seed0.pt"

TARGETS = [
    BASE_DIR / "03-vortex-rollercoaster" / "game_data.js",
    BASE_DIR / "vortex-rollercoaster-layman" / "game_data.js",
]


def main() -> None:
    wrapper = PrefixStatisticWrapper.load(CKPT_PATH, device="cpu")
    rng = np.random.default_rng(42)
    _ = sample_designs(3000, rng)
    test_pool = sample_designs(900, rng)

    # Pick 20 diverse held-out test trajectories (including the 4 named presets)
    indices = [0, 7, 14, 23, 31, 42, 55, 68, 81, 95, 112, 137, 164, 199, 240, 310, 405, 512, 678, 841]
    s_grid = np.linspace(0.0, 1.0, 601)
    fine_s = np.linspace(0.0, 1.0, 4097)
    ds_fine = fine_s[1] - fine_s[0]

    trajectories = []
    rel_l2_checks = []
    for idx in indices:
        p = test_pool[idx]
        x, y, _, _, h = trajectory_polar_np(p, s_grid)

        # High-precision cumulative integral on 4097 pts interpolated onto s_grid
        fine_h = instantaneous_energy_np(p[None, :], fine_s[None, :])[0]
        cum_fine = np.zeros_like(fine_s)
        cum_fine[1:] = np.cumsum(0.5 * (fine_h[:-1] + fine_h[1:]) * ds_fine)
        cum_grid = np.interp(s_grid, fine_s, cum_fine)

        # Real PyTorch NSF prefix predictions on s_grid (s=0 uses limit h(x, 0))
        s_pos = s_grid[1:]
        nsf_pos = wrapper.predict_prefix(np.repeat(p[None, :], len(s_pos), axis=0), s_pos)
        nsf_prefix = np.concatenate([[float(h[0])], nsf_pos])

        # Verify full-domain accuracy
        true_full = float(analytic_interval_mean(p[None, :], 0.0, 1.0)[0])
        nsf_full = float(nsf_prefix[-1])
        rel_l2_checks.append(abs(nsf_full - true_full) / max(abs(true_full), 1e-9))

        trajectories.append(
            {
                "id": int(idx),
                "label": f"Held-Out #{idx}",
                "mu_r": round(float(p[7]), 4),
                "mu_w": round(float(p[8]), 4),
                "h_max": round(float(h.max()), 2),
                "h_mean_full": round(true_full, 3),
                "nsf_full": round(nsf_full, 3),
                "x": [round(float(v), 4) for v in x],
                "y": [round(float(v), 4) for v in y],
                "h": [round(float(v), 2) for v in h],
                "cum": [round(float(v), 4) for v in cum_grid],
                "nsf": [round(float(v), 3) for v in nsf_prefix],
            }
        )

    payload = {
        "n_points": len(s_grid),
        "s": [round(float(v), 5) for v in s_grid],
        "mean_full_rel_err": round(float(np.mean(rel_l2_checks)), 5),
        "trajectories": trajectories,
    }
    js_content = "// Auto-generated from verified NSF checkpoint (checkpoints/nsf_seed0.pt)\nwindow.NSF_GAME_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"

    for t in TARGETS:
        t.write_text(js_content)
        print(f"Wrote {t} ({len(js_content)/1024:.1f} KB)")
    print(f"Exported {len(trajectories)} trajectories | mean NSF full-domain rel err = {payload['mean_full_rel_err']:.4f}")


if __name__ == "__main__":
    main()
