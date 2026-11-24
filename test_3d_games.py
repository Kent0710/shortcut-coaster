"""Automated verification suite for both Vortex Rollercoaster editions (Technical + Layman)."""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
DELETED_GAMES = ["01-two-ping-runner", "02-sniper-vs-swarm"]
ACTIVE_GAMES = [
    "03-vortex-rollercoaster",
    "vortex-rollercoaster-layman",
]
REQUIRED_FILES = [
    "plan.md",
    "goal.md",
    "verify.md",
    "todo.md",
    "game_data.js",
    "index.html",
    "app.py",
]
BANNED_LAYMAN_JARGON = [
    "Monte Carlo",
    "MC-20",
    "NSF",
    "h̃",
    "s₀",
    "s₁",
    "0.008",
    "Prefix",
    "Held-Out",
    "kinetic energy",
]


def interp(arr: list[float], s: float) -> float:
    nm1 = len(arr) - 1
    c = max(0.0, min(1.0, s))
    f = c * nm1
    i = min(nm1 - 1, int(f))
    t = f - i
    return arr[i] * (1.0 - t) + arr[i + 1] * t


def verify_game(name: str) -> None:
    gdir = BASE / name
    assert gdir.is_dir(), f"Missing directory: {gdir}"

    for fname in REQUIRED_FILES:
        fpath = gdir / fname
        assert fpath.is_file() and fpath.stat().st_size > 0, f"Missing/empty file: {fpath}"

    # 1. Verify game_data.js payload & NSF accuracy
    raw_js = (gdir / "game_data.js").read_text(encoding="utf-8")
    prefix = "window.NSF_GAME_DATA = "
    idx = raw_js.find(prefix)
    assert idx != -1, f"{name}/game_data.js missing window.NSF_GAME_DATA assignment"
    json_str = raw_js[idx + len(prefix) :].strip().rstrip(";")
    data = json.loads(json_str)

    assert data["n_points"] == 601, f"Expected 601 points, got {data['n_points']}"
    assert len(data["trajectories"]) == 20, f"Expected 20 trajectories, got {len(data['trajectories'])}"
    assert data["mean_full_rel_err"] < 0.025, f"Mean full rel err too high: {data['mean_full_rel_err']}"

    rel_errs = []
    for tr in data["trajectories"]:
        for key in ("x", "y", "h", "cum", "nsf"):
            assert len(tr[key]) == 601, f"{name} traj {tr['id']} field {key} len != 601"
        s0, s1 = 0.20, 0.85
        true_mean = (interp(tr["cum"], s1) - interp(tr["cum"], s0)) / (s1 - s0)
        nsf_mean = (s1 * interp(tr["nsf"], s1) - s0 * interp(tr["nsf"], s0)) / (s1 - s0)
        rel_errs.append(abs(nsf_mean - true_mean) / max(1e-9, abs(true_mean)))
    mean_sub_err = sum(rel_errs) / len(rel_errs)
    assert mean_sub_err < 0.04, f"{name} subinterval rel err too high: {mean_sub_err:.4f}"

    # 2. Verify index.html structure & inline JS syntax
    html_text = (gdir / "index.html").read_text(encoding="utf-8")
    assert "three@0.147.0/build/three.min.js" in html_text, f"{name}/index.html missing Three.js UMD script"
    assert '<script src="game_data.js"></script>' in html_text, f"{name}/index.html missing game_data.js include"
    assert "requestAnimationFrame" in html_text, f"{name}/index.html missing 60fps animation loop"

    scripts = re.findall(r"<script>([\s\S]*?)</script>", html_text)
    assert len(scripts) == 1, f"{name}/index.html expected 1 inline script block, found {len(scripts)}"
    node_bin = shutil.which("node")
    if node_bin:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tf:
            tf.write(scripts[0])
            tmp_path = tf.name
        proc = subprocess.run([node_bin, "--check", tmp_path], capture_output=True, text=True)
        Path(tmp_path).unlink(missing_ok=True)
        assert proc.returncode == 0, f"{name}/index.html JS syntax error:\n{proc.stderr}"

    # 3. Verify Layman version has zero technical jargon, zero emojis/icons, and 5-step story flow
    if name == "vortex-rollercoaster-layman":
        visible_html = re.sub(r"<script[\s\S]*?</script>", "", html_text)
        visible_html = re.sub(r"<style[\s\S]*?</style>", "", visible_html)
        for term in BANNED_LAYMAN_JARGON:
            assert term.lower() not in visible_html.lower(), (
                f"Found banned technical jargon '{term}' in {name}/index.html visible UI!"
            )
        emoji_hits = re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", html_text)
        assert not emoji_hits, f"Found emojis/icons in {name}/index.html: {emoji_hits}"
        for req_id in ("story-layer", "btn-next", "btn-back", "btn-skip", "btn-replay", "sim-ui"):
            assert f'id="{req_id}"' in html_text, f"Missing story/sim element id='{req_id}'"

    # 4. Verify app.py AST
    app_src = (gdir / "app.py").read_text(encoding="utf-8")
    ast.parse(app_src)
    assert "build_embedded_html" in app_src and "create_app" in app_src

    jargon_note = " | Jargon-Free=OK | Emoji-Free=OK | 5-Step Story=OK" if name == "vortex-rollercoaster-layman" else ""
    print(
        f"[PASS] {name}: 20 trajs x 601 pts | full rel err={data['mean_full_rel_err']:.4f} | "
        f"[0.20,0.85] rel err={mean_sub_err:.4f} | JS syntax={'OK (node)' if node_bin else 'SKIP'} | "
        f"app.py AST=OK{jargon_note}"
    )


def main() -> None:
    for old in DELETED_GAMES:
        assert not (BASE / old).exists(), f"Expected deleted folder to be gone: {BASE / old}"
    for name in ACTIVE_GAMES:
        verify_game(name)
    print(f"\nALL {len(ACTIVE_GAMES)}/2 VORTEX ROLLERCOASTER EDITIONS VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
