"""Extension runs beyond the proposal grid (skips finished runs):

  stage twostage : DeiT-Tiny, fine-grained item head + rule table (last-block FT, aug, 100%, 3 seeds)
  stage backbones: DeiT-Small / DINOv2-S / CLIP ViT-B/32, linear probe and last-block FT, aug, 100%, 3 seeds
                   (linear head LR chosen per backbone on validation: 1e-3 / 3e-3 / 1e-2, seed 0, no aug)
  stage augs     : DeiT-Tiny last-block FT with 'strong' and 'bgswap' augmentation, 100%, 3 seeds
  stage targeted : DeiT-Tiny last-block FT + aug with extra training splits (extra_film / extra_random)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS, RUNS

HERE = Path(__file__).resolve().parent


def launch(name: str, extra: list[str]) -> None:
    if (RUNS / name / "metrics.json").exists():
        print(f"skip {name}")
        return
    cmd = [sys.executable, "-I", str(HERE / "train.py"), "--name", name, "--epochs", "40"] + extra
    print("RUN", name, flush=True)
    subprocess.run(cmd, check=True)


def val_f1(name: str) -> float:
    return json.loads((RUNS / name / "metrics.json").read_text())["val"]["macro_f1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["twostage", "backbones", "augs"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    if "twostage" in args.stages:
        for s in args.seeds:
            launch(f"fine_lastblock_brightrot_f1.0_s{s}",
                   ["--target", "fine", "--model", "lastblock", "--aug", "brightrot", "--frac", "1.0", "--seed", str(s),
                    "--lr_head", "1e-3", "--lr_block", "1e-4"])

    if "backbones" in args.stages:
        chosen = {}
        for b in ["deit_small", "dinov2_small", "clip_b32"]:
            for lr in ["1e-3", "3e-3", "1e-2"]:
                launch(f"sweep_{b}_linear_lr{lr}", ["--backbone", b, "--model", "linear", "--aug", "none", "--frac", "1.0",
                                                   "--seed", "0", "--lr_head", lr])
            chosen[b] = max(["1e-3", "3e-3", "1e-2"], key=lambda lr: val_f1(f"sweep_{b}_linear_lr{lr}"))
            print(f"linear head LR for {b}: {chosen[b]}")
        (RESULTS / "backbone_lr.json").write_text(json.dumps(chosen, indent=2))
        for s in args.seeds:
            for b in ["deit_small", "dinov2_small", "clip_b32"]:
                launch(f"{b}_linear_brightrot_f1.0_s{s}",
                       ["--backbone", b, "--model", "linear", "--aug", "brightrot", "--frac", "1.0", "--seed", str(s),
                        "--lr_head", chosen[b]])
                launch(f"{b}_lastblock_brightrot_f1.0_s{s}",
                       ["--backbone", b, "--model", "lastblock", "--aug", "brightrot", "--frac", "1.0", "--seed", str(s),
                        "--lr_head", "1e-3", "--lr_block", "1e-4"])

    if "augs" in args.stages:
        for s in args.seeds:
            for aug in ["strong", "bgswap"]:
                launch(f"lastblock_{aug}_f1.0_s{s}",
                       ["--model", "lastblock", "--aug", aug, "--frac", "1.0", "--seed", str(s),
                        "--lr_head", "1e-3", "--lr_block", "1e-4"])

    if "targeted" in args.stages:
        for s in args.seeds:
            for extra in ["extra_film", "extra_random"]:
                launch(f"lastblock_brightrot_{extra}_s{s}",
                       ["--model", "lastblock", "--aug", "brightrot", "--frac", "1.0", "--seed", str(s),
                        "--lr_head", "1e-3", "--lr_block", "1e-4", "--extra_splits", extra])
    print("extensions complete")


if __name__ == "__main__":
    main()
