"""Run the full 2 (model) x 2 (aug) x 2 (data fraction) x 3 (seed) grid; skips finished runs.

Usage: python -I src/run_grid.py --lr_head_linear 3e-3 --lr_head_ft 1e-3 --lr_block 1e-4 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RUNS

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_head_linear", type=float, default=1e-3)
    ap.add_argument("--lr_head_ft", type=float, default=1e-3)
    ap.add_argument("--lr_block", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", default="", help="substring filter on run name")
    args = ap.parse_args()

    for seed, model, aug, frac in product(args.seeds, ["linear", "lastblock"], ["none", "brightrot"], [1.0, 0.25]):
        name = f"{model}_{aug}_f{frac}_s{seed}"
        if args.only and args.only not in name:
            continue
        if (RUNS / name / "metrics.json").exists():
            print(f"skip {name}")
            continue
        lr_head = args.lr_head_linear if model == "linear" else args.lr_head_ft
        cmd = [sys.executable, "-I", str(HERE / "train.py"), "--model", model, "--aug", aug, "--frac", str(frac),
               "--seed", str(seed), "--epochs", str(args.epochs), "--lr_head", str(lr_head),
               "--lr_block", str(args.lr_block), "--name", name]
        print("RUN", name, flush=True)
        subprocess.run(cmd, check=True)
    print("grid complete")


if __name__ == "__main__":
    main()
