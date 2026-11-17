"""Robustness check: evaluate every finished grid run on brightness- and rotation-shifted copies of the
RealWaste test split (synthetic "different lighting / orientation").

Writes results/shift_eval.csv with one row per (run, shift): accuracy and macro-F1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as TF

from common import IMG_SIZE, MEAN, RESULTS, REVIEWED, RUNS, STD, read_csv
from train import build_model, metrics, preload

SHIFTS = {
    "none": lambda x: x,
    "bright_0.5": lambda x: TF.adjust_brightness(x, 0.5),
    "bright_1.5": lambda x: TF.adjust_brightness(x, 1.5),
    "rot_30": lambda x: TF.rotate(x, 30, interpolation=InterpolationMode.BILINEAR),
    "rot_-30": lambda x: TF.rotate(x, -30, interpolation=InterpolationMode.BILINEAR),
    "rot_90": lambda x: TF.rotate(x, 90),
    "rot_180": lambda x: TF.rotate(x, 180),
}


@torch.no_grad()
def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_csv(REVIEWED / "splits.csv")
    test = [r for r in rows if r["split"] == "test"]
    x, y = preload(test)
    base = v2.Compose([v2.CenterCrop(IMG_SIZE), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
    shifted = {}
    for name, fn in SHIFTS.items():
        shifted[name] = torch.stack([base(fn(x[i])) for i in range(len(x))]).to(dev)
    out = []
    runs = sorted(p for p in RUNS.glob("*") if (p / "metrics.json").exists() and not p.name.startswith(("sweep", "smoke")))
    for run in runs:
        cfg = json.loads((run / "metrics.json").read_text())["config"]
        model = build_model(cfg["model"])
        model.load_state_dict(torch.load(run / "trainable_state.pt", map_location="cpu"), strict=False)
        model = model.eval().to(dev)
        for name, xb in shifted.items():
            probs = torch.cat([torch.softmax(model(xb[i:i + 64]).float(), -1) for i in range(0, len(xb), 64)])
            m = metrics(y.numpy(), probs.argmax(1).cpu().numpy())
            out.append({"run": run.name, "model": cfg["model"], "aug": cfg["aug"], "frac": cfg["frac"], "seed": cfg["seed"],
                        "shift": name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        print(run.name, {o["shift"]: round(o["macro_f1"], 3) for o in out if o["run"] == run.name})
    df = pd.DataFrame(out)
    df.to_csv(RESULTS / "shift_eval.csv", index=False)
    piv = df[df.frac == 1.0].groupby(["model", "aug", "shift"])["macro_f1"].mean().unstack("shift")
    piv = piv[list(SHIFTS)]
    piv.to_csv(RESULTS / "shift_eval_summary.csv")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
