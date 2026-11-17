"""Predict the U-M bin for one photo from the command line.

  python -I src/predict.py --image path/to/photo.jpg [--run results/runs/<run name>] [--out fig.png]

Works for 3-way runs and for two-stage (fine item head + rule table) runs, with any backbone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image

from common import BINS, FINE_CLASSES, FINE_TO_BIN, RESULTS, RUNS
from train import build_model, build_transforms, model_norm, pil_to_tensor256, to_bin_probs

BIN_TEXT = {
    "recycling": "Recycling bin (paper, cardboard, cartons, metal cans, rigid plastics #1-2, #4-7)",
    "compost": "Compost bin (food scraps incl. meat, napkins, paper towels, BPI-certified ware, plant material)",
    "landfill": "Landfill bin (glass, plastic film and bags, foam, wrappers, textiles, mixed items)",
}


def default_run() -> Path:
    s = RESULTS / "summary.json"
    if s.exists():
        b = json.loads(s.read_text())["best_config"]
        cand = RUNS / f"{b['model']}_{b['aug']}_f{b['frac']}_s0"
        if cand.exists():
            return cand
    runs = sorted(p for p in RUNS.glob("lastblock_*_f1.0_s0") if (p / "trainable_state.pt").exists())
    return runs[-1]


def load_model(run_dir: Path, device: str = "cpu"):
    cfg = json.loads((run_dir / "metrics.json").read_text())["config"]
    target = cfg.get("target", "bin")
    n = len(FINE_CLASSES) if target == "fine" else len(BINS)
    model = build_model(cfg["model"], cfg.get("backbone", "deit_tiny"), n)
    state = torch.load(run_dir / "trainable_state.pt", map_location="cpu")
    _, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    return model.eval().to(device), cfg


@torch.no_grad()
def predict_logits(model, im: Image.Image, device: str = "cpu") -> np.ndarray:
    mean, std = model_norm(model)
    _, eval_tf = build_transforms("none", 0.0, 0.0, mean, std)
    x = eval_tf(pil_to_tensor256(im)).unsqueeze(0).to(device)
    return model(x).float()[0].cpu().numpy()


def predict_pil(model, im: Image.Image, device: str = "cpu", target: str = "bin", temperature: float = 1.0):
    """Returns (bin_probs, head_probs): head_probs are the fine item probabilities for two-stage runs."""
    z = predict_logits(model, im, device) / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    return to_bin_probs(p[None], target)[0], p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--run", default="")
    ap.add_argument("--out", default="", help="optional path for a figure with the photo and class scores")
    args = ap.parse_args()
    run_dir = Path(args.run) if args.run else default_run()
    model, cfg = load_model(run_dir)
    target = cfg.get("target", "bin")
    im = Image.open(args.image)
    probs, head = predict_pil(model, im, target=target)
    k = int(probs.argmax())
    print(f"model: {run_dir.name}")
    if target == "fine":
        j = int(head.argmax())
        print(f"item type: {FINE_CLASSES[j]} ({head[j]:.1%}) -> campus rule: {FINE_TO_BIN[FINE_CLASSES[j]]}")
    print(f"prediction: {BINS[k].upper()}  ({probs[k]:.1%})")
    for b, p in sorted(zip(BINS, probs), key=lambda t: -t[1]):
        print(f"  {b:10s} {p:6.1%}")
    print(BIN_TEXT[BINS[k]])
    if args.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw=dict(width_ratios=[1, 1.2]))
        a1.imshow(im.convert("RGB"))
        a1.axis("off")
        colors = ["#2a78d6" if i == k else "#c3c2b7" for i in range(3)]
        a2.barh(BINS, probs, color=colors, height=0.55)
        for i, p in enumerate(probs):
            a2.text(min(p + 0.02, 0.9), i, f"{p:.1%}", va="center", fontsize=9)
        a2.set_xlim(0, 1)
        a2.invert_yaxis()
        a2.set_title(f"Predicted bin: {BINS[k]}", fontsize=11, fontweight="bold")
        for s in ("top", "right"):
            a2.spines[s].set_visible(False)
        a2.tick_params(length=0)
        fig.tight_layout()
        fig.savefig(args.out, dpi=200)
        print("figure saved to", args.out)


if __name__ == "__main__":
    main()
