"""Attention-rollout maps (Abnar & Zuidema, 2020) for a few test images of the best run.

Shows where the class token attends for three confident correct predictions and three confident mistakes
on each test set. Writes results/figures/fig_attention.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import BINS, FIGURES, RESULTS, ROOT, RUNS, REVIEWED, read_csv
from predict import load_model
from train import build_transforms, pil_to_tensor256


def attention_rollout(model, x: torch.Tensor) -> np.ndarray:
    attns = []
    hooks = []
    for blk in model.blocks:
        blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: attns.append(o.detach().cpu())))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    n = attns[0].shape[-1]
    result = torch.eye(n)
    for a in attns:
        a = a.mean(1)[0]
        a = a + torch.eye(n)
        a = a / a.sum(-1, keepdim=True)
        result = a @ result
    mask = result[0, 1:].reshape(14, 14).numpy()
    return (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)


def pick(preds: pd.DataFrame, k: int = 3):
    preds = preds.copy()
    preds["conf"] = preds.apply(lambda r: r[f"p_{r['pred']}"], axis=1)
    ok = preds[preds["true"] == preds["pred"]]
    correct = pd.concat([ok[ok["true"] == b].sort_values("conf", ascending=False).head(1) for b in BINS]).head(k)
    wrong = preds[preds["true"] != preds["pred"]].sort_values("conf", ascending=False).head(k)
    return pd.concat([correct, wrong])


def main() -> None:
    best = json.loads((RESULTS / "summary.json").read_text())["best_config"]
    run = RUNS / f"{best['model']}_{best['aug']}_f{best['frac']}_s0"
    model, _ = load_model(run)
    splits = {r["cid"]: r for r in read_csv(REVIEWED / "splits.csv")}
    _, eval_tf = build_transforms("none", 0, 0)
    sets = [("RealWaste test", pd.read_csv(run / "preds_test.csv")), ("Proxy new-photo test", pd.read_csv(run / "preds_ood_test.csv"))]
    fig, axes = plt.subplots(4, 6, figsize=(13, 9.2))
    for si, (name, preds) in enumerate(sets):
        chosen = pick(preds)
        for j, r in enumerate(chosen.itertuples()):
            im = Image.open(ROOT / splits[r.cid]["file"]).convert("RGB")
            x256 = pil_to_tensor256(im)
            x = eval_tf(x256).unsqueeze(0)
            mask = attention_rollout(model, x)
            crop = x256[:, 16:240, 16:240].permute(1, 2, 0).numpy()
            mask_up = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
            ax_img, ax_map = axes[2 * si, j], axes[2 * si + 1, j]
            ax_img.imshow(crop)
            ax_map.imshow(crop)
            ax_map.imshow(mask_up, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
            tag = "correct" if r.true == r.pred else "WRONG"
            ax_img.set_title(f"{r.orig_class}\n{r.true} → {r.pred} ({r.conf:.2f})\n{tag}", fontsize=7.5)
            for ax in (ax_img, ax_map):
                ax.axis("off")
        axes[2 * si, 0].text(-0.08, 0.5, name, transform=axes[2 * si, 0].transAxes, rotation=90, va="center",
                             ha="right", fontsize=10, fontweight="bold")
    fig.suptitle(f"Class-token attention rollout, {run.name}: 3 confident correct (left) and 3 confident wrong (right) per set",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_attention.png", dpi=170)
    print("saved", FIGURES / "fig_attention.png")


if __name__ == "__main__":
    main()
