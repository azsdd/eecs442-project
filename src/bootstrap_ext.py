"""Bootstrap 95% CIs for the extension comparisons (two-stage, backbones, CLIP zero-shot, targeted annotation).
Systems with several seeds are seed-averaged per resample; zero-shot systems have one run.
Writes results/bootstrap_ext.csv."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import BINS, RESULTS, RUNS

B = 2000
IDX = {b: i for i, b in enumerate(BINS)}
SETS = {"test": "preds_test.csv", "ood_test": "preds_ood_test.csv"}
SYSTEMS = {
    "DeiT-Tiny direct (main model)": ("lastblock_brightrot_f1.0_s{}", [0, 1, 2]),
    "DeiT-Tiny two-stage item + rule": ("fine_lastblock_brightrot_f1.0_s{}", [0, 1, 2]),
    "DeiT-Small last-block FT": ("deit_small_lastblock_brightrot_f1.0_s{}", [0, 1, 2]),
    "DINOv2-S last-block FT": ("dinov2_small_lastblock_brightrot_f1.0_s{}", [0, 1, 2]),
    "CLIP ViT-B/32 last-block FT": ("clip_b32_lastblock_brightrot_f1.0_s{}", [0, 1, 2]),
    "CLIP ViT-B/32 zero-shot items": ("clipzs_vitb32_openai_items", [None]),
    "CLIP ViT-L/14 zero-shot items": ("clipzs_vitl14_openai_items", [None]),
    "+19 targeted film images": ("lastblock_brightrot_extra_film_s{}", [0, 1, 2]),
    "+19 random images": ("lastblock_brightrot_extra_random_s{}", [0, 1, 2]),
}
PAIRS = [("DeiT-Tiny two-stage item + rule", "DeiT-Tiny direct (main model)"),
         ("DeiT-Small last-block FT", "DeiT-Tiny direct (main model)"),
         ("DINOv2-S last-block FT", "DeiT-Tiny direct (main model)"),
         ("CLIP ViT-B/32 last-block FT", "DeiT-Tiny direct (main model)"),
         ("CLIP ViT-B/32 zero-shot items", "DeiT-Tiny direct (main model)"),
         ("CLIP ViT-L/14 zero-shot items", "DeiT-Tiny direct (main model)"),
         ("CLIP ViT-L/14 zero-shot items", "DINOv2-S last-block FT"),
         ("+19 targeted film images", "DeiT-Tiny direct (main model)"),
         ("+19 random images", "DeiT-Tiny direct (main model)"),
         ("+19 targeted film images", "+19 random images")]


def preds(pattern, seeds, split):
    ys, ps = [], []
    for s in seeds:
        d = pd.read_csv(RUNS / (pattern.format(s) if s is not None else pattern) / SETS[split])
        ys.append(d["true"].map(IDX).to_numpy())
        ps.append(d["pred"].map(IDX).to_numpy())
    return np.stack(ys), np.stack(ps)


def stat(y, p, idx):
    accs, f1s = [], []
    for s in range(len(y)):
        yy, pp = y[s][idx], p[s][idx]
        accs.append((yy == pp).mean())
        f1s.append(f1_score(yy, pp, labels=[0, 1, 2], average="macro", zero_division=0))
    return float(np.mean(accs)), float(np.mean(f1s))


def main() -> None:
    rng = np.random.default_rng(442)
    rows = []
    for split in SETS:
        data = {}
        for name, (pattern, seeds) in SYSTEMS.items():
            try:
                data[name] = preds(pattern, seeds, split)
            except FileNotFoundError:
                print("missing", name)
        n = next(iter(data.values()))[0].shape[1]
        boots = rng.integers(0, n, size=(B, n))
        est = {c: np.array([stat(*data[c], idx) for idx in boots]) for c in data}
        for c in data:
            a0, f0 = stat(*data[c], np.arange(n))
            rows.append({"split": split, "kind": "system", "name": c, "acc": a0, "acc_lo": np.percentile(est[c][:, 0], 2.5),
                         "acc_hi": np.percentile(est[c][:, 0], 97.5), "f1": f0, "f1_lo": np.percentile(est[c][:, 1], 2.5),
                         "f1_hi": np.percentile(est[c][:, 1], 97.5)})
        for a, b in PAIRS:
            if a not in data or b not in data:
                continue
            d = est[a] - est[b]
            a0 = stat(*data[a], np.arange(n))
            b0 = stat(*data[b], np.arange(n))
            rows.append({"split": split, "kind": "diff", "name": f"{a} vs {b}", "acc": a0[0] - b0[0],
                         "acc_lo": np.percentile(d[:, 0], 2.5), "acc_hi": np.percentile(d[:, 0], 97.5), "f1": a0[1] - b0[1],
                         "f1_lo": np.percentile(d[:, 1], 2.5), "f1_hi": np.percentile(d[:, 1], 97.5), "p_f1_le_0": float((d[:, 1] <= 0).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "bootstrap_ext.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
