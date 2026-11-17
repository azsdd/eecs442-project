"""Bootstrap confidence intervals over test images (seed-averaged) for the main configurations and
paired comparisons; reads saved prediction files. Writes results/bootstrap.csv."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import BINS, RESULTS, RUNS

B = 2000
SEEDS = [0, 1, 2]
SETS = {"test": "preds_test.csv", "ood_test": "preds_ood_test.csv"}
IDX = {b: i for i, b in enumerate(BINS)}


def preds(config: str, split: str):
    ys, ps = [], []
    for s in SEEDS:
        d = pd.read_csv(RUNS / f"{config}_s{s}" / SETS[split])
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
    configs = ["lastblock_brightrot_f1.0", "lastblock_none_f1.0", "linear_brightrot_f1.0", "linear_none_f1.0",
               "lastblock_brightrot_f0.25", "lastblock_none_f0.25"]
    pairs = [("lastblock_brightrot_f1.0", "linear_none_f1.0", "main model vs plain linear probe"),
             ("lastblock_brightrot_f1.0", "lastblock_none_f1.0", "augmentation (last-block FT, 100%)"),
             ("lastblock_brightrot_f1.0", "linear_brightrot_f1.0", "last-block FT vs linear (aug, 100%)"),
             ("lastblock_brightrot_f1.0", "lastblock_brightrot_f0.25", "100% vs 25% data (last-block FT, aug)"),
             ("lastblock_brightrot_f0.25", "lastblock_none_f0.25", "augmentation (last-block FT, 25%)")]
    rows = []
    for split in SETS:
        data = {c: preds(c, split) for c in configs}
        n = data[configs[0]][0].shape[1]
        boots = rng.integers(0, n, size=(B, n))
        est = {c: np.array([stat(*data[c], idx) for idx in boots]) for c in configs}
        for c in configs:
            acc0, f10 = stat(*data[c], np.arange(n))
            rows.append({"split": split, "kind": "config", "name": c, "acc": acc0,
                         "acc_lo": np.percentile(est[c][:, 0], 2.5), "acc_hi": np.percentile(est[c][:, 0], 97.5),
                         "f1": f10, "f1_lo": np.percentile(est[c][:, 1], 2.5), "f1_hi": np.percentile(est[c][:, 1], 97.5)})
        for a, b, label in pairs:
            d = est[a] - est[b]
            acc0 = stat(*data[a], np.arange(n))[0] - stat(*data[b], np.arange(n))[0]
            f10 = stat(*data[a], np.arange(n))[1] - stat(*data[b], np.arange(n))[1]
            rows.append({"split": split, "kind": "diff", "name": label, "acc": acc0,
                         "acc_lo": np.percentile(d[:, 0], 2.5), "acc_hi": np.percentile(d[:, 0], 97.5),
                         "f1": f10, "f1_lo": np.percentile(d[:, 1], 2.5), "f1_hi": np.percentile(d[:, 1], 97.5),
                         "p_f1_le_0": float((d[:, 1] <= 0).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "bootstrap.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
