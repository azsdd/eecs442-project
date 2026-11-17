"""Confidence calibration (temperature scaling on the validation split) and selective prediction
(abstention) analysis for a family of runs, from their saved prediction files.

  python -I src/calibrate.py --runs "lastblock_brightrot_f1.0_s*" --tag main
Writes results/calibration_<tag>.json and results/figures/fig_abstention_<tag>.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BINS, FIGURES, RESULTS, RUNS

PAL = dict(blue="#2a78d6", orange="#eb6834", ink="#0b0b0b", ink2="#52514e", grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb")
SETS = {"test": "RealWaste test", "ood_test": "Proxy new-photo test"}


def load(run: Path, split: str):
    d = pd.read_csv(run / f"preds_{split}.csv")
    p = d[[f"p_{b}" for b in BINS]].to_numpy(dtype=float)
    y = d["true"].map({b: i for i, b in enumerate(BINS)}).to_numpy()
    return np.log(np.clip(p, 1e-12, 1.0)), y


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(logits, y, T):
    p = softmax(logits / T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def fit_temperature(logits, y) -> float:
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200))
    return float(min(grid, key=lambda T: nll(logits, y, T)))


def ece(conf, correct, n_bins=15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def risk_coverage(conf, correct):
    order = np.argsort(-conf)
    c = correct[order]
    cov = np.arange(1, len(c) + 1) / len(c)
    acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    aurc = float(np.mean(1 - acc))
    return cov, acc, aurc


def acc_at_coverage(conf, correct, target):
    cov, acc, _ = risk_coverage(conf, correct)
    k = int(np.searchsorted(cov, target))
    k = min(max(k, 0), len(acc) - 1)
    thr = float(np.sort(conf)[::-1][k])
    return float(acc[k]), thr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="lastblock_brightrot_f1.0_s*")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    runs = sorted(RUNS.glob(args.runs))
    assert runs, args.runs
    out = {"runs": [r.name for r in runs], "per_run": [], "pooled": {}}
    pooled = {s: {"conf_raw": [], "conf_cal": [], "correct": []} for s in SETS}
    for run in runs:
        lv, yv = load(run, "val")
        T = fit_temperature(lv, yv)
        rec = {"run": run.name, "temperature": T}
        for s in SETS:
            lg, y = load(run, s)
            p_raw, p_cal = softmax(lg), softmax(lg / T)
            pred = p_raw.argmax(1)
            correct = (pred == y).astype(float)
            cr, cc = p_raw.max(1), p_cal.max(1)
            rec[s] = {"ece_raw": ece(cr, correct), "ece_cal": ece(cc, correct),
                      "nll_raw": nll(lg, y, 1.0), "nll_cal": nll(lg, y, T),
                      "mean_conf_raw": float(cr.mean()), "mean_conf_cal": float(cc.mean()), "accuracy": float(correct.mean())}
            pooled[s]["conf_raw"].append(cr)
            pooled[s]["conf_cal"].append(cc)
            pooled[s]["correct"].append(correct)
        out["per_run"].append(rec)
    out["temperature_mean"] = float(np.mean([r["temperature"] for r in out["per_run"]]))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for s, lab in SETS.items():
        cr = np.concatenate(pooled[s]["conf_raw"])
        cc = np.concatenate(pooled[s]["conf_cal"])
        co = np.concatenate(pooled[s]["correct"])
        res = {"accuracy": float(co.mean()), "ece_raw": ece(cr, co), "ece_cal": ece(cc, co),
               "mean_conf_raw": float(cr.mean()), "mean_conf_cal": float(cc.mean())}
        cov, acc, aurc = risk_coverage(cc, co)
        res["aurc"] = aurc
        for t in (0.9, 0.8, 0.7, 0.5):
            a, thr = acc_at_coverage(cc, co, t)
            res[f"acc_at_cov{int(t * 100)}"] = a
            res[f"thr_at_cov{int(t * 100)}"] = thr
        for thr in (0.6, 0.7, 0.8, 0.9):
            m = cc >= thr
            res[f"cov_at_thr{int(thr * 100)}"] = float(m.mean())
            res[f"acc_at_thr{int(thr * 100)}"] = float(co[m].mean()) if m.any() else float("nan")
        out["pooled"][s] = res
        color = PAL["blue"] if s == "test" else PAL["orange"]
        keep = cov >= 0.1  # the first few most-confident photos make the curve jump around
        axes[0].plot(cov[keep], acc[keep], color=color, lw=1.8, label=lab)
        for t in (0.8,):
            a, _ = acc_at_coverage(cc, co, t)
            axes[0].plot([t], [a], marker="o", ms=5, color=color)
            axes[0].annotate(f"{a:.2f} at {int(t * 100)}%", (t, a), textcoords="offset points", xytext=(6, 6), fontsize=8, color=PAL["ink"])
        bins = np.linspace(0, 1, 11)
        for conf, ls, tag in ((cr, ":", "raw"), (cc, "-", "calibrated")):
            xs, ys = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.sum() >= 5:
                    xs.append(conf[m].mean())
                    ys.append(co[m].mean())
            axes[1].plot(xs, ys, color=color, ls=ls, marker="o", ms=4, lw=1.4, label=f"{lab}, {tag}")
    axes[0].set_xlabel("coverage (fraction of photos answered)")
    axes[0].set_ylabel("accuracy on answered photos")
    axes[0].set_title("Selective prediction: abstain on low calibrated confidence", fontsize=10)
    axes[0].set_ylim(0.6, 1.01)
    axes[0].set_xlim(0.1, 1.0)
    axes[1].plot([0, 1], [0, 1], color=PAL["axis"], lw=0.8)
    axes[1].set_xlabel("predicted confidence")
    axes[1].set_ylabel("observed accuracy")
    axes[1].set_title(f"Reliability before/after temperature scaling (T={out['temperature_mean']:.2f})", fontsize=10)
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=PAL["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor(PAL["surface"])
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig_abstention_{args.tag}.png", dpi=200)
    (RESULTS / f"calibration_{args.tag}.json").write_text(json.dumps(out, indent=2))
    for s, r in out["pooled"].items():
        print(f"{s}: acc {r['accuracy']:.3f} | ECE raw {r['ece_raw']:.3f} -> cal {r['ece_cal']:.3f} | "
              f"acc@cov90 {r['acc_at_cov90']:.3f} (thr {r['thr_at_cov90']:.2f}) acc@cov80 {r['acc_at_cov80']:.3f} "
              f"acc@cov70 {r['acc_at_cov70']:.3f} | thr0.7: cov {r['cov_at_thr70']:.2f} acc {r['acc_at_thr70']:.3f} | AURC {r['aurc']:.3f}")
    print("T:", [round(r["temperature"], 2) for r in out["per_run"]])


if __name__ == "__main__":
    main()
