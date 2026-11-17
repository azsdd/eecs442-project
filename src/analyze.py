"""Aggregate the experiment grid: summary tables, factor effects and figures.

Reads results/runs/<model>_<aug>_f<frac>_s<seed>/metrics.json (+ preds_*.csv) and writes
  results/summary.csv, results/summary.md, results/summary.json, results/figures/*.png
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
from PIL import Image

from common import BINS, FIGURES, OOD_IMAGES, RAW_RW, RESULTS, ROOT, RUNS, REVIEWED, read_csv

# Reference palette (dataviz skill): categorical slots + chrome
PAL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", ink="#0b0b0b", ink2="#52514e",
           muted="#898781", grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb")
SEQ = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
                     "figure.facecolor": "white", "savefig.dpi": 200, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.labelsize": 9.5, "legend.fontsize": 8.5, "legend.frameon": False})
SETS = {"test": "RealWaste test (n=100)", "ood": "Proxy new-photo test (n=142)"}
SET_COLOR = {"test": PAL["blue"], "ood": PAL["orange"]}


def style(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PAL["axis"])
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=PAL["ink2"], labelsize=8.5, length=0)
    if ygrid:
        ax.yaxis.grid(True, color=PAL["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(PAL["surface"])


def cfg_label(model, aug, frac):
    return f"{'last-block FT' if model == 'lastblock' else 'linear probe'}\n{'aug' if aug == 'brightrot' else 'no aug'}, {int(frac * 100)}%"


def load_runs() -> pd.DataFrame:
    rows = []
    for d in sorted(RUNS.glob("*")):
        m = d / "metrics.json"
        if not m.exists() or d.name.startswith(("sweep", "smoke")):
            continue
        r = json.loads(m.read_text())
        c = r["config"]
        rows.append({
            "name": r["name"], "model": c["model"], "aug": c["aug"], "frac": c["frac"], "seed": c["seed"],
            "val_f1": r["val"]["macro_f1"], "test_acc": r["test"]["accuracy"], "test_f1": r["test"]["macro_f1"],
            "ood_acc": r["ood_test"]["accuracy"], "ood_f1": r["ood_test"]["macro_f1"],
            "train_time": r["train_time_s"], "best_epoch": r["best_epoch"], "n_train": r["n_train"],
            "n_params": r["n_trainable_params"], "gpu_ms": r["inference"].get("gpu_batch64_ms_per_img", np.nan),
            "cpu_ms": r["inference"]["cpu_single_ms_per_img"],
            **{f"test_rec_{b}": r["test"]["per_class_recall"][b] for b in BINS},
            **{f"ood_rec_{b}": r["ood_test"]["per_class_recall"][b] for b in BINS},
            "test_cm": r["test"]["confusion_matrix"], "ood_cm": r["ood_test"]["confusion_matrix"],
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "aug", "frac"])
    cols = ["val_f1", "test_acc", "test_f1", "ood_acc", "ood_f1", "train_time", "best_epoch", "gpu_ms", "cpu_ms"] + \
           [f"test_rec_{b}" for b in BINS] + [f"ood_rec_{b}" for b in BINS]
    s = g[cols].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["n_seeds"] = g.size()
    s["n_train"] = g["n_train"].first()
    s["n_params"] = g["n_params"].first()
    return s.reset_index()


def factor_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Paired differences for each factor, averaged over the other factors and seeds."""
    out = []
    specs = [("model", "lastblock", "linear", "last-block FT vs linear probe"),
             ("aug", "brightrot", "none", "augmentation vs none"),
             ("frac", 1.0, 0.25, "100% vs 25% training data")]
    for col, hi, lo, label in specs:
        others = [c for c in ["model", "aug", "frac"] if c != col] + ["seed"]
        a = df[df[col] == hi].set_index(others)
        b = df[df[col] == lo].set_index(others)
        idx = a.index.intersection(b.index)
        d = a.loc[idx, ["test_f1", "ood_f1", "test_acc", "ood_acc"]] - b.loc[idx, ["test_f1", "ood_f1", "test_acc", "ood_acc"]]
        out.append({"factor": label, "n_pairs": len(d),
                    "d_test_f1_mean": d["test_f1"].mean(), "d_test_f1_std": d["test_f1"].std(),
                    "d_ood_f1_mean": d["ood_f1"].mean(), "d_ood_f1_std": d["ood_f1"].std(),
                    "d_test_acc_mean": d["test_acc"].mean(), "d_ood_acc_mean": d["ood_acc"].mean(),
                    "frac_pairs_test_improved": float((d["test_f1"] > 0).mean()),
                    "frac_pairs_ood_improved": float((d["ood_f1"] > 0).mean())})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- figures
def fig_main(summary: pd.DataFrame):
    order = [(m, a, f) for m in ["linear", "lastblock"] for a in ["none", "brightrot"] for f in [0.25, 1.0]]
    s = summary.set_index(["model", "aug", "frac"]).loc[order].reset_index()
    x = np.arange(len(s))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, (key, color) in enumerate(SET_COLOR.items()):
        ax.bar(x + (k - 0.5) * (w + 0.02), s[f"{key}_f1_mean"], w, color=color, label=SETS[key],
               yerr=s[f"{key}_f1_std"], error_kw=dict(ecolor=PAL["ink2"], lw=0.8, capsize=2))
        for xi, v in zip(x + (k - 0.5) * (w + 0.02), s[f"{key}_f1_mean"]):
            ax.text(xi, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([cfg_label(m, a, f) for m, a, f in order], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("macro-F1 (mean over 3 seeds, bar = ±1 sd)")
    ax.set_title("Macro-F1 on the two test sets for every configuration")
    style(ax)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_main_f1.png")
    plt.close(fig)


def fig_confusion(df: pd.DataFrame, best: tuple):
    sub = df[(df.model == best[0]) & (df.aug == best[1]) & (df.frac == best[2])]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, key in zip(axes, ["test", "ood"]):
        cm = np.sum([np.array(c) for c in sub[f"{key}_cm"]], axis=0)
        norm = cm / cm.sum(axis=1, keepdims=True)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else PAL["ink"])
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(BINS)
        ax.set_yticklabels(BINS)
        ax.set_xlabel("predicted bin")
        ax.set_ylabel("true bin")
        ax.set_title(SETS[key], fontsize=10)
        ax.tick_params(length=0, colors=PAL["ink2"], labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"Confusion matrices, {cfg_label(*best).replace(chr(10), ' ')} (3 seeds pooled)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_confusion.png")
    plt.close(fig)


def fig_recall(summary: pd.DataFrame, best: tuple):
    s = summary.set_index(["model", "aug", "frac"]).loc[[best]].iloc[0]
    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, (key, color) in enumerate(SET_COLOR.items()):
        vals = [s[f"{key}_rec_{b}_mean"] for b in BINS]
        errs = [s[f"{key}_rec_{b}_std"] for b in BINS]
        ax.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=color, label=SETS[key], yerr=errs,
               error_kw=dict(ecolor=PAL["ink2"], lw=0.8, capsize=2))
        for xi, v in zip(x + (k - 0.5) * (w + 0.02), vals):
            ax.text(xi, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(BINS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("recall (mean over 3 seeds)")
    ax.set_title(f"Per-class recall, {cfg_label(*best).replace(chr(10), ' ')}")
    style(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_per_class_recall.png")
    plt.close(fig)


def fig_frac(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
    for ax, key in zip(axes, ["test", "ood"]):
        for model, color in [("linear", PAL["blue"]), ("lastblock", PAL["orange"])]:
            for aug, ls, mk in [("brightrot", "-", "o"), ("none", ":", "s")]:
                s = summary[(summary.model == model) & (summary.aug == aug)].sort_values("frac")
                ax.errorbar([25, 100], s[f"{key}_f1_mean"], yerr=s[f"{key}_f1_std"], color=color, ls=ls, marker=mk,
                            ms=5, lw=1.6, capsize=2, elinewidth=0.8,
                            label=f"{'last-block FT' if model == 'lastblock' else 'linear probe'}, {'aug' if aug == 'brightrot' else 'no aug'}")
        ax.set_xticks([25, 100])
        ax.set_xticklabels(["25% (115 imgs)", "100% (461 imgs)"])
        ax.set_xlim(10, 115)
        ax.set_title(SETS[key], fontsize=10)
        ax.set_xlabel("training data used")
        style(ax)
    axes[0].set_ylabel("macro-F1")
    axes[0].legend(loc="lower right")
    fig.suptitle("Effect of the amount of annotated training data", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_data_fraction.png")
    plt.close(fig)


def fig_curves():
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for model, color in [("linear", PAL["blue"]), ("lastblock", PAL["orange"])]:
        for aug, ls in [("brightrot", "-"), ("none", ":")]:
            p = RUNS / f"{model}_{aug}_f1.0_s0" / "history.json"
            if not p.exists():
                continue
            h = json.loads(p.read_text())
            ax.plot([e["epoch"] for e in h], [e["val_macro_f1"] for e in h], color=color, ls=ls, lw=1.6,
                    label=f"{'last-block FT' if model == 'lastblock' else 'linear probe'}, {'aug' if aug == 'brightrot' else 'no aug'}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation macro-F1")
    ax.set_title("Validation macro-F1 during training (100% data, seed 0)")
    style(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_training_curves.png")
    plt.close(fig)


def per_material(df: pd.DataFrame, best: tuple) -> pd.DataFrame:
    rows = []
    for key, fname in [("test", "preds_test.csv"), ("ood", "preds_ood_test.csv")]:
        parts = []
        for seed in sorted(df.seed.unique()):
            p = RUNS / f"{best[0]}_{best[1]}_f{best[2]}_s{seed}" / fname
            if p.exists():
                parts.append(pd.read_csv(p))
        if not parts:
            continue
        d = pd.concat(parts)
        d["correct"] = d["true"] == d["pred"]
        g = d.groupby(["orig_class", "true"])["correct"].agg(["mean", "size"]).reset_index()
        g["set"] = key
        g["n_images"] = g["size"] // len(parts)
        rows.append(g)
    return pd.concat(rows)


def fig_per_material(pm: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1, 1.1]))
    for ax, key in zip(axes, ["test", "ood"]):
        d = pm[pm.set == key].sort_values(["true", "mean"], ascending=[True, False])
        labels = [f"{r.orig_class} ({r.true[:4]}, n={r.n_images})" for r in d.itertuples()]
        y = np.arange(len(d))
        ax.barh(y, d["mean"], color=SET_COLOR[key], height=0.62)
        for yi, v in zip(y, d["mean"]):
            ax.text(min(v, 0.97) + 0.01 if v < 0.9 else v - 0.02, yi, f"{v:.0%}", va="center", ha="left" if v < 0.9 else "right",
                    fontsize=8, color=PAL["ink"] if v < 0.9 else "white")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("accuracy (3 seeds pooled)")
        ax.set_title(SETS[key], fontsize=10)
        style(ax, ygrid=False)
        ax.xaxis.grid(True, color=PAL["grid"], linewidth=0.8)
    fig.suptitle("Accuracy by original material / source class (best configuration)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_per_material.png")
    plt.close(fig)


def fig_errors(best: tuple, seed: int = 0, max_n: int = 20):
    splits = {r["cid"]: r for r in read_csv(REVIEWED / "splits.csv")}
    for key, fname in [("test", "preds_test.csv"), ("ood", "preds_ood_test.csv")]:
        p = RUNS / f"{best[0]}_{best[1]}_f{best[2]}_s{seed}" / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        err = d[d["true"] != d["pred"]].copy()
        err["conf"] = err.apply(lambda r: r[f"p_{r['pred']}"], axis=1)
        err = err.sort_values("conf", ascending=False).head(max_n)
        n = len(err)
        if n == 0:
            continue
        cols = 5
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")
        for ax, r in zip(axes.flat, err.itertuples()):
            im = Image.open(ROOT / splits[r.cid]["file"]).convert("RGB")
            ax.imshow(im)
            ax.set_title(f"{r.orig_class}\ntrue {r.true} → pred {r.pred} ({r.conf:.2f})", fontsize=8, color=PAL["ink"])
        fig.suptitle(f"Highest-confidence mistakes on the {SETS[key]} ({cfg_label(*best).replace(chr(10), ' ')}, seed {seed})",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIGURES / f"fig_errors_{key}.png")
        plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    df = load_runs()
    assert len(df), "no finished runs"
    summary = summarize(df)
    summary["label"] = [cfg_label(m, a, f).replace("\n", ", ") for m, a, f in zip(summary.model, summary.aug, summary.frac)]
    summary.to_csv(RESULTS / "summary.csv", index=False)
    effects = factor_effects(df)
    effects.to_csv(RESULTS / "factor_effects.csv", index=False)

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="lastblock,brightrot,1.0",
                    help="configuration used for the detailed analysis and the demo (model,aug,frac)")
    args = ap.parse_args()
    full = summary[summary.frac == 1.0]
    val_row = full.sort_values("val_f1_mean", ascending=False).iloc[0]
    m, a, f = args.main.split(",")
    best = (m, a, float(f))
    best_row = summary.set_index(["model", "aug", "frac"]).loc[[best]].reset_index().iloc[0]
    pm = per_material(df, best)
    pm.to_csv(RESULTS / "per_material.csv", index=False)

    fig_main(summary)
    fig_confusion(df, best)
    fig_recall(summary, best)
    fig_frac(summary)
    fig_curves()
    fig_per_material(pm)
    fig_errors(best)

    md = ["| configuration | n_train | trainable params | val F1 | test acc | test F1 | proxy acc | proxy F1 | train time (s) | best epoch |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in summary.sort_values(["model", "aug", "frac"]).itertuples():
        md.append(f"| {r.label} | {r.n_train} | {r.n_params:,} | {r.val_f1_mean:.3f} | {r.test_acc_mean:.3f} ± {r.test_acc_std:.3f} | "
                  f"{r.test_f1_mean:.3f} ± {r.test_f1_std:.3f} | {r.ood_acc_mean:.3f} ± {r.ood_acc_std:.3f} | "
                  f"{r.ood_f1_mean:.3f} ± {r.ood_f1_std:.3f} | {r.train_time_mean:.1f} | {r.best_epoch_mean:.0f} |")
    md += ["", "| factor | Δ test F1 | Δ proxy F1 | pairs improved (test / proxy) |", "| --- | --- | --- | --- |"]
    for r in effects.itertuples():
        md.append(f"| {r.factor} | {r.d_test_f1_mean:+.3f} ± {r.d_test_f1_std:.3f} | {r.d_ood_f1_mean:+.3f} ± {r.d_ood_f1_std:.3f} | "
                  f"{r.frac_pairs_test_improved:.0%} / {r.frac_pairs_ood_improved:.0%} (n={r.n_pairs}) |")
    (RESULTS / "summary.md").write_text("\n".join(md), encoding="utf-8")
    key = {"best_config": {"model": best[0], "aug": best[1], "frac": best[2], "selection": "post-hoc: best mean macro-F1 on both test sets"},
           "val_selected_config": {"model": val_row.model, "aug": val_row.aug, "frac": float(val_row.frac),
                                   "val_f1_mean": float(val_row.val_f1_mean), "test_f1_mean": float(val_row.test_f1_mean),
                                   "ood_f1_mean": float(val_row.ood_f1_mean)},
           "best_row": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in best_row.items()},
           "n_runs": int(len(df)), "effects": effects.to_dict(orient="records"),
           "per_material": pm.to_dict(orient="records")}
    (RESULTS / "summary.json").write_text(json.dumps(key, indent=2, default=float))
    print("\n".join(md))
    print(f"\nmain config (detailed analysis + demo): {best}")
    print(f"validation-selected config (100% data): {(val_row.model, val_row.aug)} val F1 {val_row.val_f1_mean:.3f} "
          f"-> test F1 {val_row.test_f1_mean:.3f}, proxy F1 {val_row.ood_f1_mean:.3f}")


if __name__ == "__main__":
    main()
