"""Tables and figures for the extension experiments (beyond the proposal grid).

Reads results/runs/*/metrics.json + preds, results/clipzs_*.json, results/calibration_*.json, results/bootstrap.csv.
Writes results/extensions_summary.md, results/extensions.json and figures fig_backbones.png, fig_rule_items.png.
"""
from __future__ import annotations

import glob
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

PAL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", muted="#c3c2b7", ink="#0b0b0b", ink2="#52514e",
           grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "savefig.dpi": 200,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False})
BACKBONE_LABEL = {"deit_tiny": "DeiT-Tiny (5.7M)", "deit_small": "DeiT-Small (22M)", "dinov2_small": "DINOv2-S (22M)",
                  "clip_b32": "CLIP ViT-B/32 (88M)"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=PAL["ink2"], labelsize=8.5, length=0)
    ax.yaxis.grid(True, color=PAL["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(PAL["surface"])


def load_runs() -> pd.DataFrame:
    rows = []
    for d in sorted(RUNS.glob("*")):
        m = d / "metrics.json"
        if not m.exists() or d.name.startswith(("sweep", "smoke", "clipzs")):
            continue
        r = json.loads(m.read_text())
        c = r["config"]
        rows.append({"name": r["name"], "model": c["model"], "aug": c["aug"], "frac": c["frac"], "seed": c["seed"],
                     "backbone": c.get("backbone", "deit_tiny"), "target": c.get("target", "bin"),
                     "extra": "+".join(c.get("extra_splits", [])) or "none",
                     "val_f1": r["val"]["macro_f1"], "test_acc": r["test"]["accuracy"], "test_f1": r["test"]["macro_f1"],
                     "ood_acc": r["ood_test"]["accuracy"], "ood_f1": r["ood_test"]["macro_f1"],
                     "test_fine_acc": r["test"].get("fine_accuracy", np.nan), "ood_fine_acc": r["ood_test"].get("fine_accuracy", np.nan),
                     "n_params": r["n_trainable_params"], "n_train": r["n_train"], "train_time": r["train_time_s"],
                     "cpu_ms": r["inference"]["cpu_single_ms_per_img"], "gpu_ms": r["inference"].get("gpu_batch64_ms_per_img", np.nan)})
    return pd.DataFrame(rows)


def agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = df.groupby(keys)
    cols = ["test_acc", "test_f1", "ood_acc", "ood_f1", "test_fine_acc", "ood_fine_acc", "train_time", "cpu_ms"]
    s = g[cols].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["n_seeds"] = g.size()
    s["n_params"] = g["n_params"].first()
    s["n_train"] = g["n_train"].first()
    return s.reset_index()


def pm(value, sd):
    return f"{value:.3f} ± {sd:.3f}" if not np.isnan(sd) else f"{value:.3f}"


def rule_item_accuracy(run_prefix: str, seeds) -> dict:
    """Accuracy on campus-rule items (plastic film + glass), rigid plastic, and everything else, per test set."""
    out = {}
    for split, fname in [("test", "preds_test.csv"), ("ood", "preds_ood_test.csv")]:
        parts = []
        for s in seeds:
            p = RUNS / (f"{run_prefix}_s{s}" if s is not None else run_prefix) / fname
            if p.exists():
                parts.append(pd.read_csv(p))
        if not parts:
            continue
        d = pd.concat(parts)
        oc = d["orig_class"].str.lower()
        film = (oc == "plastic") & (d["true"] == "landfill")
        glass = oc.str.contains("glass")
        rigid = (oc == "plastic") & (d["true"] == "recycling")
        rule = film | glass
        ok = d["true"] == d["pred"]
        out[split] = {"film": float(ok[film].mean()) if film.any() else np.nan, "n_film": int(film.sum() / len(parts)),
                      "glass": float(ok[glass].mean()), "n_glass": int(glass.sum() / len(parts)),
                      "rigid": float(ok[rigid].mean()), "n_rigid": int(rigid.sum() / len(parts)),
                      "rule_items": float(ok[rule].mean()), "n_rule": int(rule.sum() / len(parts)),
                      "other": float(ok[~rule].mean()), "n_other": int((~rule).sum() / len(parts)),
                      "all": float(ok.mean())}
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    df = load_runs()
    md, out = [], {}

    # ---- 1. backbones (aug, 100%, 3-way head, no extra data)
    bb = df[(df.aug == "brightrot") & (df.frac == 1.0) & (df.target == "bin") & (df.extra == "none")]
    bbs = agg(bb, ["backbone", "model"])
    order = ["deit_tiny", "deit_small", "dinov2_small", "clip_b32"]
    bbs["order"] = bbs.backbone.map({b: i for i, b in enumerate(order)})
    bbs = bbs.sort_values(["order", "model"], ascending=[True, False])
    md += ["## Backbone study (last-block FT or linear probe, brightness-rotation augmentation, 100% data, 3 seeds)", "",
           "| Backbone | Adaptation | Trainable params | RealWaste acc | RealWaste F1 | Proxy acc | Proxy F1 | CPU ms/img |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in bbs.itertuples():
        md.append(f"| {BACKBONE_LABEL[r.backbone]} | {'last-block FT' if r.model == 'lastblock' else 'linear probe'} | {r.n_params:,} | "
                  f"{pm(r.test_acc_mean, r.test_acc_std)} | {pm(r.test_f1_mean, r.test_f1_std)} | {pm(r.ood_acc_mean, r.ood_acc_std)} | "
                  f"{pm(r.ood_f1_mean, r.ood_f1_std)} | {r.cpu_ms_mean:.0f} |")
    out["backbones"] = bbs.drop(columns="order").to_dict(orient="records")
    if len(bbs):
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = [f"{BACKBONE_LABEL[r.backbone].split(' (')[0]}\n{'last-block FT' if r.model == 'lastblock' else 'linear probe'}" for r in bbs.itertuples()]
        x = np.arange(len(bbs))
        w = 0.36
        for k, (key, color, lab) in enumerate([("test", PAL["blue"], "RealWaste test"), ("ood", PAL["orange"], "Proxy new-photo test")]):
            ax.bar(x + (k - 0.5) * (w + 0.02), bbs[f"{key}_f1_mean"], w, color=color, label=lab, yerr=bbs[f"{key}_f1_std"],
                   error_kw=dict(ecolor=PAL["ink2"], lw=0.8, capsize=2))
            for xi, v in zip(x + (k - 0.5) * (w + 0.02), bbs[f"{key}_f1_mean"]):
                ax.text(xi, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("macro-F1 (mean of 3 seeds, bar = ±1 sd)")
        ax.set_title("Backbone comparison (augmentation, 100% data)")
        style(ax)
        ax.legend(loc="upper left", ncol=2, fontsize=8.5)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_backbones.png")
        plt.close(fig)

    # ---- 2. two-stage vs direct, CLIP zero-shot
    md += ["", "## Item classifier + rule table (two-stage) and CLIP zero-shot", "",
           "| Model | Training | RealWaste acc | RealWaste F1 | Proxy acc | Proxy F1 | Item accuracy (RW / proxy) |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    direct = agg(df[(df.name.str.startswith("lastblock_brightrot_f1.0_s"))], ["model"])
    if len(direct):
        r = direct.iloc[0]
        md.append(f"| DeiT-Tiny, direct 3-way head (main model) | 461 reviewed images | {pm(r.test_acc_mean, r.test_acc_std)} | {pm(r.test_f1_mean, r.test_f1_std)} | "
                  f"{pm(r.ood_acc_mean, r.ood_acc_std)} | {pm(r.ood_f1_mean, r.ood_f1_std)} | - |")
    two = agg(df[(df.target == "fine")], ["model"])
    if len(two):
        r = two.iloc[0]
        md.append(f"| DeiT-Tiny, 10-way item head + rule table | 461 reviewed images | {pm(r.test_acc_mean, r.test_acc_std)} | {pm(r.test_f1_mean, r.test_f1_std)} | "
                  f"{pm(r.ood_acc_mean, r.ood_acc_std)} | {pm(r.ood_f1_mean, r.ood_f1_std)} | {r.test_fine_acc_mean:.3f} / {r.ood_fine_acc_mean:.3f} |")
        out["twostage"] = two.to_dict(orient="records")
    clip = {}
    for p in sorted(glob.glob(str(RESULTS / "clipzs_*.json"))):
        j = json.loads(Path(p).read_text())
        t, o = j["sets"]["test"], j["sets"]["ood_test"]
        md.append(f"| CLIP {j['model']} zero-shot, item prompts + rule table | none | {t['items']['accuracy']:.3f} | {t['items']['macro_f1']:.3f} | "
                  f"{o['items']['accuracy']:.3f} | {o['items']['macro_f1']:.3f} | {t['items']['fine_accuracy']:.3f} / {o['items']['fine_accuracy']:.3f} |")
        md.append(f"| CLIP {j['model']} zero-shot, direct bin prompts | none | {t['bins']['accuracy']:.3f} | {t['bins']['macro_f1']:.3f} | "
                  f"{o['bins']['accuracy']:.3f} | {o['bins']['macro_f1']:.3f} | - |")
        clip[j["model"]] = j["sets"]
    out["clip_zeroshot"] = clip

    # rule-item accuracy comparison
    systems = [("Direct 3-way (main model)", "lastblock_brightrot_f1.0", [0, 1, 2]),
               ("Two-stage item + rule", "fine_lastblock_brightrot_f1.0", [0, 1, 2]),
               ("CLIP ViT-B/32 zero-shot items", "clipzs_vitb32_openai_items", [None]),
               ("CLIP ViT-L/14 zero-shot items", "clipzs_vitl14_openai_items", [None])]
    ri = {}
    for label, prefix, seeds in systems:
        r = rule_item_accuracy(prefix, seeds)
        if r:
            ri[label] = r
    out["rule_items"] = ri
    if ri:
        md += ["", "Accuracy on campus-rule items (plastic film + glass, bin = landfill by rule), rigid plastic and all other items:", "",
               "| System | RW film | RW glass | RW rigid plastic | RW other | Proxy film | Proxy glass | Proxy rigid plastic | Proxy other |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for label, r in ri.items():
            t, o = r.get("test", {}), r.get("ood", {})
            md.append(f"| {label} | {t.get('film', np.nan):.2f} (n={t.get('n_film', 0)}) | {t.get('glass', np.nan):.2f} (n={t.get('n_glass', 0)}) | "
                      f"{t.get('rigid', np.nan):.2f} (n={t.get('n_rigid', 0)}) | {t.get('other', np.nan):.2f} | {o.get('film', np.nan):.2f} (n={o.get('n_film', 0)}) | "
                      f"{o.get('glass', np.nan):.2f} (n={o.get('n_glass', 0)}) | {o.get('rigid', np.nan):.2f} (n={o.get('n_rigid', 0)}) | {o.get('other', np.nan):.2f} |")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        cats = [("film", "plastic film"), ("glass", "glass"), ("rigid", "rigid plastic"), ("other", "all other items")]
        colors = [PAL["muted"], PAL["blue"], PAL["orange"], PAL["aqua"]]
        for ax, (split, title) in zip(axes, [("test", "RealWaste test"), ("ood", "Proxy new-photo test")]):
            x = np.arange(len(cats))
            n = len(ri)
            w = 0.8 / n
            for k, (label, r) in enumerate(ri.items()):
                vals = [r.get(split, {}).get(c, np.nan) for c, _ in cats]
                ax.bar(x + (k - (n - 1) / 2) * w, vals, w * 0.92, color=colors[k % len(colors)], label=label)
                for xi, v in zip(x + (k - (n - 1) / 2) * w, vals):
                    if not np.isnan(v):
                        ax.text(xi, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=6.5, color=PAL["ink"])
            ax.set_xticks(x)
            ax.set_xticklabels([t for _, t in cats], fontsize=8.5)
            ax.set_ylim(0, 1.08)
            ax.set_title(title, fontsize=10)
            style(ax)
        axes[0].set_ylabel("accuracy")
        axes[1].legend(fontsize=7.5, loc="lower right")
        fig.suptitle("Campus-rule items: direct model vs item classifier + rule table vs CLIP zero-shot", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_rule_items.png")
        plt.close(fig)

    # ---- 3. augmentation variants
    augs = agg(df[(df.model == "lastblock") & (df.frac == 1.0) & (df.target == "bin") & (df.backbone == "deit_tiny") & (df.extra == "none")], ["aug"])
    if len(augs):
        md += ["", "## Augmentation variants (DeiT-Tiny last-block FT, 100% data, 3 seeds)", "",
               "| Augmentation | RealWaste acc | RealWaste F1 | Proxy acc | Proxy F1 | Train time (s) |", "| --- | --- | --- | --- | --- | --- |"]
        names = {"none": "none", "brightrot": "brightness + rotation (proposal)", "strong": "strong: crop, flip, colour, rotation",
                 "bgswap": "strong + background swap (GrabCut mask, synthetic backgrounds, p=0.5)"}
        for a in ["none", "brightrot", "strong", "bgswap"]:
            s = augs[augs.aug == a]
            if len(s):
                r = s.iloc[0]
                md.append(f"| {names[a]} | {pm(r.test_acc_mean, r.test_acc_std)} | {pm(r.test_f1_mean, r.test_f1_std)} | "
                          f"{pm(r.ood_acc_mean, r.ood_acc_std)} | {pm(r.ood_f1_mean, r.ood_f1_std)} | {r.train_time_mean:.0f} |")
        out["augs"] = augs.to_dict(orient="records")

    # ---- 4. targeted annotation
    tg = df[(df.model == "lastblock") & (df.aug == "brightrot") & (df.frac == 1.0) & (df.target == "bin") & (df.backbone == "deit_tiny")]
    tgs = agg(tg, ["extra"])
    if (tgs.extra != "none").any():
        md += ["", "## Targeted vs random extra annotation (DeiT-Tiny last-block FT + aug, 3 seeds)", "",
               "| Training set | Images | RealWaste acc | RealWaste F1 | Proxy acc | Proxy F1 | RW film acc | Proxy film acc |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for e in ["none", "extra_random", "extra_film"]:
            s = tgs[tgs.extra == e]
            if not len(s):
                continue
            r = s.iloc[0]
            prefix = "lastblock_brightrot_f1.0" if e == "none" else f"lastblock_brightrot_{e}"
            ri_e = rule_item_accuracy(prefix, [0, 1, 2])
            label = {"none": "baseline (461)", "extra_random": "+ random extra images", "extra_film": "+ targeted plastic-film images"}[e]
            md.append(f"| {label} | {r.n_train} | {pm(r.test_acc_mean, r.test_acc_std)} | {pm(r.test_f1_mean, r.test_f1_std)} | "
                      f"{pm(r.ood_acc_mean, r.ood_acc_std)} | {pm(r.ood_f1_mean, r.ood_f1_std)} | "
                      f"{ri_e.get('test', {}).get('film', np.nan):.2f} | {ri_e.get('ood', {}).get('film', np.nan):.2f} |")
        out["targeted"] = tgs.to_dict(orient="records")

    # ---- 5. calibration + bootstrap summaries
    for p in sorted(glob.glob(str(RESULTS / "calibration_*.json"))):
        j = json.loads(Path(p).read_text())
        tag = Path(p).stem.replace("calibration_", "")
        md += ["", f"## Calibration and abstention ({tag}; T fitted on validation, mean T = {j['temperature_mean']:.2f})", "",
               "| Test set | Accuracy | ECE raw → calibrated | Acc at 90% coverage | Acc at 80% coverage | Acc at 70% coverage | Coverage / acc at conf ≥ 0.7 |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        for s, lab in [("test", "RealWaste test"), ("ood_test", "Proxy new-photo test")]:
            r = j["pooled"][s]
            md.append(f"| {lab} | {r['accuracy']:.3f} | {r['ece_raw']:.3f} → {r['ece_cal']:.3f} | {r['acc_at_cov90']:.3f} | {r['acc_at_cov80']:.3f} | "
                      f"{r['acc_at_cov70']:.3f} | {r['cov_at_thr70']:.2f} / {r['acc_at_thr70']:.3f} |")
        out[f"calibration_{tag}"] = j["pooled"]
    bp = RESULTS / "bootstrap.csv"
    if bp.exists():
        b = pd.read_csv(bp)
        md += ["", "## Bootstrap 95% confidence intervals (2,000 resamples of the test images, seed-averaged)", "",
               "| Test set | Comparison | Δ accuracy [95% CI] | Δ macro-F1 [95% CI] | P(ΔF1 ≤ 0) |", "| --- | --- | --- | --- | --- |"]
        for r in b[b.kind == "diff"].itertuples():
            md.append(f"| {'RealWaste' if r.split == 'test' else 'Proxy'} | {r.name} | {r.acc:+.3f} [{r.acc_lo:+.3f}, {r.acc_hi:+.3f}] | "
                      f"{r.f1:+.3f} [{r.f1_lo:+.3f}, {r.f1_hi:+.3f}] | {r.p_f1_le_0:.3f} |")
        md += ["", "| Test set | Configuration | Accuracy [95% CI] | Macro-F1 [95% CI] |", "| --- | --- | --- | --- |"]
        for r in b[b.kind == "config"].itertuples():
            md.append(f"| {'RealWaste' if r.split == 'test' else 'Proxy'} | {r.name} | {r.acc:.3f} [{r.acc_lo:.3f}, {r.acc_hi:.3f}] | {r.f1:.3f} [{r.f1_lo:.3f}, {r.f1_hi:.3f}] |")

    (RESULTS / "extensions_summary.md").write_text("\n".join(md), encoding="utf-8")
    (RESULTS / "extensions.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
