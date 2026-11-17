"""Train and evaluate one configuration of the waste-bin classifier.

Main grid (proposal):
  python -I src/train.py --model linear --aug none --frac 1.0 --seed 0
  python -I src/train.py --model lastblock --aug brightrot --frac 0.25 --seed 1
Extensions:
  --backbone {deit_tiny,deit_small,dinov2_small,clip_b32}   pretrained ViT backbone (default deit_tiny)
  --target {bin,fine}      bin = 3-way head (default); fine = 10-way item head + campus rule table
  --aug {none,brightrot,strong,bgswap}   strong = crop/flip/colour/rotation; bgswap = strong + background swap
  --extra_splits NAME ...  extra splits.csv splits added to training (targeted-annotation experiment)

Writes results/runs/<name>/{metrics.json, history.json, preds_*.csv, trainable_state.pt}
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import InterpolationMode, v2

from common import (BACKBONES, BINS, BIN_TO_IDX, DATA, FINE_CLASSES, FINE_TO_BIN, FINE_TO_IDX, IMG_SIZE,
                    REVIEWED, ROOT, RUNS, fine_class, read_csv, seed_everything, write_csv)

PRELOAD_SIZE = 256
MASKS = DATA / "masks"


# ----------------------------------------------------------------------------- data
def pil_to_tensor256(im: Image.Image) -> torch.Tensor:
    """Resize shortest side to 256 then center-crop 256x256; return uint8 CHW tensor."""
    im = im.convert("RGB")
    w, h = im.size
    s = PRELOAD_SIZE / min(w, h)
    im = im.resize((max(PRELOAD_SIZE, round(w * s)), max(PRELOAD_SIZE, round(h * s))), Image.BICUBIC)
    left, top = (im.width - PRELOAD_SIZE) // 2, (im.height - PRELOAD_SIZE) // 2
    im = im.crop((left, top, left + PRELOAD_SIZE, top + PRELOAD_SIZE))
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).contiguous()


def load_image_256(path: Path) -> torch.Tensor:
    return pil_to_tensor256(Image.open(path).convert("RGB"))


def labels_of(rows: list[dict], target: str) -> torch.Tensor:
    if target == "fine":
        return torch.tensor([FINE_TO_IDX[fine_class(r)] for r in rows], dtype=torch.long)
    return torch.tensor([BIN_TO_IDX[r["bin"]] for r in rows], dtype=torch.long)


def preload(rows: list[dict], target: str = "bin") -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.empty((len(rows), 3, PRELOAD_SIZE, PRELOAD_SIZE), dtype=torch.uint8)
    for i, r in enumerate(rows):
        x[i] = load_image_256(ROOT / r["file"])
    return x, labels_of(rows, target)


def load_masks(rows: list[dict]) -> torch.Tensor:
    """Foreground masks (uint8 0/255, 1x256x256) from data/masks/<cid>.png; a missing mask = all foreground."""
    m = torch.full((len(rows), 1, PRELOAD_SIZE, PRELOAD_SIZE), 255, dtype=torch.uint8)
    for i, r in enumerate(rows):
        p = MASKS / f"{r['cid']}.png"
        if p.exists():
            m[i, 0] = torch.from_numpy(np.asarray(Image.open(p).convert("L")).copy())
    return m


def random_background() -> torch.Tensor:
    """A synthetic 3x256x256 uint8 background: flat colour, gradient, noisy colour or stripes."""
    kind = random.choice(["flat", "gradient", "noise", "stripes"])
    c1 = torch.randint(0, 256, (3, 1, 1)).float()
    c2 = torch.randint(0, 256, (3, 1, 1)).float()
    yy, xx = torch.meshgrid(torch.linspace(0, 1, PRELOAD_SIZE), torch.linspace(0, 1, PRELOAD_SIZE), indexing="ij")
    if kind == "flat":
        bg = c1.expand(3, PRELOAD_SIZE, PRELOAD_SIZE).clone()
    elif kind == "gradient":
        a = random.random() * 2 * np.pi
        t = (xx * np.cos(a) + yy * np.sin(a) + 1) / 2
        bg = c1 * (1 - t) + c2 * t
    elif kind == "noise":
        bg = c1 + torch.randn(3, PRELOAD_SIZE, PRELOAD_SIZE) * random.uniform(5, 30)
    else:
        a = random.random() * np.pi
        f = random.uniform(4, 30)
        t = (torch.sin((xx * np.cos(a) + yy * np.sin(a)) * f * 2 * np.pi) + 1) / 2
        bg = c1 * (1 - t) + c2 * t
    return bg.clamp(0, 255).round().to(torch.uint8)


class TensorDS(torch.utils.data.Dataset):
    def __init__(self, x, y, tf, masks=None, p_swap=0.5):
        self.x, self.y, self.tf, self.masks, self.p_swap = x, y, tf, masks, p_swap

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        if self.masks is not None and random.random() < self.p_swap:
            m = self.masks[i].float() / 255.0
            x = (x.float() * m + random_background().float() * (1 - m)).round().to(torch.uint8)
        return self.tf(x), self.y[i]


def build_transforms(aug: str, rot_deg: float, brightness: float, mean, std):
    tail = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    eval_tf = v2.Compose([v2.CenterCrop(IMG_SIZE), *tail])
    if aug == "none":
        train_tf = eval_tf
    elif aug == "brightrot":
        train_tf = v2.Compose([v2.ColorJitter(brightness=brightness),
                               v2.RandomRotation(rot_deg, interpolation=InterpolationMode.BILINEAR),
                               v2.CenterCrop(IMG_SIZE), *tail])
    elif aug in ("strong", "bgswap"):
        train_tf = v2.Compose([v2.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), antialias=True),
                               v2.RandomHorizontalFlip(),
                               v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
                               v2.RandomRotation(rot_deg, interpolation=InterpolationMode.BILINEAR), *tail])
    else:
        raise ValueError(aug)
    return train_tf, eval_tf


def subsample(rows: list[dict], frac: float, seed: int) -> list[dict]:
    if frac >= 1.0:
        return rows
    rng = random.Random(1000 + seed)
    out = []
    for b in BINS:
        rb = [r for r in rows if r["bin"] == b]
        k = max(1, round(frac * len(rb)))
        out += rng.sample(rb, k)
    return sorted(out, key=lambda r: r["cid"])


# ----------------------------------------------------------------------------- model
def build_model(mode: str, backbone: str = "deit_tiny", num_classes: int = len(BINS)) -> nn.Module:
    import timm

    name = BACKBONES[backbone]
    kw = {"img_size": IMG_SIZE} if "dinov2" in name else {}
    m = timm.create_model(name, pretrained=True, num_classes=num_classes, **kw)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.head.parameters():
        p.requires_grad = True
    if mode == "lastblock":
        for p in m.blocks[-1].parameters():
            p.requires_grad = True
        for p in m.norm.parameters():
            p.requires_grad = True
    elif mode != "linear":
        raise ValueError(mode)
    return m


def model_norm(m: nn.Module):
    cfg = getattr(m, "pretrained_cfg", {}) or {}
    return tuple(cfg.get("mean", (0.485, 0.456, 0.406))), tuple(cfg.get("std", (0.229, 0.224, 0.225)))


def trainable_state(m: nn.Module) -> dict:
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if k in names}


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def predict(model: nn.Module, loader, dev: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        logits = model(xb.to(dev, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=-1).cpu())
        ys.append(yb)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


FINE_BIN_IDX = np.array([BIN_TO_IDX[FINE_TO_BIN[c]] for c in FINE_CLASSES])


def to_bin_probs(probs: np.ndarray, target: str) -> np.ndarray:
    """Bin probabilities: identity for the 3-way head, summed over the rule table for the fine head."""
    if target == "bin":
        return probs
    out = np.zeros((len(probs), len(BINS)))
    for k, b in enumerate(FINE_BIN_IDX):
        out[:, b] += probs[:, k]
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                                 recall_score)

    labels = list(range(len(BINS)))
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": dict(zip(BINS, map(float, recall_score(y, p, labels=labels, average=None, zero_division=0)))),
        "per_class_precision": dict(zip(BINS, map(float, precision_score(y, p, labels=labels, average=None, zero_division=0)))),
        "per_class_f1": dict(zip(BINS, map(float, f1_score(y, p, labels=labels, average=None, zero_division=0)))),
        "confusion_matrix": confusion_matrix(y, p, labels=labels).tolist(),
        "n": int(len(y)),
    }


@torch.no_grad()
def time_inference(model: nn.Module, x_eval: torch.Tensor, eval_tf, dev: str) -> dict:
    model.eval()
    out = {}
    xb = torch.stack([eval_tf(x_eval[i]) for i in range(min(64, len(x_eval)))])
    if dev == "cuda":
        xg = xb.to(dev)
        for _ in range(3):
            model(xg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            model(xg)
        torch.cuda.synchronize()
        out["gpu_batch64_ms_per_img"] = (time.perf_counter() - t0) / 10 / len(xg) * 1000
    cpu_model = copy.deepcopy(model).cpu().eval()
    x1 = xb[:1]
    for _ in range(3):
        cpu_model(x1)
    t0 = time.perf_counter()
    for _ in range(10):
        cpu_model(x1)
    out["cpu_single_ms_per_img"] = (time.perf_counter() - t0) / 10 * 1000
    out["cpu_threads"] = torch.get_num_threads()
    return out


# ----------------------------------------------------------------------------- main
def run(args) -> dict:
    seed_everything(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name or f"{args.model}_{args.aug}_f{args.frac}_s{args.seed}"
    out_dir = RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(REVIEWED / "splits.csv")
    splits = {s: [r for r in rows if r["split"] == s] for s in ["train", "val", "test", "ood_test"]}
    train_rows = subsample(splits["train"], args.frac, args.seed)
    for extra in args.extra_splits:
        train_rows = train_rows + [r for r in rows if r["split"] == extra]
    n_classes = len(FINE_CLASSES) if args.target == "fine" else len(BINS)

    model = build_model(args.model, args.backbone, n_classes).to(dev)
    mean, std = model_norm(model)
    t0 = time.perf_counter()
    data = {s: preload(rs, args.target) for s, rs in {**splits, "train": train_rows}.items()}
    load_time = time.perf_counter() - t0
    masks = load_masks(train_rows) if args.aug == "bgswap" else None

    train_tf, eval_tf = build_transforms(args.aug, args.rot_deg, args.brightness, mean, std)
    g = torch.Generator().manual_seed(args.seed)
    workers = args.workers if args.aug != "none" else 0
    train_loader = torch.utils.data.DataLoader(TensorDS(*data["train"], train_tf, masks), batch_size=args.bs,
                                               shuffle=True, generator=g, drop_last=False, num_workers=workers,
                                               persistent_workers=workers > 0)
    eval_loaders = {s: torch.utils.data.DataLoader(TensorDS(*data[s], eval_tf), batch_size=128, shuffle=False)
                    for s in ["val", "test", "ood_test"]}

    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head")]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
    groups = [{"params": head_params, "lr": args.lr_head}]
    if other_params:
        groups.append({"params": other_params, "lr": args.lr_block})
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(s, total_steps) / total_steps)))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def bin_eval(split):
        probs, y = predict(model, eval_loaders[split], dev)
        bp = to_bin_probs(probs, args.target)
        yb = y if args.target == "bin" else FINE_BIN_IDX[y]
        return probs, y, bp, yb

    history, best = [], {"val_macro_f1": -1.0, "epoch": -1, "state": None}
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t_train0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.model == "linear":
            model.eval()
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        _, _, bp, yb_ = bin_eval("val")
        mv = metrics(yb_, bp.argmax(1))
        history.append({"epoch": epoch, "train_loss": tot / n, "val_acc": mv["accuracy"], "val_macro_f1": mv["macro_f1"]})
        if mv["macro_f1"] > best["val_macro_f1"]:
            best = {"val_macro_f1": mv["macro_f1"], "epoch": epoch, "state": trainable_state(model)}
        if args.verbose:
            print(f"ep {epoch:3d} loss {tot / n:.4f} val_acc {mv['accuracy']:.3f} val_f1 {mv['macro_f1']:.3f}")
    train_time = time.perf_counter() - t_train0

    model.load_state_dict(best["state"], strict=False)
    results = {"name": name, "config": vars(args), "device": dev, "n_trainable_params": n_trainable,
               "n_train": len(train_rows), "n_val": len(splits["val"]), "n_test": len(splits["test"]),
               "n_ood_test": len(splits["ood_test"]), "best_epoch": best["epoch"], "epochs": args.epochs,
               "train_time_s": train_time, "train_time_per_epoch_s": train_time / args.epochs,
               "preload_time_s": load_time, "gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "",
               "mean": mean, "std": std}
    for s in ["val", "test", "ood_test"]:
        probs, y, bp, yb = bin_eval(s)
        pred = bp.argmax(1)
        results[s] = metrics(yb, pred)
        if args.target == "fine":
            results[s]["fine_accuracy"] = float((probs.argmax(1) == y).mean())
        rs = splits[s]
        recs = []
        for i, r in enumerate(rs):
            rec = {"cid": r["cid"], "orig_class": r["orig_class"], "true": BINS[int(yb[i])], "pred": BINS[int(pred[i])],
                   **{f"p_{b}": f"{bp[i, k]:.5f}" for k, b in enumerate(BINS)}}
            if args.target == "fine":
                rec["fine_true"] = FINE_CLASSES[int(y[i])]
                rec["fine_pred"] = FINE_CLASSES[int(probs[i].argmax())]
                rec.update({f"pf_{k}": f"{probs[i, k]:.5f}" for k in range(len(FINE_CLASSES))})
            recs.append(rec)
        write_csv(out_dir / f"preds_{s}.csv", recs)
    results["inference"] = time_inference(model, data["test"][0], eval_tf, dev)
    torch.save(best["state"], out_dir / "trainable_state.pt")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    extra = f" fine_acc={results['test'].get('fine_accuracy', float('nan')):.3f}" if args.target == "fine" else ""
    print(f"[{name}] best_ep={best['epoch']} val_f1={best['val_macro_f1']:.3f} "
          f"test_acc={results['test']['accuracy']:.3f} test_f1={results['test']['macro_f1']:.3f} "
          f"ood_acc={results['ood_test']['accuracy']:.3f} ood_f1={results['ood_test']['macro_f1']:.3f}{extra} "
          f"train={train_time:.1f}s")
    return results


def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["linear", "lastblock"], required=True)
    ap.add_argument("--aug", choices=["none", "brightrot", "strong", "bgswap"], default="none")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", choices=list(BACKBONES), default="deit_tiny")
    ap.add_argument("--target", choices=["bin", "fine"], default="bin")
    ap.add_argument("--extra_splits", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_block", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--rot_deg", type=float, default=45.0)
    ap.add_argument("--brightness", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="")
    ap.add_argument("--verbose", action="store_true")
    return ap


if __name__ == "__main__":
    run(get_parser().parse_args())
