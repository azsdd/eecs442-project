"""Step 2: apply review decisions and build the benchmark splits.

Review files (data/reviewed/*_review.csv) list only exceptions to the proposed bin:
  cid, action (keep | exclude | relabel), final_bin (for relabel), note
Candidates not listed are kept with their proposed bin.

Output: data/reviewed/splits.csv with columns
  cid, source (realwaste | ood), file (relative to project root), orig_class, bin, dup_group, split
Split for RealWaste is ~70/15/15 train/val/test, stratified by bin, with near-duplicate groups kept
together. All kept proxy images get split = ood_test.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BINS, OOD_IMAGES, RAW_RW, REVIEWED, ROOT, read_csv, write_csv


def load_review(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["cid"]: r for r in read_csv(path)}


def apply_review(cands: list[dict], review: dict) -> tuple[list[dict], Counter]:
    kept, stats = [], Counter()
    for c in cands:
        rv = review.get(c["cid"])
        action = (rv or {}).get("action", "keep").strip().lower() or "keep"
        if action == "exclude":
            stats["exclude"] += 1
            continue
        b = c["proposed_bin"]
        if action == "relabel":
            b = rv["final_bin"].strip().lower()
            assert b in BINS, f"bad bin {b} for {c['cid']}"
            stats["relabel"] += 1
        else:
            stats["keep"] += 1
        c = dict(c)
        c["bin"] = b
        c["note"] = (rv or {}).get("note", "")
        kept.append(c)
    return kept, stats


def group_split(rows: list[dict], seed: int, frac_val: float = 0.15, frac_test: float = 0.15) -> None:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r["dup_group"]].append(r)
    by_bin = defaultdict(list)
    for g, members in groups.items():
        bins = Counter(m["bin"] for m in members)
        if len(bins) > 1:
            print(f"  warning: group {g} mixes bins {dict(bins)}; assigned by majority")
        by_bin[bins.most_common(1)[0][0]].append(members)
    for b in BINS:
        glist = by_bin[b]
        rng.shuffle(glist)
        n = sum(len(m) for m in glist)
        n_test, n_val = round(frac_test * n), round(frac_val * n)
        c_test = c_val = 0
        # Large duplicate groups (>= 4 near-identical photos) go to train so that no single object
        # dominates a small evaluation split.
        big = [m for m in glist if len(m) >= 4]
        for members in big:
            for m in members:
                m["split"] = "train"
        glist = [m for m in glist if len(m) < 4]
        for members in glist:
            if c_test < n_test:
                split = "test"
                c_test += len(members)
            elif c_val < n_val:
                split = "val"
                c_val += len(members)
            else:
                split = "train"
            for m in members:
                m["split"] = split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=442)
    ap.add_argument("--max_per_bin", type=int, default=0, help="cap kept RealWaste images per bin (0 = none)")
    args = ap.parse_args()

    rw, st = apply_review(read_csv(REVIEWED / "realwaste_candidates.csv"), load_review(REVIEWED / "realwaste_review.csv"))
    print("RealWaste review:", dict(st), "kept:", Counter(r["bin"] for r in rw))
    if args.max_per_bin:
        rng = random.Random(args.seed + 1)
        capped = []
        for b in BINS:
            rb = [r for r in rw if r["bin"] == b]
            rng.shuffle(rb)
            capped += rb[:args.max_per_bin]
        rw = sorted(capped, key=lambda r: r["cid"])
        print("after cap:", Counter(r["bin"] for r in rw))
    group_split(rw, args.seed)
    out = []
    for r in rw:
        out.append({
            "cid": r["cid"], "source": "realwaste", "file": (RAW_RW / r["file"]).relative_to(ROOT).as_posix(),
            "orig_class": r["material"], "bin": r["bin"], "dup_group": f"rw{r['dup_group']}", "split": r["split"],
            "note": r.get("note", ""),
        })
    ood, st = apply_review(read_csv(REVIEWED / "ood_candidates.csv"), load_review(REVIEWED / "ood_review.csv"))
    print("Proxy review:", dict(st), "kept:", Counter(r["bin"] for r in ood))
    for r in ood:
        out.append({
            "cid": r["cid"], "source": "ood", "file": (OOD_IMAGES / r["file"]).relative_to(ROOT).as_posix(),
            "orig_class": r["orig_class"], "bin": r["bin"], "dup_group": f"ood{r['cid']}", "split": "ood_test",
            "note": r.get("note", ""),
        })
    write_csv(REVIEWED / "splits.csv", out, ["cid", "source", "file", "orig_class", "bin", "dup_group", "split", "note"])

    print("\nsplit x bin counts:")
    for split in ["train", "val", "test", "ood_test"]:
        c = Counter(r["bin"] for r in out if r["split"] == split)
        print(f"  {split:9s} " + "  ".join(f"{b}={c[b]:3d}" for b in BINS) + f"  total={sum(c.values())}")


if __name__ == "__main__":
    main()
