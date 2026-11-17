"""Targeted-vs-random extra annotation experiment (RealWaste).

sample : draw NEW candidates not in the benchmark: 60 from the Plastic folder (film pool) and 40 spread over
         all materials (random pool); flag near-duplicates of validation/test images; render review sheets
         (rwx_sheet_XX.jpg) and write data/reviewed/realwaste_extra_candidates.csv (cids E000...).
finalize: apply data/reviewed/realwaste_extra_review.csv (cid, action, final_bin, note; exceptions only) and append
         rows to splits.csv with split = extra_film (film-labelled plastics from the film pool) and
         split = extra_random (kept random-pool images, subsampled to the same count).
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import BINS, MATERIAL_TO_BIN, RAW_RW, REVIEWED, ROOT, make_sheets, read_csv, write_csv
from finalize_splits import apply_review, load_review
from prepare_realwaste import extract_features

RANDOM_PER_CLASS = {"Cardboard": 5, "Paper": 5, "Metal": 4, "Plastic": 4, "Glass": 4, "Miscellaneous Trash": 5,
                    "Textile Trash": 4, "Food Organics": 5, "Vegetation": 4}


def sample(args) -> None:
    rng = random.Random(args.seed)
    used = {Path(r["file"]).as_posix() for r in read_csv(REVIEWED / "realwaste_candidates.csv")}
    existing = []
    if args.append_film:
        existing = read_csv(REVIEWED / "realwaste_extra_candidates.csv")
        used |= {r["file"] for r in existing}
    rows = []
    plan = [("Plastic", args.append_film)] if args.append_film else [("Plastic", args.n_film)] + list(RANDOM_PER_CLASS.items())
    for material, n in plan:
        files = [f for f in sorted((RAW_RW / material).glob("*.jpg")) if f.relative_to(RAW_RW).as_posix() not in used]
        pool = "film_pool" if (material == "Plastic" and len(rows) == 0) else "random_pool"
        chosen = rng.sample(files, n)
        for f in chosen:
            used.add(f.relative_to(RAW_RW).as_posix())
            rows.append({"material": material, "file": f.relative_to(RAW_RW).as_posix(),
                         "proposed_bin": MATERIAL_TO_BIN[material], "pool": pool})
    for i, r in enumerate(rows):
        r["cid"] = f"E{len(existing) + i:03d}"
    feats = extract_features([RAW_RW / r["file"] for r in rows])
    cands = read_csv(REVIEWED / "realwaste_candidates.csv")
    cfeats = np.load(REVIEWED / "realwaste_candidate_feats.npy")
    eval_cids = {r["cid"] for r in read_csv(REVIEWED / "splits.csv") if r["split"] in ("val", "test")}
    ev = np.array([i for i, c in enumerate(cands) if c["cid"] in eval_cids])
    sim = feats @ cfeats[ev].T
    for r, s in zip(rows, sim.max(1)):
        r["max_sim_eval"] = f"{float(s):.3f}"
        r["dup_of_eval"] = int(s >= args.dup_threshold)
    write_csv(REVIEWED / "realwaste_extra_candidates.csv", existing + rows,
              ["cid", "material", "file", "proposed_bin", "pool", "max_sim_eval", "dup_of_eval"])
    prefix = "rwx2_sheet" if args.append_film else "rwx_sheet"
    n = make_sheets([(RAW_RW / r["file"], f"{r['cid']} {r['material'][:11]}>{r['proposed_bin'][:4]}{' DUP' if r['dup_of_eval'] else ''}")
                     for r in rows], prefix)
    print(f"{len(rows)} extra candidates, {sum(r['dup_of_eval'] for r in rows)} flagged as near-duplicates of val/test; {n} sheets ({prefix})")


def finalize(args) -> None:
    cands = [r for r in read_csv(REVIEWED / "realwaste_extra_candidates.csv") if not int(r["dup_of_eval"])]
    kept, st = apply_review(cands, load_review(REVIEWED / "realwaste_extra_review.csv"))
    print("extra review:", dict(st))
    film = [r for r in kept if r["pool"] == "film_pool" and r["bin"] == "landfill"]
    rnd = [r for r in kept if r["pool"] == "random_pool"]
    rng = random.Random(args.seed)
    rng.shuffle(rnd)
    rnd = sorted(rnd[:len(film)], key=lambda r: r["cid"])
    print(f"extra_film: {len(film)} film images; extra_random: {len(rnd)} images {Counter(r['bin'] for r in rnd)}")
    splits = [r for r in read_csv(REVIEWED / "splits.csv") if not r["split"].startswith("extra_")]
    for split, group in (("extra_film", film), ("extra_random", rnd)):
        for r in group:
            splits.append({"cid": r["cid"], "source": "realwaste", "file": (RAW_RW / r["file"]).relative_to(ROOT).as_posix(),
                           "orig_class": r["material"], "bin": r["bin"], "dup_group": f"x{r['cid']}", "split": split,
                           "note": r.get("note", "")})
    write_csv(REVIEWED / "splits.csv", splits, ["cid", "source", "file", "orig_class", "bin", "dup_group", "split", "note"])
    print("splits.csv updated:", Counter(r["split"] for r in splits))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "finalize"])
    ap.add_argument("--seed", type=int, default=4422)
    ap.add_argument("--n_film", type=int, default=60)
    ap.add_argument("--append_film", type=int, default=0, help="append N more Plastic candidates to an existing list")
    ap.add_argument("--dup_threshold", type=float, default=0.85)
    args = ap.parse_args()
    sample(args) if args.cmd == "sample" else finalize(args)


if __name__ == "__main__":
    main()
