"""Recompute near-duplicate groups for the RealWaste candidates from saved DeiT features.

Rule (calibrated by eye on data/review_sheets/rw_dup_pairs.jpg):
  two candidates are linked if cosine similarity >= --thr_any (default 0.85), OR if they come from the
  same material folder with file indices at most --adj apart and similarity >= --thr_adj (default 0.70).
  The second clause catches the same object photographed twice in a row at the landfill reception.
Groups are the connected components; dup_group is rewritten in realwaste_candidates.csv.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import REVIEWED, read_csv, write_csv
from prepare_realwaste import union_find_groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr_any", type=float, default=0.85)
    ap.add_argument("--thr_adj", type=float, default=0.70)
    ap.add_argument("--adj", type=int, default=3)
    args = ap.parse_args()

    rows = read_csv(REVIEWED / "realwaste_candidates.csv")
    feats = np.load(REVIEWED / "realwaste_candidate_feats.npy")
    assert len(rows) == len(feats)
    sim = feats @ feats.T
    idx = [int(Path(r["file"]).stem.split("_")[-1]) for r in rows]
    pairs, why = [], Counter()
    n = len(rows)
    for a in range(n):
        for b in range(a + 1, n):
            s = float(sim[a, b])
            adjacent = rows[a]["material"] == rows[b]["material"] and abs(idx[a] - idx[b]) <= args.adj
            if s >= args.thr_any:
                pairs.append((a, b))
                why["similar"] += 1
            elif adjacent and s >= args.thr_adj:
                pairs.append((a, b))
                why["adjacent"] += 1
    groups = union_find_groups(n, pairs)
    for r, g in zip(rows, groups):
        r["dup_group"] = g
    write_csv(REVIEWED / "realwaste_candidates.csv", rows, ["cid", "material", "file", "proposed_bin", "dup_group"])
    sizes = Counter(groups)
    hist = Counter(sizes.values())
    print(f"links: {dict(why)}  groups: {len(sizes)}  size histogram: {dict(sorted(hist.items()))}")
    big = [g for g, c in sizes.items() if c >= 3]
    for g in sorted(big, key=lambda g: -sizes[g])[:12]:
        members = [r["cid"] for r in rows if r["dup_group"] == g]
        print(f"  group {g} ({sizes[g]}): {' '.join(members)}")


if __name__ == "__main__":
    main()
