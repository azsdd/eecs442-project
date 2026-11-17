"""Sample extra proxy-set candidates for one class (to balance bins after review), append to
ood_candidates.csv with new cids, and render one review sheet."""
from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from common import (OOD_CLASS_TO_BIN, OOD_ID_TO_CLASS, OOD_IMAGES, OOD_PARQUET, REVIEWED, make_sheets,
                    read_csv, write_csv)

CLASS_TO_ID = {v: k for k, v in OOD_ID_TO_CLASS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="biological")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=4421)
    ap.add_argument("--max_side", type=int, default=640)
    args = ap.parse_args()

    rows = read_csv(REVIEWED / "ood_candidates.csv")
    used = {int(r["parquet_index"]) for r in rows}
    table = pq.read_table(OOD_PARQUET)
    labels = table.column("label").to_numpy()
    idxs = [i for i in np.where(labels == CLASS_TO_ID[args.cls])[0].tolist() if i not in used]
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(idxs, args.n))
    start = len(rows)
    new = []
    for k, i in enumerate(chosen):
        rec = table.slice(i, 1).to_pylist()[0]
        im = Image.open(io.BytesIO(rec["image"]["bytes"])).convert("RGB")
        im.thumbnail((args.max_side, args.max_side))
        name = Path(rec["image"]["path"]).stem + ".jpg"
        out = OOD_IMAGES / args.cls / name
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out, quality=92)
        new.append({"cid": f"O{start + k:03d}", "orig_class": args.cls, "file": f"{args.cls}/{name}",
                    "proposed_bin": OOD_CLASS_TO_BIN[args.cls], "parquet_index": i,
                    "width": im.width, "height": im.height})
    rows += new
    write_csv(REVIEWED / "ood_candidates.csv", rows,
              ["cid", "orig_class", "file", "proposed_bin", "parquet_index", "width", "height"])
    sheet_prefix = f"ood_extra_{args.cls}"
    make_sheets([(OOD_IMAGES / r["file"], f"{r['cid']} {r['orig_class'][:11]}>{r['proposed_bin'][:4]}") for r in new],
                sheet_prefix)
    print(f"added {len(new)} candidates {new[0]['cid']}..{new[-1]['cid']}; sheet {sheet_prefix}_00.jpg")


if __name__ == "__main__":
    main()
