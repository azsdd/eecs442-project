"""Step 1 (proxy "new photo" test set): sample candidates from the 12-class garbage parquet,
write them as JPEGs, and build review contact sheets.

Outputs:
  data/raw/ood/images/<class>/<name>.jpg
  data/reviewed/ood_candidates.csv   cid, orig_class, file, proposed_bin, parquet_index, width, height
  data/review_sheets/ood_sheet_XX.jpg
"""
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

from common import (OOD_CLASS_TO_BIN, OOD_ID_TO_CLASS, OOD_IMAGES, OOD_PARQUET, REVIEWED, SHEETS,
                    make_sheets, write_csv)

# Oversample; review trims to a balanced ~120-image set (~40 per bin).
CANDIDATES = {
    "biological": 48,
    "cardboard": 14, "paper": 14, "plastic": 16, "metal": 14,
    "brown-glass": 8, "green-glass": 8, "white-glass": 8,
    "trash": 24, "clothes": 12, "shoes": 8,
}
CLASS_TO_ID = {v: k for k, v in OOD_ID_TO_CLASS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=442)
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--max_side", type=int, default=640)
    args = ap.parse_args()

    table = pq.read_table(OOD_PARQUET)
    labels = table.column("label").to_numpy()
    rng = random.Random(args.seed)
    rows = []
    for cls, n in CANDIDATES.items():
        idxs = np.where(labels == CLASS_TO_ID[cls])[0].tolist()
        for i in sorted(rng.sample(idxs, n)):
            rec = table.slice(i, 1).to_pylist()[0]
            im = Image.open(io.BytesIO(rec["image"]["bytes"])).convert("RGB")
            im.thumbnail((args.max_side, args.max_side))
            name = Path(rec["image"]["path"]).stem + ".jpg"
            out = OOD_IMAGES / cls / name
            out.parent.mkdir(parents=True, exist_ok=True)
            im.save(out, quality=92)
            rows.append({
                "orig_class": cls, "file": f"{cls}/{name}", "proposed_bin": OOD_CLASS_TO_BIN[cls],
                "parquet_index": i, "width": im.width, "height": im.height,
            })
    for k, r in enumerate(rows):
        r["cid"] = f"O{k:03d}"
    write_csv(REVIEWED / "ood_candidates.csv", rows,
              ["cid", "orig_class", "file", "proposed_bin", "parquet_index", "width", "height"])
    print(f"wrote {len(rows)} candidates")
    if args.sheets:
        n = make_sheets([(OOD_IMAGES / r["file"], f"{r['cid']} {r['orig_class'][:11]}>{r['proposed_bin'][:4]}")
                         for r in rows], "ood_sheet")
        print(f"{n} sheets written to {SHEETS}")


if __name__ == "__main__":
    main()
