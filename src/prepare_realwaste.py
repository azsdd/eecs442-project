"""Step 1 (RealWaste): sample candidate images, find near-duplicates, build review contact sheets.

Outputs (data/reviewed):
  realwaste_candidates.csv   one row per candidate: cid, material, file, proposed_bin, dup_group
  realwaste_dup_pairs.csv    candidate pairs above the similarity threshold
Outputs (data/review_sheets):
  rw_sheet_XX.jpg            20 candidates per sheet, labelled with cid / material -> proposed bin
  rw_dup_pairs.jpg           the most similar pairs, for threshold calibration
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `python -I` (isolated mode)

import numpy as np
import torch
from PIL import Image, ImageDraw

from common import (MATERIAL_TO_BIN, MODEL_NAME, RAW_RW, REVIEWED, SHEETS, font, make_sheets,
                    write_csv)

# Oversample so that exclusions during review still leave ~600 images, ~200 per bin.
CANDIDATES_PER_CLASS = {
    "Cardboard": 60,
    "Paper": 60,
    "Metal": 60,
    "Plastic": 90,  # bags/film/foam will be moved to landfill or excluded at review
    "Glass": 75,
    "Miscellaneous Trash": 80,
    "Textile Trash": 70,
    "Food Organics": 115,
    "Vegetation": 115,
}


def sample_candidates(seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for material, n in CANDIDATES_PER_CLASS.items():
        files = sorted((RAW_RW / material).glob("*.jpg"), key=lambda p: int(p.stem.split("_")[-1]))
        chosen = sorted(rng.sample(files, n), key=lambda p: int(p.stem.split("_")[-1]))
        for f in chosen:
            rows.append({
                "material": material,
                "file": f.relative_to(RAW_RW).as_posix(),
                "proposed_bin": MATERIAL_TO_BIN[material],
            })
    for i, r in enumerate(rows):
        r["cid"] = f"R{i:03d}"
    return rows


@torch.no_grad()
def extract_features(paths: list, bs: int = 64) -> np.ndarray:
    import timm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).eval().to(dev)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)
    feats = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(dev)
        f = torch.nn.functional.normalize(model(batch), dim=-1)
        feats.append(f.cpu())
    return torch.cat(feats).numpy()


def union_find_groups(n: int, pairs: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return [remap[r] for r in roots]


def make_pair_sheet(rows: list[dict], pairs: list[tuple[int, int, float]], out: str, tile: int = 200) -> None:
    fnt = font(20)
    cols = 4  # 4 pairs per row
    nrows = (len(pairs) + cols - 1) // cols
    strip = 26
    sheet = Image.new("RGB", (cols * (2 * tile + 12), nrows * (tile + strip)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (a, b, s) in enumerate(pairs):
        x = (k % cols) * (2 * tile + 12)
        y = (k // cols) * (tile + strip)
        for j, idx in enumerate((a, b)):
            im = Image.open(RAW_RW / rows[idx]["file"]).convert("RGB").resize((tile, tile))
            sheet.paste(im, (x + j * tile, y + strip))
        d.text((x + 2, y + 2), f"{rows[a]['cid']}~{rows[b]['cid']} sim={s:.3f}", fill="black", font=fnt)
    sheet.save(SHEETS / out, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=442)
    ap.add_argument("--dup_threshold", type=float, default=0.90)
    ap.add_argument("--sheets", action="store_true", help="also render contact sheets")
    args = ap.parse_args()

    rows = sample_candidates(args.seed)
    paths = [RAW_RW / r["file"] for r in rows]
    feats = extract_features(paths)
    sim = feats @ feats.T
    np.fill_diagonal(sim, -1)
    iu = np.triu_indices(len(rows), k=1)
    sims = sim[iu]
    order = np.argsort(-sims)
    top = [(int(iu[0][o]), int(iu[1][o]), float(sims[o])) for o in order[:48]]
    pairs = [(int(a), int(b), float(s)) for a, b, s in zip(iu[0], iu[1], sims) if s >= args.dup_threshold]
    groups = union_find_groups(len(rows), [(a, b) for a, b, _ in pairs])
    for r, g in zip(rows, groups):
        r["dup_group"] = g

    write_csv(REVIEWED / "realwaste_candidates.csv", rows, ["cid", "material", "file", "proposed_bin", "dup_group"])
    write_csv(REVIEWED / "realwaste_dup_pairs.csv",
              [{"cid_a": rows[a]["cid"], "cid_b": rows[b]["cid"], "sim": f"{s:.4f}"}
               for a, b, s in sorted(pairs, key=lambda t: -t[2])] or [{"cid_a": "", "cid_b": "", "sim": ""}],
              ["cid_a", "cid_b", "sim"])
    np.save(REVIEWED / "realwaste_candidate_feats.npy", feats)

    n_groups = len(set(groups))
    print(f"candidates={len(rows)} pairs>={args.dup_threshold}: {len(pairs)} groups={n_groups}")
    print("top similarities:", [f"{s:.3f}" for _, _, s in top[:16]])
    make_pair_sheet(rows, top, "rw_dup_pairs.jpg")
    if args.sheets:
        n = make_sheets([(RAW_RW / r["file"], f"{r['cid']} {r['material'][:11]}>{r['proposed_bin'][:4]}") for r in rows],
                        "rw_sheet")
        print(f"{n} sheets written to {SHEETS}")


if __name__ == "__main__":
    main()
