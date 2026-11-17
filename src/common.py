"""Shared constants and helpers for the campus waste-sorting project."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_RW = DATA / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste"
RAW_OOD = DATA / "raw" / "ood"
OOD_PARQUET = RAW_OOD / "hf" / "data" / "train-00000-of-00001.parquet"
OOD_IMAGES = RAW_OOD / "images"
REVIEWED = DATA / "reviewed"
SHEETS = DATA / "review_sheets"
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
FIGURES = RESULTS / "figures"

for _d in (REVIEWED, SHEETS, RESULTS, RUNS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Target classes = U-M disposal bins.
BINS = ["recycling", "compost", "landfill"]
BIN_TO_IDX = {b: i for i, b in enumerate(BINS)}

# Default material -> bin mapping under U-M Ann Arbor campus "Where to Throw" rules.
#  * Campus recycling takes paper, cardboard, metal cans and rigid plastics #1-2, #4-7
#    (bottles, tubs, cups, containers); plastic film is not accepted (OCSI plastics fact sheet).
#  * Glass is NOT accepted by the campus recycling partner (OCSI FAQ), so it goes to landfill.
#  * Compost takes food scraps (incl. meat), coffee grounds/filters, napkins, paper towels,
#    BPI-certified compostable ware (OCSI composting page); plant material is compostable.
#  * Plastic bags / film / wrappers / foam, snack bags, sauce packets and non-compostable
#    cups are landfill.
# Every candidate image is still visually reviewed; the review can override this default.
MATERIAL_TO_BIN = {
    "Cardboard": "recycling",
    "Paper": "recycling",
    "Metal": "recycling",
    "Plastic": "recycling",
    "Glass": "landfill",
    "Miscellaneous Trash": "landfill",
    "Textile Trash": "landfill",
    "Food Organics": "compost",
    "Vegetation": "compost",
}

# Default mapping for the proxy "new photo" set (12-class garbage dataset).
OOD_CLASS_TO_BIN = {
    "cardboard": "recycling",
    "paper": "recycling",
    "plastic": "recycling",
    "metal": "recycling",
    "biological": "compost",
    "brown-glass": "landfill",
    "green-glass": "landfill",
    "white-glass": "landfill",
    "trash": "landfill",
    "clothes": "landfill",
    "shoes": "landfill",
    # "battery" is hazardous/universal waste at U-M (EHS Hazmat) -> excluded entirely.
}
OOD_ID_TO_CLASS = {
    0: "battery", 1: "biological", 2: "brown-glass", 3: "cardboard", 4: "clothes",
    5: "green-glass", 6: "metal", 7: "paper", 8: "plastic", 9: "shoes", 10: "trash",
    11: "white-glass",
}

MODEL_NAME = "deit_tiny_patch16_224.fb_in1k"
BACKBONES = {
    "deit_tiny": "deit_tiny_patch16_224.fb_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
    "clip_b32": "vit_base_patch32_clip_224.openai",
}

# Fine-grained item classes for the two-stage model (item classifier + campus rule table).
FINE_CLASSES = ["Cardboard", "Paper", "Metal", "Plastic (rigid)", "Plastic (film)", "Glass",
                "Miscellaneous Trash", "Textile Trash", "Food Organics", "Vegetation"]
FINE_TO_IDX = {c: i for i, c in enumerate(FINE_CLASSES)}
FINE_TO_BIN = {  # U-M Ann Arbor campus rule table
    "Cardboard": "recycling", "Paper": "recycling", "Metal": "recycling", "Plastic (rigid)": "recycling",
    "Plastic (film)": "landfill", "Glass": "landfill", "Miscellaneous Trash": "landfill",
    "Textile Trash": "landfill", "Food Organics": "compost", "Vegetation": "compost",
}
_OOD_FINE = {"cardboard": "Cardboard", "paper": "Paper", "metal": "Metal", "plastic": "Plastic (rigid)",
             "brown-glass": "Glass", "green-glass": "Glass", "white-glass": "Glass", "trash": "Miscellaneous Trash",
             "clothes": "Textile Trash", "shoes": "Textile Trash", "biological": "Food Organics"}


def fine_class(row: dict) -> str:
    """Fine-grained item class of a splits.csv row (material, with plastic split by the reviewed bin)."""
    oc, b = row["orig_class"], row["bin"]
    if row["source"] == "realwaste":
        if oc == "Plastic":
            return "Plastic (film)" if b == "landfill" else "Plastic (rigid)"
        return oc
    if oc == "plastic":
        return "Plastic (film)" if b == "landfill" else "Plastic (rigid)"
    if oc == "trash" and b == "compost":
        return "Food Organics"
    return _OOD_FINE[oc]


IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def font(size: int = 22):
    from PIL import ImageFont

    for cand in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def fit_square(im, tile: int):
    """Resize preserving aspect ratio and pad to tile x tile (white background)."""
    from PIL import Image

    im = im.convert("RGB")
    w, h = im.size
    s = tile / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))))
    canvas = Image.new("RGB", (tile, tile), "white")
    canvas.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2))
    return canvas


def make_sheets(items, prefix: str, cols: int = 5, nrows: int = 4, tile: int = 256) -> int:
    """items: list of (image_path, label_text). Writes SHEETS/<prefix>_XX.jpg; returns #sheets."""
    from PIL import Image, ImageDraw

    fnt = font(22)
    per = cols * nrows
    strip = 28
    n = 0
    for s in range(0, len(items), per):
        chunk = items[s:s + per]
        sheet = Image.new("RGB", (cols * tile, nrows * (tile + strip)), "white")
        d = ImageDraw.Draw(sheet)
        for k, (path, text) in enumerate(chunk):
            im = fit_square(Image.open(path), tile)
            x = (k % cols) * tile
            y = (k // cols) * (tile + strip)
            sheet.paste(im, (x, y + strip))
            d.text((x + 4, y + 2), text, fill="black", font=fnt)
        sheet.save(SHEETS / f"{prefix}_{s // per:02d}.jpg", quality=88)
        n += 1
    return n


def read_csv(path):
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    import csv

    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
