"""Foreground masks for RealWaste training images (for background-swap augmentation).

GrabCut initialised with a central rectangle on the 256x256 preloaded image; masks whose foreground covers
less than 2% or more than 85% of the image are discarded (the image is then used without swapping).
Writes data/masks/<cid>.png (0/255) and a contact sheet data/review_sheets/masks_preview.jpg.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from PIL import Image

from common import DATA, REVIEWED, ROOT, SHEETS, read_csv
from train import load_image_256

MASKS = DATA / "masks"


def grabcut_mask(img: np.ndarray, iters: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.06), int(h * 0.06), int(w * 0.88), int(h * 0.88))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return fg


def main() -> None:
    MASKS.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_csv(REVIEWED / "splits.csv") if r["source"] == "realwaste" and r["split"] == "train"]
    kept, previews = 0, []
    for i, r in enumerate(rows):
        x = load_image_256(ROOT / r["file"]).permute(1, 2, 0).numpy()
        fg = grabcut_mask(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
        frac = fg.mean() / 255.0
        if 0.02 <= frac <= 0.85:
            Image.fromarray(fg).save(MASKS / f"{r['cid']}.png")
            kept += 1
            if len(previews) < 30 and i % 15 == 0:
                over = x.copy()
                over[fg == 0] = (0.35 * over[fg == 0] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
                previews.append(over)
        if i % 100 == 0:
            print(f"{i}/{len(rows)} masks, kept {kept}")
    print(f"kept {kept}/{len(rows)} masks")
    if previews:
        cols = 6
        rows_n = (len(previews) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 256, rows_n * 256), "white")
        for k, p in enumerate(previews):
            sheet.paste(Image.fromarray(p), ((k % cols) * 256, (k // cols) * 256))
        sheet.save(SHEETS / "masks_preview.jpg", quality=85)


if __name__ == "__main__":
    main()
