"""CLIP zero-shot baselines on both test sets (no training on waste images).

Two prompt designs:
  items : one prompt set per fine-grained item class (the 10 classes of the two-stage model), scores averaged
          over prompt templates, then mapped to bins with the campus rule table.
  bins  : direct bin prompts ("a photo of recyclable waste" ...).
Writes results/clip_zeroshot.json and results/runs/clipzs_<model>_<design>/preds_*.csv for downstream analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image

from common import BINS, BIN_TO_IDX, FINE_CLASSES, FINE_TO_BIN, FINE_TO_IDX, RESULTS, ROOT, RUNS, REVIEWED, fine_class, read_csv, write_csv
from train import metrics

ITEM_PROMPTS = {
    "Cardboard": ["a cardboard box", "cardboard packaging", "a flattened cardboard carton", "a cereal box"],
    "Paper": ["a sheet of paper", "a newspaper", "an envelope", "a paper flyer", "a magazine page"],
    "Metal": ["a metal can", "an aluminium drink can", "a tin can", "a metal lid"],
    "Plastic (rigid)": ["a plastic bottle", "a plastic container", "a plastic tub", "a clear plastic cup", "a plastic clamshell tray"],
    "Plastic (film)": ["a plastic bag", "a plastic wrapper", "plastic film", "a chip bag", "a snack wrapper", "bubble wrap", "a foam piece"],
    "Glass": ["a glass bottle", "a glass jar", "broken glass", "a wine bottle", "a beer bottle"],
    "Miscellaneous Trash": ["trash", "a face mask", "a toothbrush", "a diaper", "a pen", "a rubber part", "a toy"],
    "Textile Trash": ["clothes", "a shoe", "a sock", "a piece of fabric", "a rag"],
    "Food Organics": ["food waste", "a banana peel", "rotten fruit", "leftover food", "vegetable scraps", "bread"],
    "Vegetation": ["leaves", "grass clippings", "a branch", "a plant", "garden waste"],
}
TEMPLATES = ["a photo of {}.", "a photo of {} on the ground.", "a close-up photo of {}.", "a photo of {}, waste."]
BIN_PROMPTS = {
    "recycling": ["recyclable waste", "a recyclable item", "paper, cardboard, a metal can or a plastic bottle"],
    "compost": ["compostable food waste", "food scraps or plant material", "organic waste"],
    "landfill": ["landfill trash", "non-recyclable waste", "garbage that goes to landfill"],
}


@torch.no_grad()
def main() -> None:
    import open_clip

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=dev)
    tok = open_clip.get_tokenizer(args.model)
    model.eval()

    def text_feats(prompts):
        t = tok([tpl.format(p) for p in prompts for tpl in TEMPLATES]).to(dev)
        f = model.encode_text(t).float()
        f = f / f.norm(dim=-1, keepdim=True)
        return f.mean(0)

    item_w = torch.stack([text_feats(ITEM_PROMPTS[c]) for c in FINE_CLASSES])
    item_w = item_w / item_w.norm(dim=-1, keepdim=True)
    bin_w = torch.stack([text_feats(BIN_PROMPTS[b]) for b in BINS])
    bin_w = bin_w / bin_w.norm(dim=-1, keepdim=True)

    rows = read_csv(REVIEWED / "splits.csv")
    out = {"model": f"{args.model}/{args.pretrained}", "sets": {}}
    tag = f"clipzs_{args.model.replace('-', '').lower()}_{args.pretrained}"
    for split in ["test", "ood_test"]:
        rs = [r for r in rows if r["split"] == split]
        feats = []
        for i in range(0, len(rs), 64):
            batch = torch.stack([preprocess(Image.open(ROOT / r["file"]).convert("RGB")) for r in rs[i:i + 64]]).to(dev)
            f = model.encode_image(batch).float()
            feats.append(f / f.norm(dim=-1, keepdim=True))
        feats = torch.cat(feats)
        y_bin = np.array([BIN_TO_IDX[r["bin"]] for r in rs])
        y_fine = np.array([FINE_TO_IDX[fine_class(r)] for r in rs])
        res = {}
        # items design
        p_item = torch.softmax(100 * feats @ item_w.T, dim=-1).cpu().numpy()
        fine_pred = p_item.argmax(1)
        p_bin_items = np.zeros((len(rs), 3))
        for k, c in enumerate(FINE_CLASSES):
            p_bin_items[:, BIN_TO_IDX[FINE_TO_BIN[c]]] += p_item[:, k]
        pred_items = p_bin_items.argmax(1)
        res["items"] = metrics(y_bin, pred_items)
        res["items"]["fine_accuracy"] = float((fine_pred == y_fine).mean())
        # bins design
        p_bin_direct = torch.softmax(100 * feats @ bin_w.T, dim=-1).cpu().numpy()
        pred_direct = p_bin_direct.argmax(1)
        res["bins"] = metrics(y_bin, pred_direct)
        out["sets"][split] = res
        for design, pb, pred in (("items", p_bin_items, pred_items), ("bins", p_bin_direct, pred_direct)):
            d = RUNS / f"{tag}_{design}"
            d.mkdir(parents=True, exist_ok=True)
            recs = []
            for i, r in enumerate(rs):
                rec = {"cid": r["cid"], "orig_class": r["orig_class"], "true": BINS[int(y_bin[i])], "pred": BINS[int(pred[i])],
                       **{f"p_{b}": f"{pb[i, k]:.5f}" for k, b in enumerate(BINS)}}
                if design == "items":
                    rec["fine_true"] = FINE_CLASSES[int(y_fine[i])]
                    rec["fine_pred"] = FINE_CLASSES[int(fine_pred[i])]
                recs.append(rec)
            write_csv(d / f"preds_{split}.csv", recs)
        print(f"{split}: items->rule acc {res['items']['accuracy']:.3f} f1 {res['items']['macro_f1']:.3f} "
              f"(fine acc {res['items']['fine_accuracy']:.3f}) | direct bins acc {res['bins']['accuracy']:.3f} f1 {res['bins']['macro_f1']:.3f}")
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
