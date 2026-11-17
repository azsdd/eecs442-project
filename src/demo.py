"""Gradio demo (v2): upload a photo -> item type, predicted U-M bin, calibrated class scores, and an
abstain-with-hints answer when the model is not confident.

  python -I src/demo.py [--engine twostage|direct|clip] [--port 7860]

Engines
  twostage : DeiT-Tiny 10-way item head + campus rule table (results/runs/fine_lastblock_brightrot_f1.0_s0)
  direct   : DeiT-Tiny 3-way head, the main model of the report
  clip     : CLIP ViT-B/32 zero-shot item prompts + rule table (no training on waste images)
Temperature scaling and the abstention threshold (80% coverage) are fitted on the validation split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import BINS, FINE_CLASSES, FINE_TO_BIN, RUNS
from predict import BIN_TEXT, load_model, predict_pil

HINTS = {
    frozenset(["recycling", "landfill"]): "Rigid plastic (bottle, tub, cup, clamshell), clean paper, cardboard, cartons and metal cans go to **recycling**. Plastic bags, film, wrappers, chip bags, foam and all glass go to **landfill** on campus.",
    frozenset(["compost", "landfill"]): "Food (including meat and bones), napkins, paper towels, plant material and BPI-certified compostable ware go to **compost**. Non-compostable cups, condiment packets and plastic go to **landfill**.",
    frozenset(["recycling", "compost"]): "Clean paper and cardboard go to **recycling**; food-soiled paper, napkins and paper towels go to **compost**.",
}


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200))

    def nll(T):
        z = logits / T
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()

    return float(min(grid, key=nll))


CLIP_MODEL = "ViT-B-32"  # overridden by --clip_model (ViT-L-14 is the most accurate, ~1.7 GB)


class Engine:
    def __init__(self, kind: str, device: str):
        self.kind, self.device = kind, device
        self.temperature, self.threshold = 1.0, 0.6
        if kind == "clip":
            import open_clip

            from clip_zeroshot import ITEM_PROMPTS, TEMPLATES

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained="openai", device=device)
            tok = open_clip.get_tokenizer(CLIP_MODEL)
            with torch.no_grad():
                ws = []
                for c in FINE_CLASSES:
                    t = tok([tpl.format(p) for p in ITEM_PROMPTS[c] for tpl in TEMPLATES]).to(device)
                    f = self.model.encode_text(t).float()
                    f = (f / f.norm(dim=-1, keepdim=True)).mean(0)
                    ws.append(f / f.norm())
                self.item_w = torch.stack(ws)
            self.name = f"CLIP {CLIP_MODEL} zero-shot (item prompts + rule table)"
            return
        run = RUNS / ("fine_lastblock_brightrot_f1.0_s0" if kind == "twostage" else "lastblock_brightrot_f1.0_s0")
        self.model, self.cfg = load_model(run, device)
        self.target = self.cfg.get("target", "bin")
        self.name = f"{run.name} ({'10-way item head + rule table' if self.target == 'fine' else '3-way head'})"
        val = pd.read_csv(run / "preds_val.csv")
        if self.target == "fine":
            logits = np.log(np.clip(val[[f"pf_{k}" for k in range(len(FINE_CLASSES))]].to_numpy(float), 1e-12, 1))
            y = val["fine_true"].map({c: i for i, c in enumerate(FINE_CLASSES)}).to_numpy()
        else:
            logits = np.log(np.clip(val[[f"p_{b}" for b in BINS]].to_numpy(float), 1e-12, 1))
            y = val["true"].map({b: i for i, b in enumerate(BINS)}).to_numpy()
        self.temperature = fit_temperature(logits, y)
        z = logits / self.temperature
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        if self.target == "fine":
            pb = np.zeros((len(p), 3))
            for k, c in enumerate(FINE_CLASSES):
                pb[:, BINS.index(FINE_TO_BIN[c])] += p[:, k]
            p = pb
        self.threshold = float(np.quantile(p.max(1), 0.2))  # abstain on the least confident 20% of validation

    def predict(self, im: Image.Image):
        if self.kind == "clip":
            with torch.no_grad():
                x = self.preprocess(im.convert("RGB")).unsqueeze(0).to(self.device)
                f = self.model.encode_image(x).float()
                f = f / f.norm(dim=-1, keepdim=True)
                item = torch.softmax(100 * f @ self.item_w.T, dim=-1)[0].cpu().numpy()
            bins = np.zeros(3)
            for k, c in enumerate(FINE_CLASSES):
                bins[BINS.index(FINE_TO_BIN[c])] += item[k]
            return bins, item
        bins, head = predict_pil(self.model, im, self.device, self.target, self.temperature)
        return bins, (head if self.target == "fine" else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["twostage", "direct", "clip"], default="twostage")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--clip_model", default="ViT-B-32", help="open_clip model for the clip engine, e.g. ViT-L-14")
    args = ap.parse_args()
    global CLIP_MODEL
    CLIP_MODEL = args.clip_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engines = {}

    def get_engine(kind):
        if kind not in engines:
            try:
                engines[kind] = Engine(kind, device)
            except Exception as e:  # a missing run or package
                return None, f"engine {kind} unavailable: {e}"
        return engines[kind], ""

    def classify(im: Image.Image, kind: str):
        if im is None:
            return "Upload a photo of one item.", {}, {}
        eng, err = get_engine(kind)
        if eng is None:
            return err, {}, {}
        bins, item = eng.predict(im)
        order = np.argsort(-bins)
        k, k2 = int(order[0]), int(order[1])
        conf = float(bins[k])
        scores = {b: float(p) for b, p in zip(BINS, bins)}
        item_scores = {c: float(p) for c, p in zip(FINE_CLASSES, item)} if item is not None else {}
        lines = [f"**Model:** {eng.name}"]
        if item is not None:
            j = int(item.argmax())
            lines.append(f"**Item type:** {FINE_CLASSES[j]} ({item[j]:.0%}) → campus rule: {FINE_TO_BIN[FINE_CLASSES[j]]}")
        if conf >= eng.threshold:
            lines.append(f"## {BINS[k].upper()} ({conf:.0%} calibrated confidence)")
            lines.append(BIN_TEXT[BINS[k]])
        else:
            lines.append(f"## Not sure: {BINS[k]} ({conf:.0%}) or {BINS[k2]} ({bins[k2]:.0%})")
            lines.append("Check the item against the rule before you throw it: " + HINTS[frozenset([BINS[k], BINS[k2]])])
            lines.append(f"(Confidence below the {eng.threshold:.0%} threshold set on the validation split; "
                         f"ask the [Where to Throw guide](https://ocsi.umich.edu/resources/where-to-throw/) if still unsure.)")
        return "\n\n".join(lines), scores, item_scores

    with gr.Blocks(title="Campus Waste Sorting (U-M bins)") as demo:
        gr.Markdown("# Campus Waste Sorting (U-M bins)\nUpload a photo of one item. The model names the item type, "
                    "the campus bin and its calibrated confidence, and abstains with a hint when unsure. "
                    "Campus rules: glass and plastic film go to landfill.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Photo of one item")
                eng = gr.Dropdown(["twostage", "direct", "clip"], value=args.engine, label="Engine")
                btn = gr.Button("Sort it", variant="primary")
            with gr.Column():
                verdict = gr.Markdown()
                bin_lab = gr.Label(num_top_classes=3, label="Bin scores (calibrated)")
                item_lab = gr.Label(num_top_classes=4, label="Item type scores")
        btn.click(classify, [inp, eng], [verdict, bin_lab, item_lab])
        inp.change(classify, [inp, eng], [verdict, bin_lab, item_lab])
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
