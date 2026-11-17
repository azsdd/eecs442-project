# Campus Waste Sorting with Vision Transformers (EECS 442)

Photo-based classifier for three U-M campus disposal bins (recycling / compost / landfill), built on a
pretrained DeiT-Tiny. Trained on reviewed RealWaste images relabeled with U-M "Where to Throw" rules and
evaluated on a held-out RealWaste split plus a second, different-source test set.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install timm scikit-learn matplotlib pandas seaborn huggingface_hub gradio pyarrow datasets
```

Scripts are run in isolated mode (`python -I`) because they read downloaded data.

## Data

- `data/raw/realwaste/realwaste.zip` from https://archive.ics.uci.edu/static/public/908/realwaste.zip (extract to
  `data/raw/realwaste/extracted/`).
- `data/raw/ood/hf/data/train-00000-of-00001.parquet` from the Hugging Face dataset
  `NeoAivara/Waste_Classification_data` (12-class garbage photos) used as the proxy "new photo" test set.

## Pipeline

```powershell
.\.venv\Scripts\python.exe -I src\prepare_realwaste.py --sheets   # sample 725 candidates, features, contact sheets
.\.venv\Scripts\python.exe -I src\dup_groups.py                   # near-duplicate groups (similarity + adjacency)
.\.venv\Scripts\python.exe -I src\prepare_ood.py --sheets         # 174 proxy candidates + sheets
.\.venv\Scripts\python.exe -I src\prepare_ood_extra.py --cls biological --n 20
# visual review -> data/reviewed/realwaste_review.csv, ood_review.csv (exceptions only) + review_notes_*.md
.\.venv\Scripts\python.exe -I src\finalize_splits.py              # -> data/reviewed/splits.csv (group-aware 70/15/15)
.\.venv\Scripts\python.exe -I src\sweep.py                        # validation LR sweep
.\.venv\Scripts\python.exe -I src\run_grid.py --lr_head_linear 1e-2 --lr_head_ft 1e-3 --lr_block 1e-4 --epochs 40
.\.venv\Scripts\python.exe -I src\eval_shift.py                   # brightness / rotation robustness check
.\.venv\Scripts\python.exe -I src\analyze.py                      # tables + figures in results/
.\.venv\Scripts\python.exe -I src\predict.py --image photo.jpg --out pred.png
.\.venv\Scripts\python.exe -I src\demo.py                         # Gradio demo at http://127.0.0.1:7860
```

## Extensions beyond the proposal

```powershell
.\.venv\Scripts\python.exe -I src\make_masks.py                                   # GrabCut masks for background-swap aug
.\.venv\Scripts\python.exe -I src\run_extensions.py --stages twostage backbones augs
.\.venv\Scripts\python.exe -I src\prepare_extra_realwaste.py sample               # extra candidates + sheets (review -> realwaste_extra_review.csv)
.\.venv\Scripts\python.exe -I src\prepare_extra_realwaste.py finalize             # extra_film / extra_random splits
.\.venv\Scripts\python.exe -I src\run_extensions.py --stages targeted
.\.venv\Scripts\python.exe -I src\clip_zeroshot.py --model ViT-L-14 --pretrained openai   # CLIP zero-shot items + rule table
.\.venv\Scripts\python.exe -I src\calibrate.py --runs "lastblock_brightrot_f1.0_s*" --tag main   # temperature scaling + abstention
.\.venv\Scripts\python.exe -I src\bootstrap.py; .\.venv\Scripts\python.exe -I src\bootstrap_ext.py  # 95% CIs
.\.venv\Scripts\python.exe -I src\analyze_extensions.py                           # results/extensions_summary.md + figures
.\.venv\Scripts\python.exe -I src\demo.py --engine twostage                       # demo v2: item type, calibrated confidence, abstain + hints
.\.venv\Scripts\python.exe -I src\demo.py --engine clip --clip_model ViT-L-14     # zero-shot engine
```

- `--target fine` trains a 10-way item head (materials with plastic split into rigid / film) and maps it to bins through
  the campus rule table in `common.py` (`FINE_TO_BIN`); `--backbone` selects DeiT-Tiny/Small, DINOv2-S or CLIP ViT-B/32.
- `demo.py` fits temperature scaling and an abstention threshold (80% coverage) on the validation split and shows
  "what to check" hints when the model is not confident.

## Labeling rules (U-M Ann Arbor campus)

| Bin | Items | Source |
| --- | --- | --- |
| recycling | paper, cardboard, cartons, metal cans/tins, rigid plastics #1-2 and #4-7 | OCSI plastics fact sheet; WWRA accepted list |
| compost | food scraps incl. meat, coffee grounds/filters, napkins, paper towels, BPI-certified ware, plant material | OCSI composting page |
| landfill | glass (not accepted on campus), plastic film/bags/foam, wrappers, snack bags, textiles, mixed items | OCSI FAQ; campus waste audit |

Batteries / e-waste / aerosols / construction debris are excluded (not bin items).

## Layout

- `src/` code, `data/reviewed/` candidate lists, review decisions and `splits.csv`, `data/review_sheets/` contact sheets
- `results/runs/<model>_<aug>_f<frac>_s<seed>/` metrics, predictions, trainable weights; `results/figures/` report figures
- `report/REPORT.md` final report (Markdown copy); the editable version with interactive charts is the Claude Doc
  https://claude.ai/artifact/8HjAba1VnSQBTZdG66xnuq (export to PDF/Word from there)

## Headline results (mean of 3 seeds)

| Configuration | RealWaste test acc | Proxy new-photo acc |
| --- | --- | --- |
| Linear probe, no aug, 100% | 0.760 | 0.716 |
| Last-block FT, aug, 100% (main model) | 0.803 | 0.768 |
