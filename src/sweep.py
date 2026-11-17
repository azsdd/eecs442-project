"""Small validation-set hyper-parameter sweep (seed 0, 100% data, no augmentation).

Selects learning rates by validation macro-F1 for each model type; results in results/sweep.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS
from train import get_parser, run

GRID = {
    "linear": [dict(lr_head=lr) for lr in (1e-3, 3e-3, 1e-2)],
    "lastblock": [dict(lr_head=1e-3, lr_block=lr) for lr in (5e-5, 1e-4, 3e-4)],
}


def main() -> None:
    out = []
    for model, settings in GRID.items():
        for s in settings:
            argv = ["--model", model, "--aug", "none", "--frac", "1.0", "--seed", "0", "--epochs", "40"]
            for k, v in s.items():
                argv += [f"--{k}", str(v)]
            tag = "_".join(f"{k}{v:g}" for k, v in s.items())
            argv += ["--name", f"sweep_{model}_{tag}"]
            args = get_parser().parse_args(argv)
            r = run(args)
            out.append({"model": model, **s, "best_epoch": r["best_epoch"],
                        "val_macro_f1": r["val"]["macro_f1"], "val_acc": r["val"]["accuracy"],
                        "train_time_s": r["train_time_s"]})
    (RESULTS / "sweep.json").write_text(json.dumps(out, indent=2))
    for model in GRID:
        best = max((o for o in out if o["model"] == model), key=lambda o: o["val_macro_f1"])
        print(f"BEST {model}: {best}")


if __name__ == "__main__":
    main()
