"""Same-problem comparison (released against own answers) for the fixes that wsa.rescue does not cover: thinking off at
the full budget, the end-of-thinking bias, the repetition penalty and the prompt (CPU, existing outputs only). Same
definitions, bootstrap and seed as wsa.rescue.

    python -m wsa.rescue_more        (writes results/rescue_more.json)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wsa.rescue import boot_ci, load

RUNS = Path("results/runs")
BASE = {"MATH-500": ("math", "base/think35_math500"), "GSM8K": ("math", "base/think35_gsm8k"),
        "MedQA": ("mc", "med_base/think35_medqa")}
ARMS = {"nothink32k": {"MATH-500": "e3/test/nothink35_32k_math500", "GSM8K": "e3/test/nothink35_32k_gsm8k"},
        "eot_bias": {"MATH-500": "e3/test/think35_tb8_math500", "GSM8K": "e3/test/think35_tb8_gsm8k"},
        "rep_penalty": {"MATH-500": "base/think35_rep11_math500", "GSM8K": "base/think35_rep11_gsm8k"},
        "prompt": {"MATH-500": "prompt/think35_pr2_math500", "GSM8K": "prompt/think35_pr2_gsm8k",
                   "MedQA": "prompt/think35_pr2_medqa"}}


def main():
    out = {}
    for arm, sets in ARMS.items():
        for ds, stem in sets.items():
            if not (RUNS / f"{stem}.per_problem.json").exists():
                print("missing", arm, ds, stem)
                continue
            kind, bstem = BASE[ds]
            base, fx = load(RUNS, bstem, kind), load(RUNS, stem, kind)
            keys = [k for k in base if k in fx]
            resc = [k for k in keys if not base[k][1] and fx[k][1]]
            per_p = defaultdict(lambda: ([], []))
            for k in resc:
                per_p[k[0]][0].append(fx[k][0])
            for k in keys:
                if base[k][1]:
                    per_p[k[0]][1].append(base[k][0])
            pairs = [(np.mean(r), np.mean(b)) for r, b in per_p.values() if r and b]
            d, lo, hi = boot_ci([x - y for x, y in pairs])
            out.setdefault(arm, {})[ds] = {
                "samples_paired": len(keys), "rescued": len(resc),
                "rescued_accuracy": float(np.mean([fx[k][0] for k in resc])) if resc else None,
                "within_problem": {"problems": len(pairs), "rescued_acc": float(np.mean([x for x, _ in pairs])) if pairs else None,
                                   "base_finished_acc": float(np.mean([y for _, y in pairs])) if pairs else None,
                                   "diff": d, "lo": lo, "hi": hi}}
            w = out[arm][ds]["within_problem"]
            print(f"{arm:12s} {ds:9s} n={w['problems']:4d} released {w['rescued_acc'] and w['rescued_acc']*100:.1f} own "
                  f"{w['base_finished_acc']*100:.1f} diff {d*100:+.1f} [{lo*100:.1f}, {hi*100:.1f}] rescued_acc "
                  f"{out[arm][ds]['rescued_accuracy']*100:.1f} ({len(resc)})")
    Path("results/rescue_more.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
