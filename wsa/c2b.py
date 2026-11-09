"""c2b: Qwen3.5-2B's main comparison on MATH-500, GSM8K and MedQA, 4 samples each, the arms of 0.8B's
main table; CPU only. Per set and arm: accuracy, share cut at the cap, mean tokens, and the paired difference from the
base model (think35) with a 95% bootstrap interval over problems (each problem's mean over its 4 samples).

    python -m wsa.c2b --runs results/runs/c2b --out results/c2b
"""
import argparse
import json
import os

import numpy as np

from wsa.csweep import boot

SETS = ("math500", "gsm8k", "medqa")
ARMS = {  # name -> file stem under runs/c2b
    "base (think35)": "base/think35_{s}",
    "textbook edit (layer 15, strength 6)": "edit_l15_top64_a6/think35_{s}",
    "T 1.0 (model card)": "base/card_think_{s}",
    "thinking off, 32k cap": "base/nothink35_32k_{s}",
    "thinking off, 8k cap": "base/nothink35_{s}",
    "budget forcing 8k": "base/think35_bf8192_{s}",
    "loop stop (forced answer)": "base/think35_loop_{s}",
    "prompt only": "base/think35_pr2_{s}",
}


def load(runs, stem):
    pp = os.path.join(runs, stem + ".per_problem.json")
    if not os.path.exists(pp):
        return None
    acc = {r["id"]: float(np.mean(r["correct"])) for r in json.load(open(pp))}
    cut, tok = {}, {}
    for l in open(os.path.join(runs, stem + ".jsonl")):
        r = json.loads(l)
        cut.setdefault(r["id"], []).append(r["finish"] == "length")
        tok.setdefault(r["id"], []).append(r["n_tok"])
    return acc, {i: float(np.mean(v)) for i, v in cut.items()}, {i: float(np.mean(v)) for i, v in tok.items()}


def main(a):
    out = {}
    for s in SETS:
        base = load(a.runs, ARMS["base (think35)"].format(s=s))
        ids = sorted(base[0])
        b = np.array([base[0][i] for i in ids])
        out[s] = {}
        for name, stem in ARMS.items():
            x = load(a.runs, stem.format(s=s))
            if x is None:
                continue
            acc = np.array([x[0][i] for i in ids])
            d = acc - b
            out[s][name] = {"acc": float(acc.mean()), "cut": float(np.mean([x[1][i] for i in ids])),
                            "mean_tokens": float(np.mean([x[2][i] for i in ids])),
                            "vs_base": {"d": float(d.mean()), "ci95": boot(d)}}
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    for s, arms in out.items():
        print(f"== {s}")
        for name, v in arms.items():
            print(f"  {name:38s} {v['acc'] * 100:5.1f}  cut {v['cut'] * 100:5.1f}  tok {v['mean_tokens']:7.0f}  "
                  f"vs base {v['vs_base']['d'] * 100:+5.1f} [{v['vs_base']['ci95'][0] * 100:+.1f}, {v['vs_base']['ci95'][1] * 100:+.1f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
