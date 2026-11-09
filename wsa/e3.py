"""E3: three cheap decoding baselines on Qwen3.5-0.8B, against part 1's base model and fixes on the
same test sets (MATH-500, GSM8K; 4 samples, think35 unless named); CPU only. Practice picks (sel200, one sample) as
written by the job; the no-repeat family did not run (its sampler is too slow on vLLM's V1 runner) and was
replaced by vLLM's repetition_penalty on practice only (it did not pass the gate for the test sets).

    python -m wsa.e3 --runs results/runs --out results/e3
"""
import argparse
import json
import os

import numpy as np

from wsa.c2b import load
from wsa.csweep import boot

SETS = ("math500", "gsm8k")
ARMS = {
    "base (think35)": "base/think35_{s}",
    "logit bias +8 on </think> (E3)": "e3/test/think35_tb8_{s}",
    "thinking off, 32k cap (E3)": "e3/test/nothink35_32k_{s}",
    "thinking off, 8k cap (part 1)": "base/nothink35_{s}",
    "budget forcing 8k": "base/think35_bf8192_{s}",
    "loop stop (forced answer)": "base/think35_loop_{s}",
    "T 1.0 (model card)": "base/card_think_{s}",
    "textbook edit (mean difference, strength 6)": "meandiff_top64_a6/think35_{s}",
    "authors' edit": "edit_authors/think35_{s}",
}
PRACTICE = ("nothink35_32k", "think35_tb2", "think35_tb4", "think35_tb8",
            "think35_rp105", "think35_rp11", "think35_rp12")      # repetition_penalty: the no-repeat family's substitute


def main(a):
    out = {"practice": {}, "test": {}}
    for p in PRACTICE:
        f = os.path.join(a.runs, "e3/practice", f"{p}_sel200.scores.json")
        if os.path.exists(f):
            s = json.load(open(f))
            out["practice"][p] = {"acc": s["acc"], "cut": s["cut_share"], "mean_tokens": s["mean_tokens"]}
    for s in SETS:
        base = load(a.runs, ARMS["base (think35)"].format(s=s))
        ids = sorted(base[0])
        b = np.array([base[0][i] for i in ids])
        out["test"][s] = {}
        for name, stem in ARMS.items():
            x = load(a.runs, stem.format(s=s))
            if x is None:
                continue
            acc = np.array([x[0][i] for i in ids])
            d = acc - b
            out["test"][s][name] = {"acc": float(acc.mean()), "cut": float(np.mean([x[1][i] for i in ids])),
                                    "mean_tokens": float(np.mean([x[2][i] for i in ids])),
                                    "vs_base": {"d": float(d.mean()), "ci95": boot(d)}}
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    for p, v in out["practice"].items():
        print(f"practice {p:16s} {v['acc'] * 100:5.1f}  cut {v['cut'] * 100:5.1f}  tok {v['mean_tokens']:7.0f}")
    for s, arms in out["test"].items():
        print(f"== {s}")
        for name, v in arms.items():
            print(f"  {name:45s} {v['acc'] * 100:5.1f}  cut {v['cut'] * 100:5.1f}  tok {v['mean_tokens']:7.0f}  "
                  f"vs base {v['vs_base']['d'] * 100:+5.1f} [{v['vs_base']['ci95'][0] * 100:+.1f}, {v['vs_base']['ci95'][1] * 100:+.1f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
