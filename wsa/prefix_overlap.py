"""Shared seeds and shared prefixes between the base model and each fix (CPU).

Every arm draws sample k of problem p with the same seed (wsa.gen.seed_of), so two arms produce the same tokens until
their logits differ enough to change one; budget forcing and loop stop continue the base model's own run by
construction (wsa.force). For each released pair (the base model cut sample (p, k) at the budget and the fix finished
it), we measure the prefix the two outputs share, in tokens of the 0.8B's tokenizer, and compare it with where the
base model's loop starts (wsa.direction.onset: the first 512-token step at which the loop detector fires).

Second, R1's check that does not condition on release: on the problems the base model sometimes finishes and
sometimes not (the same-problem pairing set), the accuracy of all of a fix's finished samples against the model's own.

    python -m wsa.prefix_overlap --tok ckpt/base        (writes results/prefix_overlap.json)
"""
import argparse
import json
from os.path import commonprefix
from pathlib import Path

import numpy as np

from wsa.direction import onset
from wsa.rescue import boot_ci, load

RUNS = Path("results/runs")
ARMS = {"MATH-500": {"agent's edit": "edit_authors/think35_math500", "textbook edit": "meandiff_top64_a6/think35_math500",
                     "T = 1.0": "base/card_think_math500", "budget forcing": "base/think35_bf8192_math500",
                     "loop stop": "base/think35_loop_math500", "end-of-thinking bias": "e3/test/think35_tb8_math500",
                     "repetition penalty": "base/think35_rep11_math500", "prompt": "prompt/think35_pr2_math500",
                     "thinking off, 8k": "base/nothink35_math500", "thinking off, full": "e3/test/nothink35_32k_math500"},
        "GSM8K": {"agent's edit": "edit_authors/think35_gsm8k", "textbook edit": "meandiff_top64_a6/think35_gsm8k",
                  "T = 1.0": "base/card_think_gsm8k", "budget forcing": "base/think35_bf8192_gsm8k",
                  "loop stop": "base/think35_loop_gsm8k", "end-of-thinking bias": "e3/test/think35_tb8_gsm8k",
                  "repetition penalty": "base/think35_rep11_gsm8k", "prompt": "prompt/think35_pr2_gsm8k",
                  "thinking off, 8k": "base/nothink35_gsm8k", "thinking off, full": "e3/test/nothink35_32k_gsm8k"}}
BASE = {"MATH-500": "base/think35_math500", "GSM8K": "base/think35_gsm8k"}


def texts(stem):
    return {(r["id"], r["k"]): r["text"] for r in (json.loads(l) for l in open(RUNS / f"{stem}.jsonl"))}


def q(x):
    return [float(np.percentile(x, v)) for v in (25, 50, 75)] if len(x) else None


def main(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    out = {}
    for ds, arms in ARMS.items():
        base, bt = load(RUNS, BASE[ds], "math"), texts(BASE[ds])
        cut = [key for key, v in base.items() if not v[1]]
        ons = {}
        for key in cut:                                       # loop onset of each cut base sample, in tokens
            enc = tok(bt[key], add_special_tokens=False, return_offsets_mapping=True)
            ons[key] = (onset(bt[key], enc["offset_mapping"], len(enc["input_ids"])), len(enc["input_ids"]))
        probs = sorted({p for p, _ in base})
        pair = [p for p in probs if any(base[(p, k)][1] for k in range(4)) and not all(base[(p, k)][1] for k in range(4))]
        own = {p: [base[(p, k)][0] for k in range(4) if base[(p, k)][1]] for p in pair}
        R = {}
        for name, stem in arms.items():
            X, xt = load(RUNS, stem, "math"), texts(stem)
            rel = [key for key in cut if key in X and X[key][1]]
            shared, beyond, rel_share = [], 0, []
            for key in rel:
                n = ntok(commonprefix([bt[key], xt[key]]))
                shared.append(n)
                t0, nb = ons[key]
                rel_share.append(n / nb)
                beyond += t0 is not None and n >= t0
            with_onset = sum(1 for key in rel if ons[key][0] is not None)
            # all finished samples of the fix against the model's own, same problems (no conditioning on release)
            d = []
            for p in pair:
                fin = [X[(p, k)][0] for k in range(4) if (p, k) in X and X[(p, k)][1]]
                if fin:
                    d.append(np.mean(fin) - np.mean(own[p]))
            m, lo, hi = boot_ci(d)
            R[name] = {"released": len(rel), "shared_prefix_tokens_q": q(shared),
                       "shared_share_of_base_q": q(rel_share),
                       "share_identical_first_100_tokens": float(np.mean([s >= 100 for s in shared])) if shared else None,
                       "released_with_onset": with_onset, "shared_past_loop_onset": int(beyond),
                       "all_finished_minus_own": {"problems": len(d), "diff": [m, lo, hi]}}
            r = R[name]
            print(f"{ds:8s} {name:22s} released {len(rel):5d}  shared tokens q25/50/75 "
                  f"{[round(x) for x in r['shared_prefix_tokens_q']] if shared else None}  "
                  f"past onset {beyond}/{with_onset}  all finished - own {m*100:+.1f} [{lo*100:.1f}, {hi*100:.1f}] on {len(d)}")
        on = [t for t, _ in ons.values() if t is not None]
        out[ds] = {"cut_base_samples": len(cut), "loop_onset_tokens_q": q(on), "arms": R}
        print(f"   base loop onset q25/50/75 {q(on)} ({len(on)} of {len(cut)} cut samples have an onset)")
    Path("results/prefix_overlap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    main(ap.parse_args())
