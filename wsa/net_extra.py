"""Net accuracy of the extra answers an arm finishes beyond the base model, for the arms results/abstain.json does not
cover (the agent's edit, thinking off at the full budget; thinking off at 8k, value only), and each arm's strict accuracy on the answers the base model
already finished (what an arm costs there). CPU only, from the existing outputs.

    python -m wsa.net_extra        (writes results/net_extra.json)
"""
import json
from pathlib import Path

import numpy as np

from wsa.abstain import SETS, boot, counts, load

RUNS = Path("results/runs")


def main():
    out = {}
    for name in ("MATH-500", "GSM8K", "MedQA", "MMLU-Pro health"):
        kind, base_stem, arms = SETS[name]
        B = load(RUNS, base_stem, kind)
        probs = sorted({p for p, _ in B})
        Cb = counts(B, probs)
        kept = [k for k, v in B.items() if v[1]]
        R = {"net_beyond_base": {}, "on_base_finished": {"n": len(kept), "base": sum(B[k][0] for k in kept) / len(kept)}}
        for a in ("authors_edit", "nothink32k", "t1", "bf8k", "loop", "meandiff", "nothink"):
            if a not in arms:
                continue
            A = load(RUNS, arms[a], kind)
            X = counts(A, probs)

            def marg(i=None):
                x, y = (X, Cb) if i is None else (X[i], Cb[i])
                df = x[:, 0].sum() - y[:, 0].sum()
                return (x[:, 1].sum() - y[:, 1].sum()) / df if df > 0 else np.nan
            # thinking off at 8k: the value only, so the bootstrap stream (and every interval above) is unchanged
            R["net_beyond_base"][a] = {"value": float(marg()), "ci": None if a == "nothink" else boot(marg, len(probs))}
            kk = [k for k in kept if k in A]
            R["on_base_finished"][a] = sum(A[k][0] and A[k][1] for k in kk) / len(kk)
        out[name] = R
        print(name, {a: round(100 * v["value"], 1) for a, v in R["net_beyond_base"].items()},
              {a: round(100 * v, 1) for a, v in R["on_base_finished"].items() if a != "n"})
    Path("results/net_extra.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
