"""Answers finished and strict accuracy of every setting on MATH-500 (4 samples per problem; CPU, existing outputs), for
Figure 2 (left): the settings wsa.abstain does not cover (end-of-thinking bias, repetition penalty, prompt) beside
those it does, from the same per-sample records (finished: not cut at the budget; strict: finished and right).

    python -m wsa.completion_points        (writes results/completion_points.json)
"""
import json
from pathlib import Path

import numpy as np

from wsa.coverage import SETS
from wsa.rescue import load

RUNS = Path("results/runs")


def main():
    kind, tag, decs, edit_stem = SETS["MATH-500"]
    arms = dict(decs, authors_edit=edit_stem)
    out = {}
    for a, stem in arms.items():
        X = load(RUNS, stem.format(tag), kind)
        out[a] = {"finished": float(np.mean([f for _, f, _ in X.values()])),
                  "strict": float(np.mean([c and f for c, f, _ in X.values()])), "samples": len(X)}
        print(f"{a:12s} finished {out[a]['finished']*100:5.1f}  strict {out[a]['strict']*100:5.1f}")
    Path("results/completion_points.json").write_text(json.dumps({"MATH-500": out}, indent=1))


if __name__ == "__main__":
    main()
