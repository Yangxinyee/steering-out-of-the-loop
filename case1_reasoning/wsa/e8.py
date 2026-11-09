"""E8: Qwen3.5-2B's own mean-difference edit at strength 3, 6, 9, layer 15 alone or layers 15 + 19, top-64 columns, on the
200 practice problems (sel200, one sample, think35), against base 2B; CPU only. Also: of the problems where base 2B hit
the cap, how many each edit gets right (the edit can only help there), and how many it breaks elsewhere.

    python -m wsa.e8 --runs results/runs --rows results/data/sel200.jsonl --out results/e8
"""
import argparse
import json

import numpy as np

from wsa.csweep import boot, load

CELLS = ["l15_top64_a3", "l15_top64_a6", "l15_top64_a9", "l15l19_top64_a3", "l15l19_top64_a6", "l15l19_top64_a9"]


def main(a):
    ids = [json.loads(l)["id"] for l in open(a.rows)]
    base = load(a.runs, "l2b/base", "think35_sel200", ids)
    cut0 = base["cut"] > 0
    out = {"n_problems": len(ids), "base": {"acc": float(base["acc"].mean()), "cut": float(base["cut"].mean()),
                                            "mean_tokens": float(base["tok"].mean()), "n_cut": int(cut0.sum())}, "cells": {}}
    for c in CELLS:
        x = load(a.runs, f"l2b/{c}", "think35_sel200", ids)
        if x is None:
            continue
        d = x["acc"] - base["acc"]
        out["cells"][c] = {"acc": float(x["acc"].mean()), "cut": float(x["cut"].mean()), "mean_tokens": float(x["tok"].mean()),
                           "vs_base": {"d": float(d.mean()), "ci95": boot(d)},
                           "right_on_base_cut": int(x["acc"][cut0].sum()), "base_right_on_base_cut": int(base["acc"][cut0].sum()),
                           "rescued": int(((x["acc"] > 0) & (base["acc"] == 0)).sum()),
                           "broken": int(((x["acc"] == 0) & (base["acc"] > 0)).sum())}
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
