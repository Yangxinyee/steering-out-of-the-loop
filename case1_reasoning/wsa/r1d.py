"""The R1-Distill check (reading rule written before running): DeepSeek-R1-Distill-Qwen-1.5B on MATH-500,
one sample per problem, base T 0, its own mean-difference edit at T 0 (picked on practice), base at the model card's
T 0.6; CPU only. The pre-set reading of "releases answers, not knowledge":
  released   problems where base T 0 hit the cap and the edit finished: the edit's accuracy there
  native     problems base T 0 finished itself: its accuracy there
  card       the same released problems answered at T 0.6
with bootstrap intervals over problems; plus the three arms overall, paired.

    python -m wsa.r1d --runs results/runs/r1d --out results/r1d
"""
import argparse
import json
import os

import numpy as np

from wsa.csweep import boot


def arm(path):
    stem = os.path.splitext(path)[0]
    ok = {r["id"]: bool(r["correct"][0]) for r in json.load(open(stem + ".per_problem.json"))}
    cut = {}
    for l in open(path):
        r = json.loads(l)
        if r["k"] == 0:
            cut[r["id"]] = r["finish"] == "length"
    return ok, cut


def mean_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return {"mean": float(x.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": int(len(x))}


def main(a):
    picks = open(os.path.join(a.runs, "picks.txt")).read().strip().splitlines()
    fin = picks[-1].split(": ")[1].split(" ")[0]
    B = arm(os.path.join(a.runs, "test/base/r1d_t0_math500.jsonl"))
    E = arm(os.path.join(a.runs, f"test/edit_{fin}/r1d_t0_math500.jsonl"))
    C = arm(os.path.join(a.runs, "test/base/r1d_card_math500.jsonl"))
    ids = sorted(set(B[0]) & set(E[0]) & set(C[0]))
    v = lambda X, i: float(X[0][i])
    out = {"edit": fin, "n_problems": len(ids), "arms": {}, "paired": {}, "released": {}}
    for name, X in (("base T 0", B), ("edit T 0", E), ("base T 0.6 (card)", C)):
        out["arms"][name] = {"acc": float(np.mean([v(X, i) for i in ids])), "cut": float(np.mean([X[1][i] for i in ids]))}
    for label, X, Y in (("edit T 0 - base T 0", E, B), ("base T 0.6 - base T 0", C, B), ("edit T 0 - base T 0.6", E, C)):
        d = np.array([v(X, i) - v(Y, i) for i in ids])
        out["paired"][label] = {"d": float(d.mean()), "ci95": boot(d)}
    rel = [i for i in ids if B[1][i] and not E[1][i]]                 # base looped to the cap, the edit finished
    nat = [i for i in ids if not B[1][i]]                             # base finished on its own
    out["released"] = {
        "edit_on_released": mean_ci([v(E, i) for i in rel]),
        "base_on_its_finished": mean_ci([v(B, i) for i in nat]),
        "card_on_released": mean_ci([v(C, i) for i in rel]),
        "edit_minus_card_on_released": {"d": float(np.mean([v(E, i) - v(C, i) for i in rel])),
                                        "ci95": boot(np.array([v(E, i) - v(C, i) for i in rel]))},
        "base_cut_edit_cut": int(sum(1 for i in ids if B[1][i] and E[1][i])),
        "base_finished_edit_cut": int(sum(1 for i in ids if not B[1][i] and E[1][i])),
    }
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
