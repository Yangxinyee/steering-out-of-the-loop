"""MMLU-Pro math, the test with room above the model's own answers (CPU,
existing samples).

1. Reach on the questions the unedited model never finishes under the protocol (all four samples cut). The share of the
   agent's edit's correct answers whose question the unedited model also answers correctly (finished and right) under
   the five decoding fixes run on this set (20 samples per question), by unbiased pass@k, read against:
   - guessing: each finished ruler sample replaced by a uniform guess among the question's options (the ruler's finish
     count kept), the coverage multiple choice gives with no knowledge;
   - random questions: the ruler's solve rate over all of these questions, the coverage of a random set of them;
   - each fix's own correct answers, scored against the other four fixes (16 samples), and the edit against the same four.
   Chance-corrected coverage: (observed - guessing) / (1 - guessing).
2. Same question, same seed: samples cut under the unedited model and finished under both the edit and loop stop (loop
   stop ends the unedited model's own thinking and has it answer). The edit's accuracy minus loop stop's, per question
   then averaged, and the share of identical answers; on all such questions and on the never-finished ones.
Intervals: 2,000 bootstrap replicates over questions.

    python -m wsa.headroom_reach        (writes results/headroom_reach.json)
    python -m wsa.headroom_reach --target sft_ref3/eval/think35_mmlupro_math --out results/headroom_reach_sft.json
        (the same tests on another model's answers, e.g. the distilled 0.8B, a known install)
"""
import argparse
import json
import re
from math import comb
from pathlib import Path

import numpy as np

from wsa.rescue import load

RUNS = Path("results/runs")
FIXES = {"loop stop": "think35_loop", "budget forcing": "think35_bf8192", "thinking off, 8k": "nothink35",
         "thinking off, full budget": "nothink35_32k", "T = 1.0": "card_think"}
N_BOOT = 2000


def n_options(prompt):
    block = prompt.split("Options:", 1)[1].split("\nAnswer:", 1)[0]
    return len(re.findall(r"(?m)^\s*([A-J])\. ", block))


def passk(n, c, k):                      # unbiased pass@k with c correct of n
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def guessk(n, f, m, k):                  # pass@k if the f finished of n samples were uniform guesses among m options
    q = 1 - 1 / m
    return 1 - sum(comb(f, h) * comb(n - f, k - h) / comb(n, k) * q ** h for h in range(max(0, k - (n - f)), min(f, k) + 1))


def ci(f, idx_sets):
    v = np.array([f(ix) for ix in idx_sets])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main(a):
    m_opt = {r["id"]: n_options(r["prompt"]) for r in map(json.loads, open("results/data/mmlupro_math.jsonl"))}
    B = load(RUNS, "headroom/base/think35_mmlupro_math", "mc")
    E = load(RUNS, a.target, "mc")
    F = {nm: load(RUNS, f"headroom/fix/{st}_mmlupro_math", "mc") for nm, st in FIXES.items()}
    qs = sorted({p for p, _ in B})
    right = lambda X, p: sum(1 for k in range(4) if X.get((p, k), (0, 0))[0] and X.get((p, k), (0, 0))[1])  # noqa: E731
    fin = lambda X, p: sum(1 for k in range(4) if X.get((p, k), (0, 0))[1])                                # noqa: E731
    never = [p for p in qs if fin(B, p) == 0]
    n_s = 4 * len(F)
    c = np.array([sum(right(X, p) for X in F.values()) for p in never], float)       # ruler's correct samples
    f = np.array([sum(fin(X, p) for X in F.values()) for p in never], float)         # ruler's finished samples
    m = np.array([m_opt[p] for p in never], float)
    r = np.array([right(E, p) for p in never], float)                                 # the edit's correct answers
    ks = (1, 2, 4, 8, 16, 20)
    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(never), len(never)) for _ in range(N_BOOT)]
    P = {k: np.array([passk(n_s, int(ci_), k) for ci_ in c]) for k in ks}
    G = {k: np.array([guessk(n_s, int(fi), int(mi), k) for fi, mi in zip(f, m)]) for k in ks}

    def cov(w, ix=None, table=P, k=20):
        ix = np.arange(len(never)) if ix is None else ix
        return float((w[ix] * table[k][ix]).sum() / max(w[ix].sum(), 1e-12))

    out = {"questions": len(qs), "never_finished": len(never), "ruler": list(FIXES), "ruler_samples": n_s,
           "options_mean": float(m.mean()), "edit_correct_answers": int(r.sum()), "edit_solved": int((r > 0).sum()),
           "ruler_solved": int((c > 0).sum()), "edit_solved_beyond_ruler": int(((r > 0) & (c == 0)).sum()),
           "ruler_finished_share": float(f.sum() / (n_s * len(never)))}
    one = np.ones(len(never))
    res = {}
    for k in ks:
        obs, gue, rnd = cov(r, k=k), cov(r, table=G, k=k), cov(one, k=k)
        res[str(k)] = {"edit": [obs, *ci(lambda ix: cov(r, ix, k=k), boot)],
                       "guessing": [gue, *ci(lambda ix: cov(r, ix, G, k), boot)],
                       "random_questions": [rnd, *ci(lambda ix: cov(one, ix, k=k), boot)],
                       "chance_corrected": [(obs - gue) / (1 - gue),
                                            *ci(lambda ix: (cov(r, ix, k=k) - cov(r, ix, G, k)) / (1 - cov(r, ix, G, k)), boot)],
                       "edit_minus_random": [obs - rnd, *ci(lambda ix: cov(r, ix, k=k) - cov(one, ix, k=k), boot)]}
    out["coverage"] = res
    # each fix's own correct answers against the other four fixes (16 samples), and the edit against the same four
    loo = {}
    for nm, X in F.items():
        own = np.array([right(X, p) for p in never], float)
        rest = c - own
        n = n_s - 4
        Pk = {16: np.array([passk(n, int(x), 16) for x in rest])}
        f_rest = f - np.array([fin(X, p) for p in never], float)
        Gk = {16: np.array([guessk(n, int(fi), int(mi), 16) for fi, mi in zip(f_rest, m)])}
        o, og = cov(own, table=Pk, k=16), cov(own, table=Gk, k=16)
        e, eg = cov(r, table=Pk, k=16), cov(r, table=Gk, k=16)
        loo[nm] = {"fix_correct_answers": int(own.sum()), "fix": o, "fix_guessing": og,
                   "fix_chance_corrected": (o - og) / (1 - og) if og < 1 else None,
                   "edit": e, "edit_guessing": eg, "edit_chance_corrected": (e - eg) / (1 - eg) if eg < 1 else None}
    out["leave_one_fix_out"] = loo
    # 2. released answers against loop stop's, same question and seed
    L = F["loop stop"]
    both = {}
    for p in qs:
        ks_ = [k for k in range(4) if (p, k) in B and not B[(p, k)][1] and E.get((p, k), (0, 0))[1] and L.get((p, k), (0, 0))[1]]
        if ks_:
            both[p] = ks_
    sets = {"all": sorted(both), "never_finished": [p for p in sorted(both) if p in set(never)]}
    pair = {}
    for nm, ps in sets.items():
        de = np.array([np.mean([E[(p, k)][0] for k in both[p]]) for p in ps], float)
        dl = np.array([np.mean([L[(p, k)][0] for k in both[p]]) for p in ps], float)
        ag = np.array([np.mean([E[(p, k)][2] == L[(p, k)][2] for k in both[p]]) for p in ps], float)
        bs = [rng.integers(0, len(ps), len(ps)) for _ in range(N_BOOT)]
        pair[nm] = {"questions": len(ps), "samples": int(sum(len(both[p]) for p in ps)),
                    "edit_acc": float(de.mean()), "loop_stop_acc": float(dl.mean()),
                    "edit_minus_loop_stop": [float((de - dl).mean()), *ci(lambda ix: (de[ix] - dl[ix]).mean(), bs)],
                    "same_answer": [float(ag.mean()), *ci(lambda ix: ag[ix].mean(), bs)]}
    out["released_vs_loop_stop"] = pair
    out["target"] = a.target
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("coverage", "leave_one_fix_out")}, indent=1))
    for k in ("4", "20"):
        print(k, {kk: [round(x * 100, 1) for x in v] for kk, v in res[k].items()})
    for nm, v in loo.items():
        print(nm, {kk: (round(x * 100, 1) if isinstance(x, float) else x) for kk, x in v.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="headroom/edit/think35_mmlupro_math", help="the edited (or other) model's run")
    ap.add_argument("--out", default="results/headroom_reach.json")
    main(ap.parse_args())
