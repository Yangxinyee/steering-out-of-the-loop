"""The same-model reference that adds ability, scored on the coverage ruler (the distilled 0.8B).

The distilled 0.8B (wsa.sft) under the authors' protocol: all of MATH-500, GSM8K's always-looping problems.
1. Accuracy against the base model and the agent's edit on the problems it was run on (paired, bootstrap over problems).
2. On always-looping problems, the coverage of its correct answers (finished and right) by the unedited 0.8B's 32
   samples under the eight fixes, per answer, unbiased pass@k; matched to the edit's solved count by a random subset of
   the side that solves more (4,000 draws for the point; 4,000 bootstrap replicates over problems, each with its own
   draw), as wsa.coverage_match does for Qwen3.5-2B; the difference edit minus reference uses the same replicate.
   Also the problems it solves that no ruler sample solves.
0. (printed first in the file, computed with 2) Whether it adds ability at all: on never-looping problems, strict
   accuracy against the base; on always-looping problems, problems solved against the edit, each fix and Qwen3.5-2B,
   and problems beyond the ruler against the edit. A reference that does not solve clearly more says nothing here.
3. Released against own answers on MATH-500's paired problems (as wsa.rescue).

    python -m wsa.coverage_sft [--stem sft_ref/eval/think35_{}]        (writes results/coverage_sft.json)
"""
import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

from wsa.coverage import SETS
from wsa.coverage_ref import TWO_B
from wsa.rescue import boot_ci, load

RUNS = Path("results/runs")
KS = (1, 2, 4, 8, 16, 32)
N = 4000
FILES = {"MATH-500": "math500", "GSM8K": "gsm8k_always"}


def main(a):
    out = {}
    for ds in ("MATH-500", "GSM8K"):
        kind, tag, decs, edit_stem = SETS[ds]
        stem = a.stem.format(FILES[ds])
        if not (RUNS / f"{stem}.per_problem.json").exists():
            print(ds, "no run", stem)
            continue
        S = load(RUNS, stem, kind)
        D = {k: load(RUNS, v.format(tag), kind) for k, v in decs.items()}
        base = D.pop("base")
        E = load(RUNS, edit_stem.format(tag), kind)
        sp = sorted({p for p, _ in S})
        R = {"problems_run": len(sp), "stem": stem}
        # 1. accuracy (authors' scoring and strict), finished share, tokens; paired against base and edit
        toks = [json.loads(l)["n_tok"] for l in open(RUNS / f"{stem}.jsonl")]
        acc = lambda X, p, strict: np.mean([X[(p, k)][0] and (X[(p, k)][1] or not strict) for k in range(4)])
        R["accuracy"] = {nm: {"acc": float(np.mean([acc(X, p, False) for p in sp])),
                              "strict": float(np.mean([acc(X, p, True) for p in sp])),
                              "finished": float(np.mean([X[(p, k)][1] for p in sp for k in range(4)]))}
                         for nm, X in (("reference", S), ("base", base), ("agent's edit", E))}
        R["accuracy"]["reference"]["mean_tokens"] = float(np.mean(toks))
        for nm, X in (("base", base), ("agent's edit", E)):
            R[f"reference_minus_{nm}"] = list(boot_ci([acc(S, p, False) - acc(X, p, False) for p in sp]))
        # 2. coverage on always-looping problems
        always = [p for p in sp if all(not base[(p, k)][1] for k in range(4))]
        n_s = 4 * len(D)
        c = np.array([sum(D[d][(p, k)][0] and D[d][(p, k)][1] for d in D for k in range(4) if (p, k) in D[d])
                      for p in always])
        right = lambda X: np.array([sum(1 for k in range(4) if X[(p, k)][0] and X[(p, k)][1]) for p in always], float)
        re_, rs = right(E), right(S)
        # 0. does it add ability? (the reference counts only if it solves clearly more than the 0.8B)
        #    never-looping problems (all four base samples finish; MATH-500 only): strict accuracy against the base,
        #    paired, with Qwen3.5-2B for scale. Always-looping problems: problems solved (some sample finished and
        #    right) and strict accuracy per sample, for the reference, the edit, each fix and the 2B; problems no ruler
        #    sample solves, reference minus edit (paired bootstrap over problems).
        T = load(RUNS, TWO_B[ds], kind)
        strict = lambda X, p: np.mean([X[(p, k)][0] and X[(p, k)][1] for k in range(4)])
        never = [p for p in sp if all(base[(p, k)][1] for k in range(4))]
        ab = {"never_looping": len(never)}
        if never:
            ab["never_looping_strict"] = {nm: float(np.mean([strict(X, p) for p in never]))
                                          for nm, X in (("reference", S), ("base", base), ("Qwen3.5-2B", T))}
            ab["never_looping_minus_base"] = {nm: list(boot_ci([strict(X, p) - strict(base, p) for p in never]))
                                              for nm, X in (("reference", S), ("Qwen3.5-2B", T))}
        arms = {"reference": S, "agent's edit": E, "Qwen3.5-2B": T, **{f"fix: {d}": D[d] for d in D}}
        ab["always_looping"] = {nm: {"solved": int(sum(strict(X, p) > 0 for p in always)),
                                     "strict": float(np.mean([strict(X, p) for p in always]))} for nm, X in arms.items()}
        best = max(D, key=lambda d: ab["always_looping"][f"fix: {d}"]["solved"])      # the 0.8B's best decoding
        ab["always_strict_minus"] = {nm: list(boot_ci([strict(S, p) - strict(X, p) for p in always]))
                                     for nm, X in (("agent's edit", E), (f"fix: {best}", D[best]), ("Qwen3.5-2B", T))}
        ab["ruler_solved"] = int((c > 0).sum())
        beyond = lambda r: ((r > 0) & (c == 0)).astype(float)
        bd = boot_ci(beyond(rs) - beyond(re_))
        ab["beyond_ruler_minus_edit"] = [x * len(always) for x in bd]
        ab["beyond_ruler"] = {"reference": int(beyond(rs).sum()), "agent's edit": int(beyond(re_).sum()),
                              "Qwen3.5-2B": int(beyond(right(T)).sum())}
        R["ability"] = ab
        pk ={k: np.array([1 - comb(n_s - ci, k) / comb(n_s, k) for ci in c]) for k in KS}
        cover = lambda r, ix, w: {k: float((r[ix] * w * pk[k][ix]).sum() / max((r[ix] * w).sum(), 1e-12)) for k in KS}

        def pick(r_big, r_small, ix, rng):
            w = np.zeros(len(ix))
            pos = np.flatnonzero(r_big[ix] > 0)
            n = int((r_small[ix] > 0).sum())
            w[rng.choice(pos, min(n, len(pos)), replace=False)] = 1
            return w

        sub_ref = (rs > 0).sum() >= (re_ > 0).sum()      # subsample the side that solves more
        full = np.arange(len(always))
        one = np.ones(len(full))
        rng = np.random.default_rng(0)
        pt_e, pt_r = [], []
        for _ in range(N):
            if sub_ref:
                w = pick(rs, re_, full, rng)
                pt_e.append(cover(re_, full, one))
                pt_r.append(cover(rs, full, w))
            else:
                w = pick(re_, rs, full, rng)
                pt_e.append(cover(re_, full, w))
                pt_r.append(cover(rs, full, one))
        rng = np.random.default_rng(1)
        bs_r, bs_d = {k: [] for k in KS}, {k: [] for k in KS}
        for _ in range(N):
            ix = rng.integers(0, len(always), len(always))
            o = np.ones(len(ix))
            if sub_ref:
                ce, cr = cover(re_, ix, o), cover(rs, ix, pick(rs, re_, ix, rng))
            else:
                ce, cr = cover(re_, ix, pick(re_, rs, ix, rng)), cover(rs, ix, o)
            for k in KS:
                bs_r[k].append(cr[k])
                bs_d[k].append(ce[k] - cr[k])
        R["always_looping"] = len(always)
        R["solved"] = {"reference": int((rs > 0).sum()), "agent's edit": int((re_ > 0).sum()),
                       "subsampled": "reference" if sub_ref else "agent's edit"}
        R["reference_unmatched"] = {"coverage_k32": cover(rs, full, one)[32], "correct_answers": int(rs.sum()),
                                    "solved_beyond_ruler": int(((rs > 0) & (c == 0)).sum()),
                                    "edit_solved_beyond_ruler": int(((re_ > 0) & (c == 0)).sum())}
        R["reference"] = {"coverage": {k: [float(np.mean([p[k] for p in pt_r])), float(np.percentile(bs_r[k], 2.5)),
                                           float(np.percentile(bs_r[k], 97.5))] for k in KS}}
        R["edit_matched"] = {k: float(np.mean([p[k] for p in pt_e])) for k in KS}
        R["edit_minus_reference"] = {k: [R["edit_matched"][k] - R["reference"]["coverage"][k][0],
                                         float(np.percentile(bs_d[k], 2.5)), float(np.percentile(bs_d[k], 97.5))]
                                     for k in KS}
        # 3. released against own on the paired problems (MATH-500 only: GSM8K was run on always-looping problems)
        pair = [p for p in sp if any(base[(p, k)][1] for k in range(4)) and not all(base[(p, k)][1] for k in range(4))]
        if pair:
            d = []
            for p in pair:
                own = [base[(p, k)][0] for k in range(4) if base[(p, k)][1]]
                rel = [S[(p, k)][0] for k in range(4) if not base[(p, k)][1] and S[(p, k)][1]]
                if rel:
                    d.append(np.mean(rel) - np.mean(own))
            R["released_minus_own"] = {"problems": len(d), "diff": list(boot_ci(d))}
        out[ds] = R
        A = R["accuracy"]
        print(ds, f"run on {len(sp)}; accuracy ref {A['reference']['acc']*100:.1f} (strict {A['reference']['strict']*100:.1f},"
                  f" finished {A['reference']['finished']*100:.1f}, {A['reference']['mean_tokens']/1000:.1f}k tokens),"
                  f" base {A['base']['acc']*100:.1f}, edit {A["agent's edit"]['acc']*100:.1f}")
        print("   ref - edit", [round(x * 100, 1) for x in R["reference_minus_agent's edit"]],
              " solved", R["solved"], " unmatched", R["reference_unmatched"])
        cv, dv = R["reference"]["coverage"], R["edit_minus_reference"]
        print(f"   matched: reference k=32 {cv[32][0]*100:.1f} [{cv[32][1]*100:.1f}, {cv[32][2]*100:.1f}], k=4 {cv[4][0]*100:.1f};"
              f" edit k=32 {R['edit_matched'][32]*100:.1f}; edit - ref k=32 {dv[32][0]*100:+.1f} [{dv[32][1]*100:.1f}, {dv[32][2]*100:.1f}]"
              f" k=4 {dv[4][0]*100:+.1f} [{dv[4][1]*100:.1f}, {dv[4][2]*100:.1f}]")
        if "released_minus_own" in R:
            print("   released - own", R["released_minus_own"])
        ab = R["ability"]
        if "never_looping_strict" in ab:
            print(f"   ability, {ab['never_looping']} never-looping problems, strict:",
                  {k: round(v * 100, 1) for k, v in ab["never_looping_strict"].items()},
                  " minus base:", {k: [round(x * 100, 1) for x in v] for k, v in ab["never_looping_minus_base"].items()})
        print(f"   ability, {len(always)} always-looping problems (solved, strict %):",
              {k: (v["solved"], round(v["strict"] * 100, 1)) for k, v in ab["always_looping"].items()})
        print("   reference minus, strict:", {k: [round(x * 100, 1) for x in v] for k, v in ab["always_strict_minus"].items()},
              " ruler solves", ab["ruler_solved"], " beyond ruler", ab["beyond_ruler"], " ref - edit", [round(x, 1) for x in ab["beyond_ruler_minus_edit"]])
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="sft_ref/eval/think35_{}")
    ap.add_argument("--out", default="results/coverage_sft.json")
    main(ap.parse_args())
