"""pass@k coverage of the agent's edit's correct answers on always-looping problems (CPU, existing outputs).

A problem is always-looping when all four base-model samples under the authors' protocol are cut at the budget. On those
problems, an answer of the edit counts when it is finished and correct (strict). It is covered when the unedited model,
under any of the paper's decodings and any of their samples, finishes the same problem correctly at least once. Per
answer: the share of the edit's correct answers whose problem is covered; per problem: the share of problems the edit
solves at least once that are covered. 95% intervals by bootstrap over problems (ratio of sums for the per-answer share).

    python -m wsa.coverage        (writes results/coverage.json)
"""
import json
from pathlib import Path

import numpy as np

from wsa.rescue import load

RUNS = Path("results/runs")
SETS = {
    "MATH-500": ("math", "math500", {
        "base": "base/think35_{}", "t1": "base/card_think_{}", "bf8k": "base/think35_bf8192_{}",
        "loop": "base/think35_loop_{}", "eot_bias": "e3/test/think35_tb8_{}", "rep_penalty": "base/think35_rep11_{}",
        "prompt": "prompt/think35_pr2_{}", "nothink8k": "base/nothink35_{}", "nothink32k": "e3/test/nothink35_32k_{}"},
        "edit_authors/think35_{}"),
    "GSM8K": ("math", "gsm8k", None, "edit_authors/think35_{}"),
    "MedQA": ("mc", "medqa", {
        "base": "med_base/think35_{}", "t1": "med_base/card_think_{}", "bf8k": "med_base/think35_bf8192_{}",
        "loop": "med_base/think35_loop_{}", "prompt": "prompt/think35_pr2_{}", "nothink8k": "med_base/nothink35_{}"},
        "med_edit/think35_{}"),
}
SETS["GSM8K"] = ("math", "gsm8k", SETS["MATH-500"][2], "edit_authors/think35_{}")
THINK_ON = ("base", "t1", "bf8k", "loop", "eot_bias", "rep_penalty", "prompt")
THINK_OFF = ("nothink8k", "nothink32k")


def boot(num, den, n=10000, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(seed).integers(0, len(num), (n, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(num.sum() / den.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    out = {}
    for name, (kind, tag, decs, edit_stem) in SETS.items():
        D = {k: load(RUNS, v.format(tag), kind) for k, v in decs.items()}
        E = load(RUNS, edit_stem.format(tag), kind)
        base = D["base"]
        probs = sorted({p for p, _ in base})
        always = [p for p in probs if all(not base[(p, k)][1] for k in range(4) if (p, k) in base)]
        released = [key for key in E if key in base and not base[key][1] and E[key][1]]
        rel_always = sum(1 for p, _ in released if p in set(always))

        def solved(dec_keys, p):          # some sample of these decodings finishes p correctly
            return any(D[d][(p, k)][0] and D[d][(p, k)][1] for d in dec_keys for k in range(4) if (p, k) in D[d])

        right = {p: sum(1 for k in range(4) if (p, k) in E and E[(p, k)][0] and E[(p, k)][1]) for p in always}
        sa = set(always)
        per_sample = {d: float(np.mean([v[0] and v[1] for key, v in X.items() if key[0] in sa]))
                      for d, X in list(D.items()) + [("authors_edit", E)]}
        R = {"always_looping_problems": len(always), "problems": len(probs),
             "strict_accuracy_per_sample_on_always_looping": per_sample,
             "released_answers": len(released), "released_on_always_looping": rel_always,
             "edit_correct_answers_on_always_looping": int(sum(right.values())),
             "edit_solved_problems": int(sum(1 for v in right.values() if v)), "coverage": {}}
        groups = {"all decodings": tuple(D), "thinking on": tuple(d for d in THINK_ON if d in D),
                  "thinking off": tuple(d for d in THINK_OFF if d in D)}
        groups.update({d: (d,) for d in D})
        for g, keys in groups.items():
            cov = {p: solved(keys, p) for p in always}
            num_a = [right[p] * cov[p] for p in always]
            den_a = [right[p] for p in always]
            sp = [p for p in always if right[p]]
            pa, pl, ph = boot(num_a, den_a)
            pp, ql, qh = boot([cov[p] for p in sp], [1] * len(sp))
            R["coverage"][g] = {"decodings": list(keys), "samples_per_problem": 4 * len(keys),
                                "per_answer": [pa, pl, ph], "per_problem": [pp, ql, qh],
                                "base_solves_always_looping": int(sum(cov.values())),
                                "base_solves_but_edit_never": int(sum(1 for p in always if cov[p] and not right[p]))}
        # multiple choice: coverage against chance (a ruler of uniform guesses among the options, with the same number
        # of finished samples per problem), chance-corrected as (coverage - guessing) / (1 - guessing)
        if kind == "mc":
            from wsa.headroom_reach import guessk, n_options
            n_opt = {r["id"]: n_options(r["prompt"]) for r in map(json.loads, open(f"results/data/{tag}.jsonl"))}
            keys = tuple(D)
            fin = {p: sum(1 for d in keys for k in range(4) if (p, k) in D[d] and D[d][(p, k)][1]) for p in always}
            n_s = 4 * len(keys)
            g = {p: guessk(n_s, fin[p], n_opt[p], n_s) for p in always}
            cv = {p: solved(keys, p) for p in always}
            w = np.array([right[p] for p in always], float)
            cvv, gg = np.array([cv[p] for p in always], float), np.array([g[p] for p in always])
            idx = np.random.default_rng(0).integers(0, len(always), (10000, len(always)))
            obs = (w * cvv).sum() / w.sum(); gue = (w * gg).sum() / w.sum()
            o_b = (w[idx] * cvv[idx]).sum(1) / w[idx].sum(1); g_b = (w[idx] * gg[idx]).sum(1) / w[idx].sum(1)
            cc = (o_b - g_b) / (1 - g_b)
            R["chance_corrected_all_decodings"] = {"coverage": float(obs), "guessing": float(gue),
                                                   "chance_corrected": [float((obs - gue) / (1 - gue)),
                                                                        float(np.percentile(cc, 2.5)), float(np.percentile(cc, 97.5))]}
            print("   chance-corrected:", R["chance_corrected_all_decodings"])
        # solved problems with four samples: thinking off at the full budget against the edit (paired over problems)
        if "nothink32k" in D:
            a = np.array([solved(("nothink32k",), p) for p in always], float)
            e = np.array([right[p] > 0 for p in always], float)
            idx = np.random.default_rng(0).integers(0, len(always), (10000, len(always)))
            bs = (a[idx] - e[idx]).sum(1)
            R["solved_k4_thinking_off_minus_edit"] = {"thinking_off": int(a.sum()), "edit": int(e.sum()),
                                                      "diff": int(a.sum() - e.sum()),
                                                      "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            print("   solved with 4 samples, thinking off vs edit:", R["solved_k4_thinking_off_minus_edit"])
        out[name] = R
        c = R["coverage"]
        print(name, f"always-looping {len(always)}/{len(probs)}; released {rel_always}/{len(released)} on them;",
              f"edit right {R['edit_correct_answers_on_always_looping']} answers on {R['edit_solved_problems']} problems")
        for g in ("all decodings", "thinking on", "thinking off"):
            if g in c and c[g]["decodings"]:
                a, p_ = c[g]["per_answer"], c[g]["per_problem"]
                print(f"   {g:14s} k={c[g]['samples_per_problem']:2d}: per answer {a[0]*100:.1f} [{a[1]*100:.1f}, {a[2]*100:.1f}]"
                      f"  per problem {p_[0]*100:.1f} [{p_[1]*100:.1f}, {p_[2]*100:.1f}]  base-only {c[g]['base_solves_but_edit_never']}")
    Path("results/coverage.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
