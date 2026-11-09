"""E1: "a loop is an abstention". Existing outputs only, CPU only.

(a) Each fix as a point on the base model's risk-coverage plane: coverage = share of samples finished (a cut sample
    abstains), risk = share of finished answers that are wrong. Each arm's own curve ranks its finished answers by
    agreement (the share of the problem's samples with the same answer); AURC is the mean risk over that curve.
    Arms are compared at equal coverage; the answers the agent's edit releases beyond a decoding fix are scored by their
    marginal accuracy (extra correct / extra finished, paired bootstrap over problems); and each decoding fix's share
    of the edit's gain is given with an interval, on all four benchmarks.
(b) MATH-500 loop rate by difficulty level (1-5); where loops start and how long their period is, math against
    MedQA; how many of the base model's loops a temperature of 1.0 breaks.
(c) Within a problem: the base model's own finished samples against what a fix rescues (wsa.rescue); and whether the
    number of base samples cut on a problem predicts that a fix's answer is wrong (AUROC), the basis of "fix, then flag".
Also the ratios a reviewer computed from summaries, checked per sample.

    python -m wsa.abstain --runs results/runs --levels <MATH-500 test.jsonl with level> --tok ckpt/base \
        --out results/abstain
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wsa.rescue import SETS as RESCUE_SETS, load

# thinking off with the full 32k budget is the strongest decoding baseline (results/e3.json); it joins
# the math sets here, beside part 1's 8k thinking off ("nothink"), which stays as the weaker control
SETS = {k: (v[0], v[1], dict(v[2]), *v[3:]) for k, v in RESCUE_SETS.items()}
for _name, _stem in (("MATH-500", "math500"), ("GSM8K", "gsm8k")):
    SETS[_name][2]["nothink32k"] = f"e3/test/nothink35_32k_{_stem}"

DECODE = ("t1", "bf8k", "loop", "nothink")
NAMES = {"base": "base", "authors_edit": "agent's edit", "t1": "T=1.0", "bf8k": "budget forcing 8k", "loop": "loop stop",
         "meandiff": "mean difference (textbook direction)", "nothink": "thinking off", "nothink32k": "thinking off (full budget, 32k)"}
PICKED_ON_PRACTICE = {"MATH-500": "nothink", "GSM8K": "nothink"}     # chosen on the practice set
RNG = np.random.default_rng(0)
NB = 4000


def by_problem(arm):
    P = defaultdict(list)
    for (p, k), v in arm.items():
        P[p].append(v)
    return P


def agreement(arm):
    P = by_problem(arm)
    votes = {p: Counter(a for _, _, a in v if a is not None) for p, v in P.items()}
    n = {p: len(v) for p, v in P.items()}
    return {key: (votes[key[0]][a] / n[key[0]] if a is not None else 0.0) for key, (_, _, a) in arm.items()}


def curve(arm):
    """risk-coverage points of the arm's finished answers ranked by agreement: [(coverage, risk, threshold)]"""
    conf = agreement(arm)
    N = len(arm)
    pts = []
    for t in sorted({c for c in conf.values() if c > 0}, reverse=True):
        keep = [k for k, c in conf.items() if c >= t]
        right = sum(arm[k][0] and arm[k][1] for k in keep)
        pts.append((len(keep) / N, 1 - right / len(keep), t))
    return pts


def risk_at(pts, c):
    """risk at coverage c on the curve (linear between points; from the first point back to coverage 0 flat)"""
    if not pts or c > pts[-1][0] + 1e-12:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if c <= xs[0]:
        return ys[0]
    return float(np.interp(c, xs, ys))


def aurc(pts):
    if not pts:
        return None
    xs, ys = [0.0] + [p[0] for p in pts], [pts[0][1]] + [p[1] for p in pts]
    area = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1))
    return float(area / xs[-1])


def counts(arm, probs):
    """per problem: (finished, strict correct, official correct, samples)"""
    P = by_problem(arm)
    return np.array([[sum(f for _, f, _ in P[p]), sum(c and f for c, f, _ in P[p]), sum(c for c, _, _ in P[p]), len(P[p])]
                     for p in probs], float)


def boot(fn, n_probs):
    idx = RNG.integers(0, n_probs, (NB, n_probs))
    vals = np.array([fn(i) for i in idx])
    vals = vals[np.isfinite(vals)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (None, None)


def auroc(score, wrong):
    """P(score of a wrong answer > score of a right one), ties half"""
    s, w = np.array(score, float), np.array(wrong, bool)
    if w.all() or not w.any():
        return None
    levels = np.unique(s)
    hw = np.array([(s[w] == v).sum() for v in levels], float)   # wrong answers per score
    hr = np.array([(s[~w] == v).sum() for v in levels], float)  # right answers per score
    below = np.concatenate([[0], np.cumsum(hr)[:-1]])            # right answers with a lower score
    return float((hw * (below + hr / 2)).sum() / (hw.sum() * hr.sum()))


def part_a(runs):
    out = {}
    for name in ("MATH-500", "GSM8K", "MedQA", "MMLU-Pro health"):
        kind, base_stem, arms = SETS[name]
        A = {"base": load(runs, base_stem, kind)}
        for a, stem in arms.items():
            A[a] = load(runs, stem, kind)
        probs = sorted({p for p, _ in A["base"]})
        C = {a: counts(v, probs) for a, v in A.items()}
        R = {"arms": {}, "pairs": {}, "recovered": {}}
        for a, arm in A.items():
            c = C[a]
            fin, strict, off, n = c[:, 0].sum(), c[:, 1].sum(), c[:, 2].sum(), c[:, 3].sum()
            pts = curve(arm)
            R["arms"][a] = {"coverage": fin / n, "finished_accuracy": strict / fin if fin else None, "accuracy_strict": strict / n,
                            "accuracy_official": off / n, "official_over_coverage": (off / n) / (fin / n) if fin else None,
                            "curve": [{"coverage": x, "risk": y, "min_agreement": t} for x, y, t in pts], "aurc": aurc(pts)}
        e = C["authors_edit"]
        for g in [x for x in A if x not in ("base", "authors_edit")]:
            G = C[g]
            # marginal accuracy of what the edit releases beyond g, and of what g releases beyond the base
            def marg(X, Y, i=None):
                X, Y = (X, Y) if i is None else (X[i], Y[i])
                df = X[:, 0].sum() - Y[:, 0].sum()
                return (X[:, 1].sum() - Y[:, 1].sum()) / df if df > 0 else np.nan
            m_eg, m_gb = marg(e, G), marg(G, C["base"])
            ci_eg = boot(lambda i: marg(e, G, i), len(probs))
            ci_gb = boot(lambda i: marg(G, C["base"], i), len(probs))
            ci_d = boot(lambda i: marg(e, G, i) - marg(G, C["base"], i), len(probs))
            # equal coverage: the edit's risk on its own curve at g's coverage
            cg = R["arms"][g]["coverage"]
            r_e = risk_at(curve(A["authors_edit"]), cg) if cg <= R["arms"]["authors_edit"]["coverage"] else None
            R["pairs"][g] = {"coverage_g": cg, "risk_g": 1 - R["arms"][g]["finished_accuracy"] if R["arms"][g]["finished_accuracy"] is not None else None,
                             "edit_risk_at_coverage_g": r_e,
                             "marginal_acc_edit_beyond_g": m_eg, "ci": ci_eg,
                             "marginal_acc_g_beyond_base": m_gb, "ci_g": ci_gb, "difference_ci": ci_d}
            # share of the edit's gain a decoding fix recovers (official and strict)
            for col, lab in ((2, "official"), (1, "strict")):
                def frac(i=None):
                    X, Y, B = (G, e, C["base"]) if i is None else (G[i], e[i], C["base"][i])
                    d = Y[:, col].sum() - B[:, col].sum()
                    return (X[:, col].sum() - B[:, col].sum()) / d if d > 0 else np.nan
                R["recovered"].setdefault(g, {})[lab] = {"share": float(frac()), "ci": boot(frac, len(probs))}
        R["picked_on_practice"] = PICKED_ON_PRACTICE.get(name)
        out[name] = R
    return out


def period_words(text):
    """smallest period (words) of a verbatim repeat at the end (>= 3 repeats, >= 40 words); None if not verbatim"""
    w = text.split()[:-1][-1200:]
    n = len(w)
    for p in range(1, 201):
        reps = max(3, -(-40 // p))
        if reps * p > n:
            break
        blk = w[n - p:]
        if all(w[n - (j + 1) * p:n - j * p] == blk for j in range(1, reps)):
            return p
    return None


def part_b(runs, levels_file, tok_dir):
    from transformers import AutoTokenizer

    from wsa.direction import onset
    lv = {json.loads(l)["unique_id"]: json.loads(l)["level"] for l in open(levels_file)}
    out = {"by_level": {}}
    kind, base_stem, arms = SETS["MATH-500"]
    for a, stem in [("base", base_stem)] + list(arms.items()):
        arm = load(runs, stem, kind)
        t = defaultdict(lambda: [0, 0, 0, 0])
        for (p, k), (c, f, _) in arm.items():
            L = lv[p]
            t[L][0] += 1
            t[L][1] += not f
            t[L][2] += c and f
            t[L][3] += f
        out["by_level"][a] = {L: {"samples": v[0], "cut": v[1] / v[0], "finished_accuracy": v[2] / v[3] if v[3] else None,
                                  "problems": sum(1 for p in lv if lv[p] == L)} for L, v in sorted(t.items())}
    tok = AutoTokenizer.from_pretrained(tok_dir)
    out["loops"] = {}
    for name in ("MATH-500", "GSM8K", "MedQA", "MMLU-Pro health"):
        kind, base_stem, arms = SETS[name]
        ons, rel, per, verb, n_cut, ntok = [], [], [], 0, 0, []
        rows = [r for r in (json.loads(l) for l in open(runs / f"{base_stem}.jsonl")) if r["finish"] == "length"]
        if len(rows) > 1500:                                    # at most 1,500 cut answers per benchmark (seeded sample)
            rows = [rows[i] for i in sorted(np.random.default_rng(0).choice(len(rows), 1500, replace=False))]
        for r in rows:
            n_cut += 1
            enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
            n = len(enc["input_ids"])
            t0 = onset(r["text"], enc["offset_mapping"], n)
            if t0 is not None:
                ons.append(t0)
                rel.append(t0 / n)
            p = period_words(r["text"][-20000:])
            if p is not None:
                verb += 1
                per.append(p)
            ntok.append(n)
        base = load(runs, base_stem, kind)
        t1 = load(runs, arms["t1"], kind)
        cut = [k for k, v in base.items() if not v[1]]
        broke = sum(t1[k][1] for k in cut if k in t1)
        q = lambda x: [float(np.percentile(x, s)) for s in (25, 50, 75)] if x else None
        out["loops"][name] = {"cut_samples": n_cut, "with_onset": len(ons), "onset_tokens_q": q(ons), "onset_share_q": q(rel),
                              "verbatim_share": verb / n_cut if n_cut else None, "period_words_q": q(per),
                              "t1_breaks_share": broke / len(cut) if cut else None,
                              "cut_share_base": len(cut) / len(base), "cut_share_t1": float(np.mean([not v[1] for v in t1.values()]))}
    return out


def part_c(runs):
    out = {}
    for name in ("MATH-500", "GSM8K", "MedQA", "MMLU-Pro health"):
        kind, base_stem, arms = SETS[name]
        base = load(runs, base_stem, kind)
        ncut = Counter(p for (p, k), (_, f, _) in base.items() if not f)
        R = {}
        for a, stem in arms.items():
            fx = load(runs, stem, kind)
            fin = [(ncut.get(p, 0), not c) for (p, k), (c, f, _) in fx.items() if f]
            resc = [(ncut.get(k[0], 0), not fx[k][0]) for k in fx if fx[k][1] and k in base and not base[k][1]]
            flag = {}
            for th in (3, 4):
                fl = [w for s, w in fin if s >= th]
                un = [w for s, w in fin if s < th]
                flag[th] = {"flagged_share": len(fl) / len(fin), "acc_flagged": 1 - np.mean(fl) if fl else None,
                            "acc_unflagged": 1 - np.mean(un) if un else None}
            R[a] = {"auroc_all_finished": auroc(*zip(*fin)) if fin else None,
                    "auroc_rescued": auroc(*zip(*resc)) if resc else None,
                    "flag_if_base_cut_at_least": flag}
        out[name] = R
    return out


def ratios(runs):
    grid = json.load(open(runs.parent / "part1b_grid.json"))["cells"]
    x = np.array([1 - v["cut"] for v in grid.values()])
    y = np.array([v["acc"] for v in grid.values()])
    b = float((x @ y) / (x @ x))
    res = y - b * x
    r2_origin = 1 - (res @ res) / (y @ y)
    r2_centered = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    slope, icpt = np.polyfit(x, y, 1)
    res2 = y - (slope * x + icpt)
    layers = json.load(open(runs.parent / "part1b_layers.json"))["arms"]
    lay = {f"L{a['layer']}" if a["layer"] is not None else "L15+19": (1 - a["cut"], a["acc"], a["acc"] - b * (1 - a["cut"]))
           for a in layers}
    return {"grid_cells": len(x), "acc_over_finished_slope_through_origin": b, "r2_through_origin_uncentered": float(r2_origin),
            "r2_through_origin_centered": float(r2_centered), "free_fit": {"slope": float(slope), "intercept": float(icpt),
            "r2": float(1 - (res2 @ res2) / ((y - y.mean()) @ (y - y.mean())))},
            "layers_residual_from_line": {k: {"finished": v[0], "acc": v[1], "residual": v[2]} for k, v in lay.items()}}


def main(a):
    runs = Path(a.runs)
    res = {"a": part_a(runs), "c": part_c(runs), "ratios": ratios(runs)}
    res["b"] = part_b(runs, a.levels, a.tok)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print("written", a.out + ".json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--levels", required=True)
    ap.add_argument("--tok", default="ckpt/base")
    ap.add_argument("--out", default="results/abstain")
    main(ap.parse_args())
