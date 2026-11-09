"""Released-answer analyses from existing outputs only (CPU, no new generations).

1. Rescued answers, one definition everywhere: a sample (problem, sample index) the base model cut at the token limit
   and the fix finished. Rescued accuracy is set against the base model's own finished answers, overall and matched
   on difficulty: (a) within the same problem (problems where the base model both finished and was cut), and (b) by
   Qwen3.5-2B's answer to the same problem where 2B was run (MedQA 200, practice 200).
2. Confidence from agreement: a sample's confidence is the share of the arm's samples of that problem that gave the
   same answer (a cut sample gives no answer). Selective answering answers a problem only when its most common answer
   has at least t of the 4 votes; accuracy-coverage over t = 1..4, per arm.

Answers that grade correct are one group (they equal the reference); wrong answers group by their extracted text.
    python -m wsa.rescue --runs results/runs --out results/rescue
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wsa.grade import extract, mc_extract

ARMS = {"authors_edit": "agent's edit", "t1": "T=1.0 (model card)", "bf8k": "budget forcing 8k", "loop": "loop stop",
        "meandiff": "mean-difference direction", "nothink": "thinking off"}
SETS = {   # name: (kind, base, {arm: path}, problems file stem); paths are <dir>/<file stem without .jsonl>
    "MATH-500": ("math", "base/think35_math500", {
        "authors_edit": "edit_authors/think35_math500", "t1": "base/card_think_math500", "bf8k": "base/think35_bf8192_math500",
        "loop": "base/think35_loop_math500", "meandiff": "meandiff_top64_a6/think35_math500", "nothink": "base/nothink35_math500"}),
    "GSM8K": ("math", "base/think35_gsm8k", {
        "authors_edit": "edit_authors/think35_gsm8k", "t1": "base/card_think_gsm8k", "bf8k": "base/think35_bf8192_gsm8k",
        "loop": "base/think35_loop_gsm8k", "meandiff": "meandiff_top64_a6/think35_gsm8k", "nothink": "base/nothink35_gsm8k"}),
    "practice": ("math", "base/think35_sel", {
        "t1": "base/card_think_sel", "bf8k": "base/think35_bf8192_sel", "loop": "base/think35_loop_sel",
        "meandiff": "meandiff_top64_a6/think35_sel", "nothink": "base/nothink35_sel"}),
    "MedQA": ("mc", "med_base/think35_medqa", {
        "authors_edit": "med_edit/think35_medqa", "t1": "med_base/card_think_medqa", "bf8k": "med_base/think35_bf8192_medqa",
        "loop": "med_base/think35_loop_medqa", "meandiff": "med_md_a6/think35_medqa", "nothink": "med_base/nothink35_medqa"}),
    "MMLU-Pro health": ("mc", "mmlu_base/think35_mmlupro_health", {
        "authors_edit": "mmlu_edit/think35_mmlupro_health", "t1": "mmlu_base/card_think_mmlupro_health",
        "bf8k": "mmlu_base/think35_bf8192_mmlupro_health", "loop": "mmlu_base/think35_loop_mmlupro_health",
        "meandiff": "mmlu_md_a6/think35_mmlupro_health", "nothink": "mmlu_base/nothink35_mmlupro_health"}),
}
TWO_B = {"MedQA": "med_qwen35_2b/think35_medqa_s200", "practice": "l2b/base/think35_sel200"}


def load(runs, stem, kind):
    """{(id, k): (correct, finished, answer group)}"""
    corr = {}
    for r in json.load(open(runs / f"{stem}.per_problem.json")):
        for k, c in enumerate(r["correct"]):
            corr[(r["id"], k)] = bool(c)
    out = {}
    for line in open(runs / f"{stem}.jsonl"):
        r = json.loads(line)
        key = (r["id"], r["k"])
        if key not in corr:
            continue
        fin = r["finish"] != "length"
        if not fin:
            ans = None                                         # a cut sample gives no answer (strict)
        elif corr[key]:
            ans = "<correct>"
        else:
            ans = mc_extract(r["text"]) if kind == "mc" else extract(r["text"])[0]
            ans = "<none>" if ans is None else str(ans).replace(" ", "")
        out[key] = (corr[key], fin, ans)
    return out


def boot_ci(per_problem_diffs, n=10000, seed=0):
    d = np.array(per_problem_diffs, float)
    if len(d) == 0:
        return None, None, None
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def confidence(arm):
    """per problem: Counter of answers; per sample: share of the problem's samples with its answer"""
    by_p = defaultdict(list)
    for (pid, k), (_, _, ans) in arm.items():
        by_p[pid].append(ans)
    votes = {p: Counter(a for a in v if a is not None) for p, v in by_p.items()}
    n = {p: len(v) for p, v in by_p.items()}
    conf = {key: (votes[key[0]][ans] / n[key[0]] if ans is not None else 0.0) for key, (_, _, ans) in arm.items()}
    return votes, n, conf


def selective(arm):
    votes, n, _ = confidence(arm)
    total = len(votes)
    rows = []
    for t in range(max(n.values()), 0, -1):
        answered = [p for p, c in votes.items() if c and c.most_common(1)[0][1] >= t]
        right = sum(votes[p].most_common(1)[0][0] == "<correct>" for p in answered)
        rows.append({"min_votes": t, "coverage": len(answered) / total, "accuracy": right / len(answered) if answered else None,
                     "answered": len(answered), "right": right})
    return rows


def main(a):
    runs = Path(a.runs)
    res = {"definition": "rescued = a sample (problem, index) the base model cut at the token limit and the fix finished",
           "sets": {}}
    for name, (kind, base_stem, arms) in SETS.items():
        base = load(runs, base_stem, kind)
        bfin = [c for c, f, _ in base.values() if f]
        cut_p = {p for (p, k), (_, f, _) in base.items() if not f}
        base_votes, _, base_conf = confidence(base)
        S = {"kind": kind, "samples": len(base), "problems": len({p for p, _ in base}),
             "base": {"cut_share": float(np.mean([not f for _, f, _ in base.values()])),
                      "finished_accuracy": float(np.mean(bfin)) if bfin else None, "finished": len(bfin),
                      "selective": selective(base)},
             "chance": 0.25 if name == "MedQA" else None, "arms": {}}
        two_b = None
        if name in TWO_B and (runs / f"{TWO_B[name]}.per_problem.json").exists():
            two_b = {r["id"]: bool(r["correct"][0]) for r in json.load(open(runs / f"{TWO_B[name]}.per_problem.json"))}
            stuck = [two_b[p] for p in two_b if p in cut_p]
            free = [two_b[p] for p in two_b if p not in cut_p and any((p, k) in base for k in range(4))]
            S["2b_accuracy_on_stuck_vs_other"] = {"stuck": float(np.mean(stuck)) if stuck else None, "n_stuck": len(stuck),
                                                  "other": float(np.mean(free)) if free else None, "n_other": len(free)}
        for arm, stem in arms.items():
            if not (runs / f"{stem}.per_problem.json").exists():
                continue
            fx = load(runs, stem, kind)
            keys = [k for k in base if k in fx]
            resc = [k for k in keys if not base[k][1] and fx[k][1]]
            r_acc = float(np.mean([fx[k][0] for k in resc])) if resc else None
            # difficulty matched within the problem: rescued vs the base model's finished samples of the same problem
            per_p = defaultdict(lambda: ([], []))
            for k in resc:
                per_p[k[0]][0].append(fx[k][0])
            for k in keys:
                if base[k][1]:
                    per_p[k[0]][1].append(base[k][0])
            diffs = [np.mean(r) - np.mean(b) for r, b in per_p.values() if r and b]
            within = boot_ci(diffs)
            pairs = [(np.mean(r), np.mean(b)) for r, b in per_p.values() if r and b]
            A = {"rescued": len(resc), "rescued_accuracy": r_acc,
                 "newly_cut": sum(base[k][1] and not fx[k][1] for k in keys),
                 "within_problem": {"problems": len(diffs), "rescued_acc": float(np.mean([x for x, _ in pairs])) if pairs else None,
                                    "base_finished_acc": float(np.mean([y for _, y in pairs])) if pairs else None,
                                    "diff": within[0], "lo": within[1], "hi": within[2]}}
            if two_b is not None:
                strata = {}
                for lab, val in (("2B right", True), ("2B wrong", False)):
                    rr = [fx[k][0] for k in resc if two_b.get(k[0]) is val]
                    bb = [base[k][0] for k in keys if base[k][1] and two_b.get(k[0]) is val]
                    strata[lab] = {"rescued_acc": float(np.mean(rr)) if rr else None, "n_rescued": len(rr),
                                   "base_finished_acc": float(np.mean(bb)) if bb else None, "n_base_finished": len(bb)}
                A["by_2b"] = strata
            # confidence of rescued answers, right and wrong, next to the base model's finished answers
            _, _, fconf = confidence(fx)
            A["confidence"] = {
                "rescued_right": float(np.mean([fconf[k] for k in resc if fx[k][0]])) if any(fx[k][0] for k in resc) else None,
                "rescued_wrong": float(np.mean([fconf[k] for k in resc if not fx[k][0]])) if any(not fx[k][0] for k in resc) else None,
                "base_finished_right": float(np.mean([base_conf[k] for k in keys if base[k][1] and base[k][0]])),
                "base_finished_wrong": float(np.mean([base_conf[k] for k in keys if base[k][1] and not base[k][0]]))}
            if max(k for _, k in fx) > 0:
                A["selective"] = selective(fx)
            S["arms"][arm] = A
        res["sets"][name] = S
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/rescue")
    main(ap.parse_args())
