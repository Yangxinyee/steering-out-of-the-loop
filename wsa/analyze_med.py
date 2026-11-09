"""Part 2 table (Qwen3.5-0.8B, 4 samples per question; MedQA test, or MMLU-Pro health with --prefix mmlu --tag
mmlupro_health): every arm's accuracy by two rules, its share cut at the cap,
mean tokens, paired gain over the base model (95% bootstrap interval over questions), and what happens to the answers
the base model lost to the cap.

  official  the MMLU-Pro reading (wsa.grade.mc_extract) of every answer, a cut one included (its last letter)
  strict    a cut answer is wrong, whatever letter its unfinished reasoning ends on
  rescued   same question and sample index: cut under the base model, finished under the arm. Medicine needs to know
            whether these are right, since a wrong answer is worse than none: their accuracy is set against the base
            model's on the answers it finished itself, and chance (1 / the question's number of options).
The mean-difference coefficient is always the one picked on MedQA train (runs/med_md_a<x>/think35_medqa_sel); on
MMLU-Pro health the MedQA direction and coefficient are used unchanged (no health question chooses anything).

    python -m wsa.analyze_med --runs runs --rows data/bench/medqa.jsonl --sel-rows data/practice/medqa_sel.jsonl \
        --out results/part2_full.json
    python -m wsa.analyze_med --runs runs --rows data/bench/mmlupro_health.jsonl --sel-rows data/practice/medqa_sel.jsonl \
        --prefix mmlu --tag mmlupro_health --out results/part2_mmlu_health.json
"""
import argparse
import json
import os

import numpy as np

from wsa.analyze import boot_ci
from wsa.grade import mc_extract

ARMS = {"base/think35": "base (think35)", "base/card_think": "T=1.0 (model card)", "base/nothink35": "thinking off",
        "base/think35_bf8192": "budget forcing 8k", "base/think35_loop": "loop breaking",
        "edit/think35": "authors' math edit, unchanged"}
ALPHAS = (3, 6, 9)


def chance(row):
    """1 / the number of lettered options in the question's prompt"""
    opts = row["prompt"].split("\nOptions: ", 1)[1].split("\nAnswer:", 1)[0]
    return 1 / sum(1 for line in opts.split("\n") if len(line) > 2 and line[1:3] == ". " and line[0].isupper())


def load(runs, arm, rows, tag="medqa"):
    """{(id, k): (official, strict, cut, n_tok)} of one arm's generation file."""
    out = {}
    for line in open(os.path.join(runs, f"{arm}_{tag}.jsonl")):
        g = json.loads(line)
        cut = g["finish"] == "length"
        ok = mc_extract(g["text"]) == rows[g["id"]]["answer"]
        out[(g["id"], g["k"])] = (ok, ok and not cut, cut, g["n_tok"])
    return out


def per_question(d, col):
    q = {}
    for (i, _), v in d.items():
        q.setdefault(i, []).append(v[col])
    return {i: float(np.mean(v)) for i, v in q.items()}


def gain(a, b, col):
    pa, pb = per_question(a, col), per_question(b, col)
    d = np.array([pa[i] - pb[i] for i in pa if i in pb])
    return {"gain": float(d.mean()), "ci95": boot_ci(d)}


def summary(d):
    v = np.array(list(d.values()), dtype=float)
    return {"official": v[:, 0].mean(), "strict": v[:, 1].mean(), "cut": v[:, 2].mean(), "mean_tokens": v[:, 3].mean(),
            "n": len(v)}


def rescue(arm, base):
    keys = [k for k in base if k in arm]
    resc = [k for k in keys if base[k][2] and not arm[k][2]]
    newly_cut = [k for k in keys if not base[k][2] and arm[k][2]]
    lost = [k for k in keys if base[k][1] and not arm[k][0]]      # base right and finished, arm wrong
    return {"base_cut": sum(base[k][2] for k in keys), "rescued": len(resc),
            "rescued_correct": sum(arm[k][0] for k in resc),
            "rescued_acc": float(np.mean([arm[k][0] for k in resc])) if resc else None,
            "newly_cut": len(newly_cut), "base_right_now_wrong": len(lost)}


def main(a):
    rows = {r["id"]: r for r in (json.loads(l) for l in open(a.rows))}
    sel_rows = {r["id"]: r for r in (json.loads(l) for l in open(a.sel_rows))}
    prac = {}
    for arm in ["med_base/think35"] + [f"med_md_a{x}/think35" for x in ALPHAS]:
        p = os.path.join(a.runs, f"{arm}_medqa_sel.jsonl")
        if os.path.exists(p):
            prac[arm] = summary(load(a.runs, arm, sel_rows, "medqa_sel"))
    best = max((x for x in ALPHAS if f"med_md_a{x}/think35" in prac),
               key=lambda x: prac[f"med_md_a{x}/think35"]["official"], default=None)   # first maximum: a tie keeps the smaller
    picked = f"med_md_a{best}/think35" if best else None
    arms = {f"{a.prefix}_{k}": v for k, v in ARMS.items()}
    if best:
        arms[f"{a.prefix}_md_a{best}/think35"] = f"mean-difference direction from MedQA train, coefficient {best}"
    BASE = f"{a.prefix}_base/think35"
    data = {arm: load(a.runs, arm, rows, a.tag) for arm in arms if os.path.exists(os.path.join(a.runs, f"{arm}_{a.tag}.jsonl"))}
    base = data[BASE]
    chance_level = float(np.mean([chance(rows[i]) for i in {i for i, _ in base}]))
    fin = [v for v in base.values() if not v[2]]
    base_finished_acc = float(np.mean([v[0] for v in fin]))
    table = {}
    for arm, d in data.items():
        t = {"label": arms[arm], **summary(d)}
        if arm != BASE:
            t["vs_base_official"] = gain(d, base, 0)
            t["vs_base_strict"] = gain(d, base, 1)
            t["rescue"] = rescue(d, base)
        table[arm] = t
    out = {"test": table, "practice": prac, "picked": picked, "base_finished_acc": base_finished_acc, "chance": chance_level}
    json.dump(out, open(a.out, "w"), indent=1)

    print("practice (MedQA train, 500 questions, 1 sample): official / strict / cut")
    for arm, p in prac.items():
        print(f"  {arm:<26} {100 * p['official']:5.1f} {100 * p['strict']:5.1f} {100 * p['cut']:5.1f}%"
              + ("  <- picked" if arm == picked else ""))
    print(f"\ntest ({len({i for i, _ in base}):,} questions x 4): official, strict, cut, tokens, gain over base [95%] official | strict")
    for arm, t in table.items():
        g = lambda k: (f"{100 * t[k]['gain']:+5.1f} [{100 * t[k]['ci95'][0]:+.1f},{100 * t[k]['ci95'][1]:+.1f}]"  # noqa: E731
                       if k in t else "")
        print(f"  {t['label']:<52} {100 * t['official']:5.1f} {100 * t['strict']:5.1f} {100 * t['cut']:5.1f}% "
              f"{t['mean_tokens'] / 1000:5.1f}k  {g('vs_base_official'):<20} {g('vs_base_strict')}")
    print(f"\nrescued answers (cut under base, finished under the arm): accuracy; base on its own finished answers "
          f"{100 * base_finished_acc:.1f}%, chance {100 * chance_level:.1f}%")
    for arm, t in table.items():
        r = t.get("rescue")
        if r:
            ra = f"{100 * r['rescued_acc']:.1f}%" if r["rescued_acc"] is not None else "-"
            print(f"  {t['label']:<52} {r['rescued']:5d} of {r['base_cut']} rescued, {ra} right; "
                  f"{r['newly_cut']} newly cut, {r['base_right_now_wrong']} base-right now wrong")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--sel-rows", required=True, help="the coefficient set (practice/medqa_sel.jsonl)")
    ap.add_argument("--prefix", default="med", help="run folders <prefix>_base, <prefix>_edit, <prefix>_md_a<x>")
    ap.add_argument("--tag", default="medqa", help="the test set's file tag")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
