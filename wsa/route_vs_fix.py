"""Route or fix? (CPU only, existing outputs only.) When Qwen3.5-0.8B (the
authors' protocol, think35) is headed for a loop, does handing the question to a stronger Qwen3.5 (9B; 4B and 2B as
cheaper ones, each one sample under the same protocol from the screens) beat the fixes that answer every item, and does
a loop signal beat a plain length threshold?

Items: the 200 practice math problems (sel200) and the 200 MedQA questions (medqa_s200), where all models ran, and (medqa_full)
all 1,273 MedQA test questions with 9B as the strong model. The
small model's sample is k = 0 (MedQA also reports the mean over its 4 samples). Every answer is read the authors' way
(per_problem "correct", a cut answer's last answer included).

Triggers on the small model's sample, each deferring the item to the strong model at the moment it fires:
  loop         the authors' looping rule (wsa.grade.looping) on the text so far, checked every 256 tokens
  ngram_k      some 30-token n-gram seen k times so far (k = 3, 5, 10, 20; 20 is Pipis et al.'s whole-answer threshold)
  length_L     the answer passes L tokens (L every 256 tokens up to the cap): the Fail Fast style baseline
  cap          the answer reaches the 32,768-token cap
  disagree_a   MedQA only, from the 4 samples (4x the small-model tokens): defer when fewer than a of the 4 agree with
               their majority answer; a kept item gets the majority answer
Per trigger and strong model: deferral rate (= strong-model calls), overall accuracy (kept small answers plus the strong
model on deferred items), delivered-error rate among kept answers, false deferrals (deferred items the small model got
right), small-model tokens before deferral (mean over deferred items) and in all (mean per item). Fix arms answer every
item: accuracy and delivered errors. Criteria (pre-set, both needed to pivot):
  1. MedQA: deferring to 9B gains >= 8 points over the best fix arm, at <= 30% deferral
  2. at a matched deferral rate the loop trigger beats the length trigger: fewer delivered errors, or >= 30% fewer
     small-model tokens before deferral, with no more false deferrals; on both domains

    nice -n 10 env OMP_NUM_THREADS=4 python -m wsa.route_vs_fix --out results/route_vs_fix
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from wsa.grade import TAIL_CHARS, looping, mc_extract

RUNS = Path("results/runs")
CAP, STEP, NG = 32768, 256, 30
DOMAINS = {
    "math": {"set": "results/data/sel200.jsonl", "small": "base/think35_sel",
             "strong": {"9B": "screen/qwen35_9b/think35_sel200", "4B": "screen/qwen35_4b/think35_sel200",
                        "2B": "l2b/base/think35_sel200"},
             "fixes": {"T 1.0 (model card)": "base/card_think_sel", "thinking off (8k cap)": "base/nothink35_sel",
                       "budget forcing 8k": "base/think35_bf8192_sel", "loop stop (forced answer)": "base/think35_loop_sel",
                       "textbook edit (authors' hyperparameters)": "meandiff_top64_a6/think35_sel",
                       "authors' edit": "csweep/c6/think35_sel200"},
             "missing": None},
    "medqa": {"set": "results/data/medqa_s200.jsonl", "small": "med_base/think35_medqa",
              "strong": {"9B": "screen/qwen35_9b/think35_medqa_s200", "4B": "screen/qwen35_4b/think35_medqa_s200",
                         "2B": "med_qwen35_2b/think35_medqa_s200"},
              "fixes": {"authors' edit": "med_edit/think35_medqa", "T 1.0 (model card)": "med_base/card_think_medqa",
                        "thinking off (8k cap)": "med_base/nothink35_medqa", "budget forcing 8k": "med_base/think35_bf8192_medqa",
                        "loop stop (forced answer)": "med_base/think35_loop_medqa", "textbook edit": "med_md_a6/think35_medqa",
                        "prompt only": "prompt/think35_pr2_medqa"},
              "missing": None},
    # all 1,273 MedQA test questions (once medqa9b gave 9B there); 9B only, the same fixes, the same criteria
    "medqa_full": {"set": "results/data/medqa.jsonl", "small": "med_base/think35_medqa",
                   "strong": {"9B": "medqa9b/think35_medqa"},
                   "fixes": {"authors' edit": "med_edit/think35_medqa", "T 1.0 (model card)": "med_base/card_think_medqa",
                             "thinking off (8k cap)": "med_base/nothink35_medqa", "budget forcing 8k": "med_base/think35_bf8192_medqa",
                             "loop stop (forced answer)": "med_base/think35_loop_medqa", "textbook edit": "med_md_a6/think35_medqa",
                             "prompt only": "prompt/think35_pr2_medqa"},
                   "missing": None},
}


def correct_map(stem):
    return {(r["id"], k): bool(c) for r in json.load(open(RUNS / f"{stem}.per_problem.json")) for k, c in enumerate(r["correct"])}


def gens(stem, ids):
    out = {}
    for line in open(RUNS / f"{stem}.jsonl"):
        g = json.loads(line)
        if g["id"] in ids:
            out[(g["id"], g["k"])] = g
    return out


def loop_onset(text, tok):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    for t in range(STEP, len(ids) + STEP, STEP):
        end = offs[min(t, len(ids)) - 1][1]
        if looping(text[max(0, end - TAIL_CHARS):end]):
            return min(t, len(ids)), ids
        if t >= len(ids):
            break
    return None, ids


def ngram_onsets(ids, ks):
    """{k: the token position at which some 30-gram is seen for the k-th time, or None}"""
    c, out, need = Counter(), {}, sorted(ks)
    for i in range(NG, len(ids) + 1):
        g = tuple(ids[i - NG:i])
        c[g] += 1
        while need and c[g] >= need[0]:
            out[need.pop(0)] = i
        if not need:
            break
    return {k: out.get(k) for k in ks}


def evaluate(items, small_ok, small_tok, strong_ok, defer_at):
    """defer_at: {item: token position of deferral} for deferred items"""
    n = len(items)
    D = [i for i in items if i in defer_at]
    K = [i for i in items if i not in defer_at]
    acc = (sum(small_ok[i] for i in K) + sum(strong_ok[i] for i in D)) / n
    kept_err = (sum(not small_ok[i] for i in K) / len(K)) if K else None
    fd = sum(small_ok[i] for i in D)
    before = float(np.mean([defer_at[i] for i in D])) if D else 0.0
    spent = float(np.mean([defer_at[i] if i in defer_at else small_tok[i] for i in items]))
    return {"defer_rate": len(D) / n, "acc": acc, "kept_err": kept_err, "false_defer": fd, "false_defer_rate": fd / n,
            "tokens_before_defer": before, "small_tokens_per_item": spent, "n_defer": len(D),
            "vec": [int(strong_ok[i]) if i in defer_at else int(small_ok[i]) for i in items]}


def boot(a, b, n=10000, seed=0):
    d = np.array(a, float) - np.array(b, float)
    rng = np.random.default_rng(seed)
    s = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = {"domains": {}}
    for dom, cfg in DOMAINS.items():
        rows = [json.loads(l) for l in open(cfg["set"])]
        ids = [r["id"] for r in rows]
        gold = {r["id"]: r["answer"] for r in rows}
        sc, sg = correct_map(cfg["small"]), gens(cfg["small"], set(ids))
        ks = sorted({k for (_, k) in sg})
        R = {"n_items": len(ids), "small_samples": ks, "missing_fix": cfg["missing"], "fixes": {}, "strong": {}}
        # fix arms: every item answered, sample 0 (and the mean over samples where there are 4)
        for name, stem in cfg["fixes"].items():
            c = correct_map(stem)
            R["fixes"][name] = {"acc": float(np.mean([c[(i, 0)] for i in ids])),
                                "acc_all_samples": float(np.mean([c[(i, k)] for i in ids for k in ks if (i, k) in c])),
                                "vec": [int(c[(i, 0)]) for i in ids]}
        small_ok = {i: sc[(i, 0)] for i in ids}
        R["small_alone"] = {"acc": float(np.mean(list(small_ok.values()))),
                            "cut": float(np.mean([sg[(i, 0)]["finish"] == "length" for i in ids]))}
        # triggers on sample 0 (loop and n-gram onsets from the token prefix)
        onset, ngram, ntok, cut = {}, {}, {}, {}
        for i in ids:
            g = sg[(i, 0)]
            t0, tid = loop_onset(g["text"], tok)
            onset[i], ngram[i] = t0, ngram_onsets(tid, (3, 5, 10, 20))
            ntok[i], cut[i] = g["n_tok"], g["finish"] == "length"
        R["loop_onset_tokens_median"] = float(np.median([v for v in onset.values() if v])) if any(onset.values()) else None
        R["loop_fires"] = int(sum(1 for v in onset.values() if v))
        R["loop_fires_on_cut"] = int(sum(1 for i in ids if onset[i] and cut[i]))
        R["loop_fires_on_finished"] = int(sum(1 for i in ids if onset[i] and not cut[i]))
        R["loop_fires_finished_correct"] = int(sum(1 for i in ids if onset[i] and not cut[i] and small_ok[i]))
        trig = {"loop": {i: onset[i] for i in ids if onset[i]}, "cap": {i: CAP for i in ids if cut[i]}}
        for k in (3, 5, 10, 20):
            trig[f"ngram_{k}"] = {i: ngram[i][k] for i in ids if ngram[i][k]}
        lengths = {f"length_{L}": {i: L for i in ids if ntok[i] > L} for L in range(STEP, CAP, STEP)}
        # disagreement among the 4 samples (MedQA)
        dis = {}
        if len(ks) == 4:
            maj = {}
            for i in ids:
                ans = [mc_extract(sg[(i, k)]["text"]) for k in ks]
                top, cnt = Counter(ans).most_common(1)[0]
                maj[i] = (top, cnt, top == gold[i])
            for thr in (4, 3):
                dis[f"disagree_{thr}"] = {i: 4 * ntok[i] for i in ids if maj[i][1] < thr}
            R["majority_vote_acc"] = float(np.mean([maj[i][2] for i in ids]))
        for sname, stem in cfg["strong"].items():
            stc = correct_map(stem)
            strong_ok = {i: stc[(i, 0)] for i in ids}
            S = {"alone_acc": float(np.mean(list(strong_ok.values()))), "triggers": {}, "length_curve": {}}
            for tname, dat in {**trig, **lengths}.items():
                e = evaluate(ids, small_ok, ntok, strong_ok, dat)
                (S["length_curve"] if tname.startswith("length_") else S["triggers"])[tname] = e
            for tname, dat in dis.items():
                kept_ok = {i: (maj[i][2] if i not in dat else small_ok[i]) for i in ids}
                e = evaluate(ids, kept_ok, {i: 4 * ntok[i] for i in ids}, strong_ok, dat)
                e["note"] = "4 samples per item (4x the small-model tokens); a kept item gets the majority answer"
                S["triggers"][tname] = e
            # criterion 2: each loop-type trigger against the length threshold with the nearest deferral rate
            S["matched"] = {}
            curve = sorted(S["length_curve"].items(), key=lambda kv: kv[1]["defer_rate"])
            for tname in ["loop"] + [f"ngram_{k}" for k in (3, 5, 10, 20)]:
                e = S["triggers"][tname]
                Ln, en = min(curve, key=lambda kv: abs(kv[1]["defer_rate"] - e["defer_rate"]))
                fewer_err = e["kept_err"] is not None and en["kept_err"] is not None and e["kept_err"] < en["kept_err"]
                earlier = en["tokens_before_defer"] > 0 and e["tokens_before_defer"] <= 0.7 * en["tokens_before_defer"]
                S["matched"][tname] = {"length": Ln, "rate_trigger": e["defer_rate"], "rate_length": en["defer_rate"],
                                       "kept_err": [e["kept_err"], en["kept_err"]], "false_defer": [e["false_defer"], en["false_defer"]],
                                       "tokens_before": [e["tokens_before_defer"], en["tokens_before_defer"]],
                                       "acc": [e["acc"], en["acc"]],
                                       "passes": bool((fewer_err or earlier) and e["false_defer"] <= en["false_defer"]),
                                       "rate_gap": abs(e["defer_rate"] - en["defer_rate"])}
            R["strong"][sname] = S
        # criterion 1 (MedQA, 9B): the best configuration at <= 30% deferral against the best fix arm
        best_fix = max(R["fixes"].items(), key=lambda kv: kv[1]["acc"])
        S9 = R["strong"]["9B"]
        cand = {**S9["triggers"], **S9["length_curve"]}
        ok30 = {k: v for k, v in cand.items() if v["defer_rate"] <= 0.30}
        R["criterion1"] = {"best_fix": best_fix[0], "best_fix_acc": best_fix[1]["acc"]}
        for label, key in (("loop", "loop"), ("cap", "cap"), ("best_at_30", max(ok30, key=lambda k: ok30[k]["acc"]) if ok30 else None)):
            if key is None or key not in cand or cand[key]["defer_rate"] > 0.30:
                R["criterion1"][label] = None if key is None else {"trigger": key, "defer_rate": cand[key]["defer_rate"], "over_30": True}
                continue
            m, lo, hi = boot(cand[key]["vec"], best_fix[1]["vec"])
            R["criterion1"][label] = {"trigger": key, "defer_rate": cand[key]["defer_rate"], "acc": cand[key]["acc"],
                                      "gain": m, "lo": lo, "hi": hi}
        out["domains"][dom] = R
    for R in out["domains"].values():               # drop the per-item vectors from the saved file
        for f in R["fixes"].values():
            f.pop("vec", None)
        for S in R["strong"].values():
            for d in (S["triggers"], S["length_curve"]):
                for e in d.values():
                    e.pop("vec", None)
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    print(json.dumps({d: {k: v for k, v in R.items() if k in ("small_alone", "criterion1", "loop_fires", "loop_fires_on_cut",
                                                            "loop_fires_on_finished", "loop_onset_tokens_median")}
                      for d, R in out["domains"].items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="ckpt/base")
    ap.add_argument("--out", default="results/route_vs_fix")
    main(ap.parse_args())
