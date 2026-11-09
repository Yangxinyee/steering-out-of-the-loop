"""Every number the paper prints, read from the result files (no number typed by hand): paper/numbers.tex (LaTeX
macros, prefix \\w) and the generated tables paper/tables/{main,med,frontier,pair,ledger,levels,scale}.tex. Half-up rounding of the value as
computed (7.55 -> 7.6).

    python -m wsa.paper_numbers
"""
import json
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper"


def J(name):
    return json.load(open(R / name))


def score(stem):
    s = json.load(open(R / "runs" / f"{stem}.scores.json"))
    return s["acc"], 1 - s["cut_share"], s["mean_tokens"]


def rnd(x, nd=1):               # half-up rounding of the value as computed (7.55 -> 7.6), not of its binary float
    return Decimal(f"{x:.10f}").quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def fx(x, nd=1):                # plain string, no "-0.0"
    d = rnd(x, nd)
    return str(abs(d) if d == 0 else d)


def f1(x):                      # a share as a percentage, one decimal
    return fx(x * 100)


def num(x, nd=1):               # a plain number, minus sign typeset (no "-0.0")
    return fx(x, nd).replace("-", "$-$")


def sg(x, nd=1):                # a signed difference
    d = rnd(x, nd)
    s = ("+" if d >= 0 else "") + str(abs(d) if d == 0 else d)
    return s.replace("-", "$-$").replace("+", "$+$")


def ci(lo, hi, nd=1, scale=100):
    return f"[{num(lo * scale, nd)}, {num(hi * scale, nd)}]"


M = {}


def put(name, value):
    assert name not in M, name
    M[name] = value


# ---- accuracy and completion, MATH-500 and GSM8K (4 samples per problem, the authors' scoring)
e3 = J("e3.json")["test"]
ARM = {"base": "base (think35)", "t1": "T 1.0 (model card)", "bf": "budget forcing 8k", "loop": "loop stop (forced answer)",
       "eot": "logit bias +8 on </think> (E3)", "off8k": "thinking off, 8k cap (part 1)",
       "off": "thinking off, 32k cap (E3)", "textbook": "textbook edit (mean difference, strength 6)",
       "edit": "authors' edit"}
for ds, tag in (("math500", "Math"), ("gsm8k", "Gsm")):
    T = {k: e3[ds][v] for k, v in ARM.items()}
    put(f"wBase{tag}", f1(T["base"]["acc"]))
    put(f"wEdit{tag}", f1(T["edit"]["acc"]))
    put(f"wOff{tag}", f1(T["off"]["acc"]))
    put(f"wBaseCut{tag}", fx(T['base']['cut'] * 100, 0))
    put(f"wOffShare{tag}", fx((T['off']['acc'] - T['base']['acc']) / (T['edit']['acc'] - T['base']['acc']) * 100, 0))
    put(f"wEditTok{tag}", fx(T['edit']['mean_tokens'] / 1000) + "k")
    put(f"wOffTok{tag}", fx(T['off']['mean_tokens'] / 1000) + "k")
    d = J("decontam32k.json")[ds]["all"]["edit_minus_thinking_off_32k"]
    put(f"wEditMinusOff{tag}", sg(d["gain"] * 100))
    put(f"wEditMinusOff{tag}CI", ci(*d["ci95"]))

# ---- majority vote at matched tokens (thinking off at the full budget, j samples)
mk = J("majk.json")
for ds, tag, j in (("MATH-500", "Math", "2"), ("GSM8K", "Gsm", "4")):
    v = mk[ds]["edit_minus_thinking_off_majj"][j]
    put(f"wMaj{tag}J", j)
    put(f"wMaj{tag}", sg(v["diff"][0] * 100))
    put(f"wMaj{tag}CI", ci(v["diff"][1], v["diff"][2]))
    put(f"wMaj{tag}Tok", fx(v['thinking_off_tokens'] / 1000) + "k")
    put(f"wMaj{tag}EditTok", fx(v['edit_tokens'] / 1000) + "k")

# ---- coverage of the edit's correct answers on always-looping problems
cv = J("coverage.json")
cm = J("coverage_match.json")
cn = J("coverage_null.json")
for ds, tag in (("MATH-500", "Math"), ("GSM8K", "Gsm")):
    c = cv[ds]["coverage"]["all decodings"]["per_answer"]
    put(f"wCov{tag}", f1(c[0]))
    put(f"wCov{tag}CI", ci(c[1], c[2]))
    put(f"wAlways{tag}", f"{cv[ds]['always_looping_problems']:,}".replace(",", "{,}"))
    put(f"wEditCorrect{tag}", f"{cv[ds]['edit_correct_answers_on_always_looping']:,}".replace(",", "{,}"))
    two = cm[ds]["references"]["Qwen3.5-2B, random"]["coverage"]["32"]
    put(f"wCovTwoB{tag}", f1(two[0]))
    put(f"wCovTwoB{tag}CI", ci(two[1], two[2]))
    put(f"wCovSamples{tag}", str(cv[ds]["coverage"]["all decodings"]["samples_per_problem"]))
    fixes = [v["targets"]["left-out fix"]["coverage"][0] for k, v in cn[ds]["left_out"].items() if k != "thinking off (both budgets)"]
    put(f"wCovFixLo{tag}", f1(min(fixes)))
    put(f"wCovFixHi{tag}", fx(max(fixes) * 100).replace("100.0", "100"))

# ---- released against own answers on the same problems
rs = J("rescue.json")["sets"]
for ds, tag in (("MATH-500", "Math"), ("GSM8K", "Gsm"), ("MedQA", "Med")):
    w = rs[ds]["arms"]["authors_edit"]["within_problem"]
    put(f"wRel{tag}", f1(w["rescued_acc"]))
    put(f"wOwn{tag}", f1(w["base_finished_acc"]))
    put(f"wRelMinusOwn{tag}", sg(w["diff"] * 100))
    put(f"wRelMinusOwn{tag}CI", ci(w["lo"], w["hi"]))
    put(f"wPaired{tag}", f"{w['problems']:,}".replace(",", "{,}"))
w = rs["MMLU-Pro health"]["arms"]["authors_edit"]["within_problem"]
put("wRelMinusOwnHealth", sg(w["diff"] * 100))
put("wRelMinusOwnHealthCI", ci(w["lo"], w["hi"]))
hd = J("headroom.json")
h = hd["MMLU-Pro math, agent's edit"]
put("wMmluBase", f1(h["base"]["acc"]))
put("wMmluEdit", f1(h["fix"]["acc"]))
put("wRelMmlu", f1(h["within_problem"]["released_acc"]))
put("wOwnMmlu", f1(h["within_problem"]["own_acc"]))
put("wRelMinusOwnMmlu", sg(h["within_problem"]["diff"] * 100))
put("wRelMinusOwnMmluCI", ci(h["within_problem"]["lo"], h["within_problem"]["hi"]))
put("wPairedMmlu", str(h["within_problem"]["problems"]))
fxs = [v["fix"]["acc"] for k, v in hd.items() if k.startswith("MMLU-Pro math, ") and k != "MMLU-Pro math, agent's edit"]
put("wMmluFixLo", f1(min(fxs)))
put("wMmluFixHi", f1(max(fxs)))

# ---- matched completion under one selector (error of thinking off minus the edit's, at the lower completion)
rk = J("riskcov.json")
for ds, tag in (("MATH-500", "Math"), ("GSM8K", "Gsm")):
    g = rk[ds]["vs_edit"]["nothink32k"]["error_gap_at_common_coverage"]
    put(f"wRiskOff{tag}", sg(g[0] * 100))
    put(f"wRiskOff{tag}CI", ci(g[1], g[2]))
put("wRiskOnMax", fx(max(abs(rk[ds]['vs_edit'][k]['error_gap_at_common_coverage'][0]) for ds in rk for k in ('t1', 'bf8k', 'loop')) * 100))
lv = J("abstain.json")["b"]["by_level"]["authors_edit"]
put("wLevelOneCut", fx(lv['1']['cut'] * 100, 0))
put("wLevelFiveCut", fx(lv['5']['cut'] * 100, 0))
put("wLevelOneRight", fx(lv['1']['finished_accuracy'] * 100, 0))
put("wLevelFiveRight", fx(lv['5']['finished_accuracy'] * 100, 0))
# ---- the loops are genuine: 81,920 tokens instead of 32,768 (first sample, MATH-500)
e6p = J("e6.json")["paired"]
put("wLongBase", sg(e6p["base: 81,920 - cut at 32,768"]["d"] * 100))
put("wLongBaseCI", ci(*e6p["base: 81,920 - cut at 32,768"]["ci95"]))
put("wLongEditLead", sg(e6p["at 81,920: edit - base"]["d"] * 100))
put("wLongEditLeadCI", ci(*e6p["at 81,920: edit - base"]["ci95"]))

# ---- medicine: the edit against sampling at T = 1.0 on MedQA (authors' scoring)
p2 = J("part2_full.json")
put("wMedBase", f1(p2["test"]["med_base/think35"]["official"]))
put("wMedEdit", f1(p2["test"]["med_edit/think35"]["official"]))
put("wMedTone", f1(p2["test"]["med_base/card_think"]["official"]))
g = p2["pairwise"]["med_edit/think35 - med_base/card_think (official)"]
put("wMedEditMinusTone", sg(g["gain"] * 100))
put("wMedEditMinusToneCI", ci(*g["ci95"]))

# ---- what the agent's search contributed: direction against neurons (200 practice problems)
e2 = J("csweep.json")["e2"]
put("wDirEffect", sg(e2["direction_effect"]["d"] * 100))
put("wDirEffectCI", ci(*e2["direction_effect"]["ci95"]))
put("wColEffect", sg(e2["columns_effect"]["d"] * 100))
put("wColEffectCI", ci(*e2["columns_effect"]["ci95"]))

# ---- two frontier agents redo the task
ae = J("agent_eval.json")
for a, tag in (("openai_1", "Gpt"), ("claude_1", "Claude")):
    v = ae["MATH-500"][a]
    put(f"wAgent{tag}Math", f1(v["acc"]))
    put(f"wAgent{tag}Gain", sg(v["vs"]["base"]["diff"] * 100))
    put(f"wAgent{tag}GainCI", ci(v["vs"]["base"]["lo"], v["vs"]["base"]["hi"]))
rel = [ae[ds][a]["within_problem"]["diff"] for ds in ae for a in ae[ds]]
put("wAgentRelLo", sg(min(rel) * 100))
put("wAgentRelHi", sg(max(rel) * 100))

# ---- the frontier agents' calls, from the harness's own log (failed forward requests do not count)
au = J("agent_audit.json")["agents"]
for nm, tag in (("Claude Opus 5.5", "Claude"), ("gpt-6.1-sol", "Gpt")):
    t = au[nm]["tool_calls"]
    put(f"wAgent{tag}Runs", str(t["python"]))
    put(f"wAgent{tag}FwdFailed", str(t["forward_failed"]))
    assert t["passes_used"] == t["pass_budget"]
put("wAgentFwdBudget", str(au["Claude Opus 5.5"]["tool_calls"]["pass_budget"]))

# ---- deferral to Qwen3.5-9B on MedQA
dc = J("defer_control.json")["triggers"]["loop"]
put("wDefer", sg(dc["minus_random"][0] * 100))
put("wDeferCI", ci(dc["minus_random"][1], dc["minus_random"][2]))
put("wDeferRate", fx(dc['defer_rate'] * 100))

# ---- appendix: MMLU-Pro math, released counts and accuracy by fix
for k, tag in (("MMLU-Pro math, agent's edit", "Edit"), ("MMLU-Pro math, thinking off, full budget", "Off")):
    put(f"wMmluRel{tag}N", f"{hd[k]['released']:,}".replace(",", "{,}"))
    put(f"wMmluRel{tag}Acc", f1(hd[k]["released_accuracy"]))
on = [hd[f"MMLU-Pro math, {nm}"] for nm in ("loop stop", "budget forcing", "T = 1.0")]
put("wMmluRelOnLo", f"{min(v['released'] for v in on):,}".replace(",", "{,}"))
put("wMmluRelOnHi", f"{max(v['released'] for v in on):,}".replace(",", "{,}"))
put("wMmluRelOnAccLo", fx(min(v['released_accuracy'] for v in on) * 100, 0))
put("wMmluRelOnAccHi", fx(max(v['released_accuracy'] for v in on) * 100, 0))

# ---- appendix: the strength sweep of the agent's edit (200 practice problems, one sample)
cs = J("csweep.json")
put("wSweepBase", f1(cs["arms"]["base"]["acc"]))
for c, nm in (("c1", "COne"), ("c2", "CTwo"), ("c4", "CFour"), ("c6", "CSix"), ("c8", "CEight")):
    put(f"wSweep{nm}", f1(cs["arms"][c]["acc"]))
cl = J("decontam32k.json")["math500"]["clean"]
put("wCleanProblems", str(cl["problems"]))
put("wCleanLead", sg(cl["edit_minus_thinking_off_32k"]["gain"] * 100))
put("wCleanLeadCI", ci(*cl["edit_minus_thinking_off_32k"]["ci95"]))
put("wSweepTextbook", f1(cs["arms"]["textbook"]["acc"]))
fl = cs["flip_share"]
for L in ("15", "19"):
    put(f"wFlipFour{'A' if L == '15' else 'B'}", f1(fl[f"L{L}_looping"]["c4"]))
    put(f"wFlipSix{'A' if L == '15' else 'B'}", fx(fl[f'L{L}_looping']['c6'] * 100, 0))

# ---- appendix: deferral
d0 = J("defer_control.json")
put("wDeferAcc", f1(dc["acc"]))
put("wDeferRandomAcc", f1(dc["random_acc"]))
put("wNineB", f1(d0["strong_alone"]))
put("wSmallAlone", f1(d0["small_alone"]))

# ==== Paper, Section 4 (What the Edit Fixes) ====
import numpy as np  # noqa: E402
dcl = J("decontam32k.json")["math500"]
put("wCleanEditAcc", f1(dcl["clean"]["edit_acc"]))
put("wNearDup", str(dcl["all"]["problems"] - dcl["clean"]["problems"]))
for ds, tag in (("math500", "Math"), ("gsm8k", "Gsm")):
    T = {k: e3[ds][v] for k, v in ARM.items()}
    put(f"wOffOverEight{tag}", fx((T["off"]["acc"] - T["off8k"]["acc"]) * 100))
    put(f"wTokRatio{tag}", fx(T["edit"]["mean_tokens"] / T["off"]["mean_tokens"]))
    put(f"wEot{tag}", f1(T["eot"]["acc"]))
    put(f"wOffFin{tag}", fx((1 - T["off"]["cut"]) * 100, 0))
put("wPromptMath", f1(score("prompt/think35_pr2_math500")[0]))
lb6 = J("e6.json")["arms"]["base"]
put("wLongCutN", str(round(lb6["reached_32768"] * 500)))
put("wLongFinN", str(round((lb6["reached_32768"] - lb6["cut_81920"]) * 500)))
ab = J("abstain.json")["a"]
put("wOwnFinAccMath", f1(ab["MATH-500"]["arms"]["base"]["finished_accuracy"]))
put("wOwnFinAccMed", f1(ab["MedQA"]["arms"]["base"]["finished_accuracy"]))
put("wOwnFinAccHealth", f1(ab["MMLU-Pro health"]["arms"]["base"]["finished_accuracy"]))
g = J("part1b_grid.json")
cells = g["cells"]
fg = np.array([1 - v["cut"] for v in cells.values()]) * 100
ag = np.array([v["acc"] for v in cells.values()]) * 100
k_, b_ = np.polyfit(fg, ag, 1)
put("wGridCells", str(len(cells)))
put("wGridSlope", fx(k_, 3))
put("wGridIntercept", fx(b_))
put("wGridRtwo", fx(1 - ((ag - (k_ * fg + b_)) ** 2).sum() / ((ag - ag.mean()) ** 2).sum(), 3))
# released against own, over the five test sets and every fix
more = J("rescue_more.json")
SETS5 = ("MATH-500", "GSM8K", "MedQA", "MMLU-Pro health")
allw = {}
for ds in SETS5:
    A = dict(rs[ds]["arms"], **{k: v[ds] for k, v in more.items() if ds in v})
    for k, v in A.items():
        allw[(ds, k)] = v["within_problem"]
for k, v in hd.items():
    if k.startswith("MMLU-Pro math, "):
        allw[("MMLU-Pro math", k)] = v["within_problem"]
allw[("MMLU-Pro health", "meandiff")] = hd["MMLU-Pro health, medical-direction edit, 8 samples"]["within_problem"]  # the 8-sample run replaces the 4-sample one
edit_his = [w["hi"] for (ds, k), w in allw.items() if k in ("authors_edit", "MMLU-Pro math, agent's edit")]
put("wEditRelHiMax", sg(max(edit_his) * 100))
put("wFixRelMax", sg(max(w["diff"] for w in allw.values()) * 100))
assert max(w["lo"] for w in allw.values()) <= 0
on = [w["diff"] for (ds, k), w in allw.items() if ds in SETS5 and k in ("authors_edit", "meandiff", "t1", "bf8k", "loop", "prompt")]
put("wThinkOnRelLo", sg(min(on) * 100))
put("wThinkOnRelHi", sg(max(on) * 100))
worse = [-w["diff"] for (ds, k), w in allw.items() if ds in SETS5 and k in ("nothink", "nothink32k", "rep_penalty", "eot_bias")]
put("wWorseLo", fx(min(worse) * 100, 0))
put("wWorseHi", fx(max(worse) * 100, 0))
pov = J("prefix_overlap.json")["MATH-500"]
put("wPrefixMedian", fx(pov["arms"]["agent's edit"]["shared_prefix_tokens_q"][1], 0))
put("wLoopOnsetMedian", f"{pov['loop_onset_tokens_q'][1]:,.0f}".replace(",", "{,}"))
af = pov["arms"]["agent's edit"]["all_finished_minus_own"]["diff"]
put("wAllFinMinusOwn", sg(af[0] * 100))
put("wAllFinMinusOwnCI", ci(af[1], af[2]))
put("wAbsRelMinusOwnMmlu", fx(-h["within_problem"]["diff"] * 100))
# MMLU-Pro math reach against guessing, and released against loop stop (wsa.headroom_reach)
hr = J("headroom_reach.json")
put("wMmluNever", str(hr["never_finished"]))
put("wMmluQuestions", f"{hr['questions']:,}".replace(",", "{,}"))
put("wMmluEditCorrect", f"{hr['edit_correct_answers']:,}".replace(",", "{,}"))
put("wMmluOptions", fx(hr["options_mean"]))
c20 = hr["coverage"]["20"]
put("wMmluCov", f1(c20["edit"][0])); put("wMmluCovCI", ci(c20["edit"][1], c20["edit"][2]))
put("wMmluGuess", f1(c20["guessing"][0])); put("wMmluRandom", f1(c20["random_questions"][0]))
put("wMmluCorr", f1(c20["chance_corrected"][0])); put("wMmluCorrCI", ci(c20["chance_corrected"][1], c20["chance_corrected"][2]))
lo = hr["leave_one_fix_out"]
put("wMmluLooEditLo", f1(min(v["edit_chance_corrected"] for v in lo.values())))
put("wMmluLooEditHi", f1(max(v["edit_chance_corrected"] for v in lo.values())))
put("wMmluLooFixLo", f1(min(v["fix_chance_corrected"] for v in lo.values())))
put("wMmluLooFixHi", f1(max(v["fix_chance_corrected"] for v in lo.values())))
rl = hr["released_vs_loop_stop"]
put("wLoopEditAcc", f1(rl["all"]["edit_acc"])); put("wLoopAcc", f1(rl["all"]["loop_stop_acc"]))
put("wEditMinusLoop", sg(rl["all"]["edit_minus_loop_stop"][0] * 100))
put("wEditMinusLoopCI", ci(*rl["all"]["edit_minus_loop_stop"][1:]))
put("wLoopSamples", f"{rl['all']['samples']:,}".replace(",", "{,}")); put("wLoopQuestions", str(rl["all"]["questions"]))
put("wLoopSame", fx(rl["all"]["same_answer"][0] * 100, 0))
put("wEditMinusLoopNever", sg(rl["never_finished"]["edit_minus_loop_stop"][0] * 100))
put("wEditMinusLoopNeverCI", ci(*rl["never_finished"]["edit_minus_loop_stop"][1:]))
# within reach
cvj = J("coverage.json")
for ds, tag in (("MATH-500", "Math"), ("GSM8K", "Gsm")):
    v = cvj[ds]
    put(f"wReleasedAlwaysShare{tag}", fx(v["released_on_always_looping"] / v["released_answers"] * 100, 0))
    put(f"wEditSolved{tag}", str(cm[ds]["edit"]["solved"]))
    em = cm[ds]["references"]["Qwen3.5-2B, random"]["edit_minus_ref"]
    for k in ("32", "4"):
        put(f"wEditMinusTwoB{tag}K{'ThirtyTwo' if k == '32' else 'Four'}", sg(em[k][0] * 100))
        put(f"wEditMinusTwoB{tag}K{'ThirtyTwo' if k == '32' else 'Four'}CI", ci(em[k][1], em[k][2]))
    put(f"wCovTwoBDiff{tag}", f1(cm[ds]["references"]["Qwen3.5-2B, difficulty"]["coverage"]["32"][0]))
    sk = v["solved_k4_thinking_off_minus_edit"]
    put(f"wOffSolvedK{tag}", str(sk["thinking_off"])); put(f"wEditSolvedK{tag}", str(sk["edit"]))
    put(f"wOffMinusEditSolved{tag}", (f"+{sk['diff']}" if sk["diff"] >= 0 else str(sk["diff"])).replace("-", "$-$").replace("+", "$+$"))
    put(f"wOffMinusEditSolved{tag}CI", f"[{int(sk['ci'][0])}, {int(sk['ci'][1])}]".replace("-", "$-$"))
    st = v["strict_accuracy_per_sample_on_always_looping"]
    put(f"wEditStrictAlways{tag}", f1(st["authors_edit"])); put(f"wOffStrictAlways{tag}", f1(st["nothink32k"]))
    lo_ = cn[ds]["left_out"]
    emf = [w["edit_minus_fix"][0] for kk, w in lo_.items() if kk != "thinking off (both budgets)"]
    put(f"wEditMinusFixLo{tag}", sg(min(emf) * 100)); put(f"wEditMinusFixHi{tag}", sg(max(emf) * 100))
    f2b = [w["fix_minus_2b"][0] for kk, w in lo_.items() if kk != "thinking off (both budgets)"]
    put(f"wFixMinusTwoBLo{tag}", f1(min(f2b))); put(f"wFixMinusTwoBHi{tag}", f1(max(f2b)))
# matched completion: every fix's error gap to the edit at the lower completion rate (Table riskcov)
for ds, tag in (("MATH-500", "Math"), ("GSM8K", "Gsm")):
    for k, nm in (("t1", "Tone"), ("bf8k", "Bf"), ("loop", "Loop"), ("eot_bias", "Eot"), ("rep_penalty", "Rep"),
                  ("prompt", "Prompt"), ("meandiff", "Textbook"), ("nothink8k", "OffEight")):
        put(f"wRisk{nm}{tag}", sg(rk[ds]["vs_edit"][k]["error_gap_at_common_coverage"][0] * 100))
pr_ = ab["MATH-500"]["pairs"]
put("wExtraBeyondBf", fx(pr_["bf8k"]["marginal_acc_edit_beyond_g"] * 100, 0))
put("wExtraBeyondBfCI", f"[{fx(pr_['bf8k']['ci'][0] * 100, 0)}, {fx(pr_['bf8k']['ci'][1] * 100, 0)}]")
put("wFixExtraLo", fx(min(pr_[k]["marginal_acc_g_beyond_base"] for k in ("bf8k", "loop")) * 100, 0))
put("wFixExtraHi", fx(max(pr_[k]["marginal_acc_g_beyond_base"] for k in ("bf8k", "loop")) * 100, 0))

# ==== Paper, Section 5 (medicine) ====
pt, pw = p2["test"], p2["pairwise"]
mw = J("part2_mmlu_health.json")["pairwise"]
put("wMedEditGain", sg(pt["med_edit/think35"]["vs_base_official"]["gain"] * 100))
put("wMedEditGainStrict", sg(pt["med_edit/think35"]["vs_base_strict"]["gain"] * 100))
put("wMedToneGain", sg(pt["med_base/card_think"]["vs_base_official"]["gain"] * 100))
put("wMedToneGainStrict", sg(pt["med_base/card_think"]["vs_base_strict"]["gain"] * 100))
put("wMedMdGain", sg(pt["med_md_a6/think35"]["vs_base_official"]["gain"] * 100))
put("wMedMdGainStrict", sg(pt["med_md_a6/think35"]["vs_base_strict"]["gain"] * 100))
for src, pre, tag in ((pw, "med_edit/think35 - med_base/card_think", "MedEditMinusToneS"), (mw, "mmlu_edit/think35 - mmlu_base/card_think", "HealthEditMinusTone"),
                      (pw, "med_edit/think35 - med_md_a6/think35", "MedEditMinusMd")):
    for kind, suf in (("official", ""), ("strict", "Strict")):
        if tag == "MedEditMinusToneS" and kind == "official":
            continue                                        # \wMedEditMinusTone above
        v = src[f"{pre} ({kind})"]
        name = ("wMedEditMinusToneStrict" if tag == "MedEditMinusToneS" else f"w{tag}{suf}")
        put(name, sg(v["gain"] * 100)); put(name + "CI", ci(*v["ci95"]))
med_on = [allw[(ds, k)]["diff"] for ds in ("MedQA", "MMLU-Pro health") for k in ("authors_edit", "meandiff", "t1", "bf8k", "loop", "prompt")
          if (ds, k) in allw]
put("wMedOnMaxAbs", fx(max(abs(x) for x in med_on) * 100))
md8 = hd["MMLU-Pro health, medical-direction edit, 8 samples"]["within_problem"]
put("wMdHealthEight", sg(md8["diff"] * 100)); put("wMdHealthEightCI", ci(md8["lo"], md8["hi"]))
racc = {ds: [rs[ds]["arms"][k]["rescued_accuracy"] for k in ("t1", "bf8k", "loop")] for ds in ("MATH-500", "MedQA")}
put("wRelOnMedLo", fx(min(racc["MedQA"]) * 100, 0)); put("wRelOnMedHi", fx(max(racc["MedQA"]) * 100, 0))
put("wRelOnMathLo", fx(min(racc["MATH-500"]) * 100, 0)); put("wRelOnMathHi", fx(max(racc["MATH-500"]) * 100, 0))
lp = J("abstain.json")["b"]["loops"]
put("wToneBreakMed", f1(lp["MedQA"]["t1_breaks_share"])); put("wToneBreakMath", f1(lp["MATH-500"]["t1_breaks_share"]))
on_ = [lp[d]["onset_tokens_q"][1] for d in ("MATH-500", "GSM8K", "MedQA", "MMLU-Pro health")]
put("wOnsetLo", f"{min(on_):,.0f}".replace(",", "{,}")); put("wOnsetHi", f"{max(on_):,.0f}".replace(",", "{,}"))
br = J("abstain.json")["b"]["t1_break_by_period_words"]
put("wBreakShortMed", fx(br["MedQA"]["1-15"][1] * 100, 0)); put("wBreakLongMed", fx(br["MedQA"]["101-200"][1] * 100, 0))
put("wBreakShortMath", fx(br["MATH-500"]["1-15"][1] * 100, 0)); put("wBreakLongMath", fx(br["MATH-500"]["101-200"][1] * 100, 0))

# ==== Paper, Section 6 (the agent's search) ====
sv = J("svd_edit.json")["two_edits"]["known_directions"]
put("wCosFifteen", fx(sv["meandiff_u15"]["cos_same_layer"], 2)); put("wCosNineteen", fx(sv["meandiff_u19"]["cos_same_layer"], 2))
put("wCosRandP", fx(max(sv["meandiff_u15"]["random_abs_cos_p99"], sv["meandiff_u19"]["random_abs_cos_p99"]), 2))
csa = cs["arms"]
put("wSearchGap", fx((csa["c6"]["acc"] - csa["textbook"]["acc"]) * 100))
put("wGridBest", f1(max(v["acc"] for v in cells.values())))
v46 = cs["vs"]["c4 - c6"]
put("wFourToSixAcc", sg(-v46["acc"]["d"] * 100)); put("wFourToSixAccCI", ci(-v46["acc"]["ci95"][1], -v46["acc"]["ci95"][0]))
put("wFourToSixFinish", str(round(v46["cut"]["d"] * cs["n_problems"])))
lw = J("runs/loopwrite/loop_write.json")["layers"]
put("wSweepGainOne", fx((csa["c1"]["acc"] - csa["base"]["acc"]) * 100)); put("wSweepGainTwo", fx((csa["c2"]["acc"] - csa["base"]["acc"]) * 100))
put("wWriteRatioA", fx(lw["15"]["looping"]["s"]["mean"] / lw["15"]["productive"]["s"]["mean"]))
put("wWriteRatioB", fx(lw["19"]["looping"]["s"]["mean"] / lw["19"]["productive"]["s"]["mean"]))

# ==== Paper, Section 7 (frontier agents) ====
for a, tag in (("openai_1", "Gpt"), ("claude_1", "Claude")):
    v = ae["MATH-500"][a]["vs"]["nothink32k"]
    put(f"wAgent{tag}BelowOff", fx(-v["diff"] * 100))
    put(f"wAgent{tag}MedGain", fx(ae["MedQA"][a]["vs"]["base"]["diff"] * 100))
    w = ae["GSM8K"][a]["within_problem"]
    put(f"wAgent{tag}RelGsmAbs", fx(-w["diff"] * 100)); put(f"wAgent{tag}RelGsmCI", ci(w["lo"], w["hi"]))
g_ = ae["GSM8K"]["openai_1"]["vs"]["nothink32k"]
put("wAgentGptOverOffGsm", sg(g_["diff"] * 100)); put("wAgentGptOverOffGsmCI", ci(g_["lo"], g_["hi"]))
put("wAgentRelHiMax", sg(max(ae[d][a]["within_problem"]["hi"] for d in ae for a in ae[d]) * 100))

# ---- appendix (app:harness): the frontier agents' runs and every interval of their edits
def dci(w):
    return f"{sg(w['diff'] * 100)} {ci(w['lo'], w['hi'])}"


us = J("agent_usage.json")
dc_ = J("agent_decontam.json")["clean"]
assert dc_["gpt-6.1-sol"]["problems"] == 500 - int(M["wNearDup"])
words = {1: "one", 2: "two", 3: "three", 4: "four"}
for a, nm, tag in (("openai_1", "gpt-6.1-sol", "Gpt"), ("claude_1", "Claude Opus 5.5", "Claude")):
    u = us[nm]
    put(f"wFr{tag}Turns", str(u["turns"]))
    put(f"wFr{tag}In", fx(u["input_tokens"] / 1e6) + "M")
    put(f"wFr{tag}Out", fx(u["output_tokens"] / 1e6) + "M")
    put(f"wFr{tag}Cache", fx(u["cache_read_share"] * 100, 0))
    put(f"wFr{tag}Tensors", words[len(u["tensors"])])
    for d, dt in (("MATH-500", "Math"), ("GSM8K", "Gsm"), ("MedQA", "Med")):
        v = ae[d][a]
        put(f"wFr{tag}Base{dt}", dci(v["vs"]["base"]))
        put(f"wFr{tag}Edit{dt}", dci(v["vs"]["edit"]))
        put(f"wFr{tag}Rel{dt}", dci(v["within_problem"]))
        if "nothink32k" in v["vs"]:
            put(f"wFr{tag}Off{dt}", dci(v["vs"]["nothink32k"]))
    put(f"wFr{tag}TOneMed", dci(ae["MedQA"][a]["vs"]["t1"]))
    put(f"wFr{tag}RelAccMed", f1(ae["MedQA"][a]["released_accuracy"]))
    c_ = dc_[nm]
    put(f"wFr{tag}CleanBase", f"{sg(c_['minus base']['gain'] * 100)} {ci(*c_['minus base']['ci95'])}")
    put(f"wFr{tag}CleanOff", f"{sg(c_['minus thinking off, full budget']['gain'] * 100)} "
                             f"{ci(*c_['minus thinking off, full budget']['ci95'])}")

# ==== Paper, Section 8 (other models) ====
from wsa.rescue import boot_ci  # noqa: E402
c2 = J("c2b.json")
put("wTwoBTextbookGsm", sg(c2["gsm8k"]["textbook edit (layer 15, strength 6)"]["vs_base"]["d"] * 100))
put("wTwoBTextbookMath", sg(c2["math500"]["textbook edit (layer 15, strength 6)"]["vs_base"]["d"] * 100))
put("wTwoBCutMed", f1(c2["medqa"]["base (think35)"]["cut"]))
put("wTwoBOffCostMed", fx(-c2["medqa"]["thinking off, 32k cap"]["vs_base"]["d"] * 100))


def pp_(stem):
    return {r["id"]: float(np.mean(r["correct"])) for r in json.load(open(R / "runs" / f"{stem}.per_problem.json"))}


def paired(a, b):                                   # a minus b, per problem, paired bootstrap over problems
    A, B = pp_(a), pp_(b)
    return boot_ci([A[i] - B[i] for i in sorted(set(A) & set(B))])


d1 = paired("c2b/base/nothink35_32k_math500", "c2b/edit_l15_top64_a6/think35_math500")
put("wTwoBOffOverTextbook", fx(d1[0] * 100)); put("wTwoBOffOverTextbookCI", ci(d1[1], d1[2]))
d2 = paired("c2b/base/think35_loop_medqa", "c2b/edit_l15_top64_a6/think35_medqa")
put("wTwoBLoopMinusTextbookMed", sg(d2[0] * 100)); put("wTwoBLoopMinusTextbookMedCI", ci(d2[1], d2[2]))
d3 = paired("c2b/edit_l15_top64_a6/think35_gsm8k", "c2b/base/think35_loop_gsm8k")
put("wTwoBTextbookOverLoopGsm", fx(d3[0] * 100)); put("wTwoBTextbookOverLoopGsmCI", ci(d3[1], d3[2]))
ph = J("part4_screen_hard.json")
put("wLargeCut", str(sum(v["n_cut"] for m in ("Qwen3.5-4B", "Qwen3.5-9B") for v in ph[m].values())))
put("wLargeLoop", str(sum(v["cut_looping"] for m in ("Qwen3.5-4B", "Qwen3.5-9B") for v in ph[m].values())))
put("wRoneLoopLFive", fx(J("lowtemp.json")["r1d_qwen15b/r1d_t0"]["true_loop"] * 100, 0))
r1 = J("r1d.json")["paired"]
put("wRoneEdit", sg(r1["edit T 0 - base T 0"]["d"] * 100)); put("wRoneEditCI", ci(*r1["edit T 0 - base T 0"]["ci95"]))
put("wRoneCard", sg(r1["base T 0.6 - base T 0"]["d"] * 100)); put("wRoneCardCI", ci(*r1["base T 0.6 - base T 0"]["ci95"]))

# ==== Paper, Section 9 (deployment) ====
put("wDeferMatchEditRate", f1(d0["random_rate_to_match_edit"]))
put("wDeferMinusEdit", sg(dc["minus_edit"][0] * 100)); put("wDeferMinusEditCI", ci(dc["minus_edit"][1], dc["minus_edit"][2]))
put("wEditMedOne", f1(d0["agents_edit"]))
mt = J("route_vs_fix.json")["domains"]["medqa_full"]["strong"]["9B"]["matched"]["loop"]
put("wLoopTokensBefore", f"{mt['tokens_before'][0]:,.0f}".replace(",", "{,}")); put("wLenTokensBefore", f"{mt['tokens_before'][1]:,.0f}".replace(",", "{,}"))
put("wLoopEarlier", fx((1 - mt["tokens_before"][0] / mt["tokens_before"][1]) * 100, 0))
put("wLoopMatchedAcc", f1(mt["acc"][0])); put("wLenMatchedAcc", f1(mt["acc"][1]))

# ==== Paper: abstract, introduction, limitations ====
put("wDeferAbs", fx(dc["minus_random"][0] * 100))
cr = J("coverage_ref.json")
put("wTwoBReliableMath", f1(cr["MATH-500"]["targets"]["Qwen3.5-2B, matched"]["per_answer"][0]))
put("wTwoBReliableGsm", f1(cr["GSM8K"]["targets"]["Qwen3.5-2B, matched"]["per_answer"][0]))

# ---- Table 1 (tab:main): accuracy, answers finished and mean tokens on MATH-500 and GSM8K; the 81,920-token row
EXTRA = {"prompt": "prompt/think35_pr2_{}", "rep": "base/think35_rep11_{}"}
ROWS = (("Base", "base"), ("Prompt only", "prompt"), ("$T = 1.0$", "t1"), ("Repetition penalty", "rep"),
        ("Budget forcing", "bf"), ("Loop stop", "loop"), ("Thinking off, 8k", "off8k"), ("End-of-thinking bias", "eot"),
        ("Thinking off, full budget", "off"), ("Textbook edit", "textbook"), ("Agent's edit", "edit"))


def row_vals(k, ds):
    if k in EXTRA:
        return score(EXTRA[k].format(ds))
    v = e3[ds][ARM[k]]
    return v["acc"], 1 - v["cut"], v["mean_tokens"]


best = {ds: max((row_vals(k, ds)[0] for _, k in ROWS)) for ds in ("math500", "gsm8k")}
A = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.6pt}", r"\begin{tabular}{@{}lrrrrrr@{}}",
     r"\toprule", r"& \multicolumn{3}{c}{MATH-500} & \multicolumn{3}{c}{GSM8K} \\", r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
     r"& Acc & Fin & Tok & Acc & Fin & Tok \\", r"\midrule"]
for name, k in ROWS:
    cells = []
    for ds in ("math500", "gsm8k"):
        acc, fin, tok = row_vals(k, ds)
        cells += [rf"\textbf{{{f1(acc)}}}" if acc == best[ds] else f1(acc), f1(fin), fx(tok / 1000)]
    A.append(name + " & " + " & ".join(cells) + r" \\")
lb = J("e6.json")["arms"]["base"]
A += [r"\midrule", "Base, 81{,}920 tokens & " + f"{f1(lb['acc_81920'])} & {f1(1 - lb['cut_81920'])} & {fx(lb['mean_tokens'] / 1000)}"
      + r" & -- & -- & -- \\", r"\bottomrule", r"\end{tabular}",
      r"\caption{Qwen3.5-0.8B on the test sets (4 samples per problem; 81{,}920-token row: first sample). Acc: accuracy,",
      r"authors' scoring (\%); Fin: answers finished (\%), forced answers included; Tok: mean generated tokens (thousands).}",
      r"\label{tab:main}", r"\end{table}"]
(OUT / "tables").mkdir(exist_ok=True)
(OUT / "tables" / "main.tex").write_text("\n".join(A) + "\n")

# ---- Table 2 (tab:med): MedQA and MMLU-Pro health, strict accuracy, authors' scoring, answers finished
mh = J("part2_mmlu_health.json")["test"]
pq = J("prompt_only.json")["sets"]["MedQA"]["arms"]["prompt"]
MED = (("Base", "med_base/think35", "mmlu_base/think35"), ("Prompt only", "prompt", None),
       ("Thinking off, 8k", "med_base/nothink35", "mmlu_base/nothink35"), ("$T = 1.0$", "med_base/card_think", "mmlu_base/card_think"),
       ("Budget forcing", "med_base/think35_bf8192", "mmlu_base/think35_bf8192"),
       ("Loop stop", "med_base/think35_loop", "mmlu_base/think35_loop"),
       ("Medical-direction edit", "med_md_a6/think35", "mmlu_md_a6/think35"), ("Agent's edit", "med_edit/think35", "mmlu_edit/think35"))
get = lambda src, k: (pq if k == "prompt" else src[k]) if k else None  # noqa: E731
cols = {}
for c, src in ((1, p2["test"]), (2, mh)):
    vals = [get(src, r[c]) for r in MED]
    cols[c] = {"strict": max(v["strict"] for v in vals if v), "official": max(v["official"] for v in vals if v)}
T2 = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}lrrrrrr@{}}",
      r"\toprule", r"& \multicolumn{3}{c}{MedQA} & \multicolumn{3}{c}{MMLU-Pro h.} \\", r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
      r"& Strict & Acc & Fin & Strict & Acc & Fin \\", r"\midrule"]
for name, a_, b_ in MED:
    cells = []
    for c, src, k in ((1, p2["test"], a_), (2, mh, b_)):
        v = get(src, k)
        if v is None:
            cells += ["--"] * 3
            continue
        bold = lambda key: (rf"\textbf{{{f1(v[key])}}}" if v[key] == cols[c][key] else f1(v[key]))  # noqa: E731
        cells += [bold("strict"), bold("official"), f1(1 - v["cut"])]
    T2.append(name + " & " + " & ".join(cells) + r" \\")
T2 += [r"\bottomrule", r"\end{tabular}",
       r"\caption{Qwen3.5-0.8B on medical QA (4 samples per question; \%). Strict counts cut answers as wrong. The agent's edit",
       r"is transferred unchanged from mathematics.}", r"\label{tab:med}", r"\end{table}"]
(OUT / "tables" / "med.tex").write_text("\n".join(T2) + "\n")

# ---- Table (tab:frontier): the frontier agents' edits
base_acc = {"MATH-500": e3["math500"][ARM["base"]]["acc"], "GSM8K": e3["gsm8k"][ARM["base"]]["acc"],
            "MedQA": p2["test"]["med_base/think35"]["official"]}
edit_acc = {"MATH-500": e3["math500"][ARM["edit"]]["acc"], "GSM8K": e3["gsm8k"][ARM["edit"]]["acc"],
            "MedQA": p2["test"]["med_edit/think35"]["official"]}
off_acc = {"MATH-500": e3["math500"][ARM["off"]]["acc"], "GSM8K": e3["gsm8k"][ARM["off"]]["acc"], "MedQA": None}
rows = [("Base", base_acc), ("Thinking off, full budget", off_acc), ("Published agent's edit", edit_acc),
        ("gpt-6.1-sol: layer 15 MLP", {d: ae[d]["openai_1"]["acc"] for d in ae}),
        ("Claude: 4 attention layers", {d: ae[d]["claude_1"]["acc"] for d in ae})]
top = {d: max(v[d] for _, v in rows if v[d] is not None) for d in ae}
F = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
     r"& MATH-500 & GSM8K & MedQA \\", r"\midrule"]
for nm, v in rows:
    F.append(nm + " & " + " & ".join("--" if v[d] is None else (rf"\textbf{{{f1(v[d])}}}" if v[d] == top[d] else f1(v[d]))
                                     for d in ae) + r" \\")
F += [r"\midrule", r"\multicolumn{4}{@{}l}{\emph{Released $-$ own, same problems (points)}} \\"]
for a, nm in (("openai_1", "gpt-6.1-sol"), ("claude_1", "Claude")):
    F.append(nm + " & " + " & ".join(sg(ae[d][a]["within_problem"]["diff"] * 100) for d in ae) + r" \\")
F += [r"\bottomrule", r"\end{tabular}",
      r"\caption{Edits found by two frontier agents redoing the task (Qwen3.5-0.8B, test sets, 4 samples per problem; accuracy",
      rf"\%). Their edits cut {f1(ae['MATH-500']['openai_1']['cut_share'])}\% and {f1(ae['MATH-500']['claude_1']['cut_share'])}\% of MATH-500 answers. Intervals below.}}",
      r"\label{tab:frontier}", r"\end{table}"]
(OUT / "tables" / "frontier.tex").write_text("\n".join(F) + "\n")

# ---- appendix A (protocol) and B (released answers): reproduction, the long budget, net accuracy, shared seeds, voting
E = e3["math500"][ARM["edit"]]
put("wEditCutPctMath", f1(E["cut"]))
ea = J("e6.json")["arms"]
put("wLongBaseAcc", f1(ea["base"]["acc_81920"])); put("wLongBaseCutAcc", f1(ea["base"]["acc_cut_at_32768"]))
put("wLongEditAcc", f1(ea["edit"]["acc_81920"]))
eo = ea["edit"]["of_those_reaching_32768"]
put("wLongEditFin", str(eo["finished_by_81920"])); put("wLongEditReach", str(eo["n"])); put("wLongEditRight", str(eo["right_at_81920"]))
put("wLongEditGain", sg(e6p["edit: 81,920 - cut at 32,768"]["d"] * 100)); put("wLongEditGainCI", ci(*e6p["edit: 81,920 - cut at 32,768"]["ci95"]))
put("wLongNoAnswer", str(J("e6.json")["edit_4_samples"]["no_answer_finished"]))
nxm = J("net_extra.json")["MedQA"]["net_beyond_base"]
put("wNetMedOnLo", fx(min(nxm["bf8k"]["value"], nxm["loop"]["value"]) * 100, 0))
assert fx(nxm["bf8k"]["value"] * 100, 0) == fx(nxm["loop"]["value"] * 100, 0)
for k, tag in (("authors_edit", "Edit"), ("t1", "TOne")):
    put(f"wNetMed{tag}", f1(nxm[k]["value"])); put(f"wNetMed{tag}CI", ci(*nxm[k]["ci"]))
put("wNetMedOwn", f1(J("net_extra.json")["MedQA"]["on_base_finished"]["base"]))
tb = rs["MedQA"]["2b_accuracy_on_stuck_vs_other"]
put("wTwoBStuck", f1(tb["stuck"])); put("wTwoBOther", f1(tb["other"]))
put("wMedOffCost", fx((p2["test"]["med_base/think35"]["official"] - p2["test"]["med_base/nothink35"]["official"]) * 100))
pm = J("prompt_only.json")["sets"]["MedQA"]["arms"]["prompt"]
put("wMedPromptCut", f1(pm["cut"])); put("wMedPromptCost", fx(-pm["gain_official"]["gain"] * 100))
PO = J("prefix_overlap.json")
for ds, tag in (("MATH-500", "Math"), ("GSM8K", "Gsm")):
    A = PO[ds]["arms"]
    for nm, k in (("agent's edit", "Edit"), ("textbook edit", "Textbook"), ("T = 1.0", "TOne"), ("repetition penalty", "Rep"),
                  ("prompt", "Prompt"), ("end-of-thinking bias", "Eot"), ("thinking off, 8k", "Off")):
        put(f"wPre{k}{tag}", fx(A[nm]["shared_prefix_tokens_q"][1], 0))
    assert A["thinking off, full"]["shared_prefix_tokens_q"][1] == 0
    put(f"wPreOnset{tag}", f"{A["agent's edit"]['released_with_onset']:,}".replace(",", "{,}"))
    assert A["agent's edit"]["shared_past_loop_onset"] == 0
pm_ = PO["MATH-500"]["arms"]
put("wPreBf", f"{pm_['budget forcing']['shared_prefix_tokens_q'][1]:,.0f}".replace(",", "{,}"))
put("wPreLoop", f"{pm_['loop stop']['shared_prefix_tokens_q'][1]:,.0f}".replace(",", "{,}"))
ag_ = PO["GSM8K"]["arms"]["agent's edit"]["all_finished_minus_own"]["diff"]
put("wAllFinMinusOwnGsm", sg(ag_[0] * 100)); put("wAllFinMinusOwnGsmCI", ci(ag_[1], ag_[2]))
v3 = mk["MATH-500"]["edit_minus_thinking_off_majj"]["3"]
put("wMajMathThree", sg(v3["diff"][0] * 100)); put("wMajMathThreeCI", ci(v3["diff"][1], v3["diff"][2]))
put("wMajMathThreeTok", fx(v3["thinking_off_tokens"] / 1000) + "k")

# ---- appendix C: the prompt on the test sets, and how often its cut answers still repeat
PR = J("prompt_only.json")["sets"]
g = {ds: PR[ds]["arms"]["prompt"]["gain_official"] for ds in PR}
put("wPromptGainMath", sg(g["MATH-500"]["gain"] * 100)); put("wPromptGainMathCI", ci(g["MATH-500"]["lo"], g["MATH-500"]["hi"]))
put("wPromptGainGsm", sg(g["GSM8K"]["gain"] * 100))
put("wPromptGainMed", sg(g["MedQA"]["gain"] * 100)); put("wPromptGainMedCI", ci(g["MedQA"]["lo"], g["MedQA"]["hi"]))
pl = [PR[ds]["arms"]["prompt"]["cut_looping"] * 100 for ds in PR]
bl_ = [PR[ds]["base"]["cut_looping"] * 100 for ds in PR]
put("wPromptRepeatLo", fx(min(pl), 0)); put("wPromptRepeatHi", fx(max(pl), 0))
put("wBaseRepeatLo", fx(min(bl_), 0)); put("wBaseRepeatHi", fx(max(bl_), 0))

# ---- appendix: R1-Distill's released answers against the recommended temperature on the same problems
rr = J("r1d.json")["released"]
put("wRoneN", str(rr["edit_on_released"]["n"])); put("wRoneRel", f1(rr["edit_on_released"]["mean"]))
put("wRoneCardRel", f1(rr["card_on_released"]["mean"])); put("wRoneOwn", f1(rr["base_on_its_finished"]["mean"]))
put("wRoneRelGap", sg(rr["edit_minus_card_on_released"]["d"] * 100)); put("wRoneRelGapCI", ci(*rr["edit_minus_card_on_released"]["ci95"]))

# ---- Table (tab:pair): released against own answers on the same problems, every fix and test set
def rel_cell(w):
    return f"{sg(w['diff'] * 100)} {ci(w['lo'], w['hi'])}"


MM = {"t1": "T = 1.0", "bf8k": "budget forcing", "loop": "loop stop", "nothink32k": "thinking off, full budget",
      "nothink": "thinking off, 8k"}
pw = {(ds, k): w for (ds, k), w in allw.items()}
for k, nm in MM.items():
    pw[("MMLU-Pro math", k)] = hd[f"MMLU-Pro math, {nm}"]["within_problem"]
pw[("MMLU-Pro math", "authors_edit")] = h["within_problem"]


def own_pair(ds):
    w = pw[(ds, "authors_edit")]
    return f"{f1(w.get('rescued_acc', w.get('released_acc')))} / {f1(w.get('base_finished_acc', w.get('own_acc')))}"


def pair_block(cols, rows, star=()):
    out = [r"Agent's edit: released / own (\%) & " + " & ".join(own_pair(d) for d in cols) + " &" * (3 - len(cols)) + r" \\",
           r"\multicolumn{4}{@{}l}{\emph{Released $-$ own, same problems (points, 95\% CI)}} \\"]
    used = []
    for nm, k in rows:
        cells = []
        for d in cols:
            w = pw.get((d, k))
            cells.append("--" if w is None else rel_cell(w) + (r"$^{*}$" if (d, k) in star else ""))
            if w is not None:
                used.append((d, w["problems"]))
        out.append(nm + " & " + " & ".join(cells) + " &" * (3 - len(cols)) + r" \\")
    rng_ = [f"{min(n for d_, n in used if d_ == d)}--{max(n for d_, n in used if d_ == d)}" for d in cols]
    out.append("Problems paired & " + " & ".join(rng_) + " &" * (3 - len(cols)) + r" \\")
    return out


r1 = J("r1d.json")["released"]
P = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.0pt}", r"\begin{tabular}{@{}llll@{}}",
     r"\toprule", r"& MATH-500 & GSM8K & MMLU-Pro math \\", r"\midrule"]
P += pair_block(("MATH-500", "GSM8K", "MMLU-Pro math"),
                (("Agent's edit", "authors_edit"), ("$T = 1.0$", "t1"), ("Budget forcing", "bf8k"), ("Loop stop", "loop"),
                 ("Textbook edit", "meandiff"), ("Prompt only", "prompt"), ("End-of-thinking bias", "eot_bias"),
                 ("Repetition penalty", "rep_penalty"), ("Thinking off, full budget", "nothink32k"),
                 ("Thinking off, 8k", "nothink")))
em = r1["edit_minus_card_on_released"]
P.insert(-1, r"R1-Distill: textbook edit $-$ $T{=}0.6$ & " + f"{sg(em['d'] * 100)} {ci(*em['ci95'])}" + r" & -- & -- \\")
P += [r"\midrule", r"& MedQA & MMLU-Pro health & \\", r"\midrule"]
P += pair_block(("MedQA", "MMLU-Pro health"),
                (("Agent's edit", "authors_edit"), ("$T = 1.0$", "t1"), ("Budget forcing", "bf8k"), ("Loop stop", "loop"),
                 ("Medical-direction edit", "meandiff"), ("Prompt only", "prompt"), ("Thinking off, 8k", "nothink")),
                star={("MMLU-Pro health", "meandiff")})
P += [r"\bottomrule", r"\end{tabular}",
      r"\caption{Released answers against the model's own on the same problems (Qwen3.5-0.8B; problems on which the base model",
      rf"finishes some samples). R1-Distill: its textbook edit's released answers against $T=0.6$ on the same {r1['edit_on_released']['n']} problems",
      rf"({f1(r1['edit_on_released']['mean'])}\% against {f1(r1['card_on_released']['mean'])}\%). $^{{*}}$Eight samples per question.}}",
      r"\label{tab:pair}", r"\end{table*}"]
(OUT / "tables" / "pair.tex").write_text("\n".join(P) + "\n")

# ---- Table (tab:ledger): what each fix releases (accuracy of released answers; net accuracy of the extra answers)
nx = J("net_extra.json")


def rel_acc(ds, k):             # released accuracy, recomputed from the count of released answers
    A = dict(rs[ds]["arms"], **{kk: v[ds] for kk, v in more.items() if ds in v})
    if k not in A:
        return None
    n = A[k]["rescued"]
    return round(A[k]["rescued_accuracy"] * n) / n


LD = ("MATH-500", "MedQA", "MMLU-Pro health")
Lg = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.8pt}", r"\begin{tabular}{@{}lrrrrrr@{}}",
      r"\toprule", r"& \multicolumn{2}{c}{MATH-500} & \multicolumn{2}{c}{MedQA} & \multicolumn{2}{c}{MMLU-Pro h.} \\",
      r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}", r"& Rel. & Net & Rel. & Net & Rel. & Net \\", r"\midrule",
      "Base, own finished & " + " & ".join(f"{f1(nx[d]['on_base_finished']['base'])} & " for d in LD) + r" \\", r"\midrule"]
for nm, k, dsets in (("$T = 1.0$", "t1", LD), ("Budget forcing", "bf8k", LD), ("Loop stop", "loop", LD),
                     ("Textbook edit", "meandiff", ("MATH-500",)),
                     (r"\begin{tabular}[t]{@{}l@{}}Medical-direction\\\quad edit\end{tabular}", "meandiff", ("MedQA", "MMLU-Pro health")),
                     ("Agent's edit", "authors_edit", LD), ("Thinking off, 8k", "nothink", LD),
                     ("Thinking off, full", "nothink32k", ("MATH-500",))):
    cells = []
    for d in LD:
        if d in dsets:
            cells += [f1(rel_acc(d, k)), num(nx[d]["net_beyond_base"][k]["value"] * 100)]
        else:
            cells += ["--", "--"]
    Lg.append(nm + " & " + " & ".join(cells) + r" \\")
Lg += [r"\bottomrule", r"\end{tabular}",
       r"\caption{What each fix releases (Qwen3.5-0.8B, \%). Rel.: accuracy of released answers; Net: net accuracy of extra",
       r"finished answers beyond the base model. First row: the base model's accuracy on its own finished answers.}",
       r"\label{tab:ledger}", r"\end{table}"]
(OUT / "tables" / "ledger.tex").write_text("\n".join(Lg) + "\n")

# ---- Table (tab:levels): MATH-500 by difficulty level
bl = J("abstain.json")["b"]["by_level"]
LV = ("1", "2", "3", "4", "5")
Lv = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.5pt}", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
      r"MATH level & 1 & 2 & 3 & 4 & 5 \\", r"\midrule", "Problems & " + " & ".join(str(bl["base"][l]["problems"]) for l in LV) + r" \\",
      r"\midrule", r"\multicolumn{6}{@{}l}{\emph{Cut (\%)}} \\"]
for nm, k in (("Base", "base"), ("Loop stop", "loop"), ("Agent's edit", "authors_edit")):
    Lv.append(nm + " & " + " & ".join(fx(bl[k][l]["cut"] * 100, 0) for l in LV) + r" \\")
Lv += [r"\midrule", r"\multicolumn{6}{@{}l}{\emph{Finished answers right (\%)}} \\"]
for nm, k in (("Base", "base"), ("Loop stop", "loop"), ("Agent's edit", "authors_edit")):
    Lv.append(nm + " & " + " & ".join(fx(bl[k][l]["finished_accuracy"] * 100, 0) for l in LV) + r" \\")
Lv += [r"\bottomrule", r"\end{tabular}",
       r"\caption{MATH-500 by difficulty level (Qwen3.5-0.8B, 4 samples per problem). The edit finishes far more hard problems,",
       r"and those answers are the ones that are often wrong.}", r"\label{tab:levels}", r"\end{table}"]
(OUT / "tables" / "levels.tex").write_text("\n".join(Lv) + "\n")

# ---- Table (tab:scale): other models
cb = J("c2b.json")
C2 = {"math500": "MATH-500", "gsm8k": "GSM8K", "medqa": "MedQA"}
srows = (("Textbook edit", "textbook edit (layer 15, strength 6)"), ("Thinking off, full budget", "thinking off, 32k cap"),
         ("Loop stop", "loop stop (forced answer)"), ("$T = 1.0$", "T 1.0 (model card)"))
top2 = {d: max(cb[d][k]["acc"] for _, k in (("", "base (think35)"),) + srows) for d in C2}


def bold(d, k):
    v = f1(cb[d][k]["acc"])
    return rf"\textbf{{{v}}}" if cb[d][k]["acc"] == top2[d] else v


Sc = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.0pt}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
      r"Qwen3.5-2B & MATH-500 & GSM8K & MedQA \\", r"\midrule",
      r"Base (cut \%) & " + " & ".join(f"{bold(d, 'base (think35)')} ({f1(cb[d]['base (think35)']['cut'])})" for d in C2) + r" \\"]
Sc += [nm + " & " + " & ".join(bold(d, k) for d in C2) + r" \\" for nm, k in srows]
ra = J("r1d.json")["arms"]
rtop = max(v["acc"] for v in ra.values())
rb = lambda k: rf"\textbf{{{f1(ra[k]['acc'])}}}" if ra[k]["acc"] == rtop else f1(ra[k]["acc"])  # noqa: E731
Sc += [r"\midrule", r"R1-Distill-1.5B & \multicolumn{3}{l}{MATH-500} \\", r"\midrule",
       rf"Base, $T=0$ (cut \%) & {rb('base T 0')} ({f1(ra['base T 0']['cut'])}) \\",
       rf"Textbook edit, $T=0$ & {rb('edit T 0')} \\", rf"Base, $T=0.6$ (card) & {rb('base T 0.6 (card)')} \\",
       r"\bottomrule", r"\end{tabular}",
       r"\caption{Other models (accuracy \%). Qwen3.5-2B: 4 samples per problem; each model's own textbook edit. R1-Distill: 1 sample per problem, greedy ($T=0$, where it loops) except at its recommended $T=0.6$.}",
       r"\label{tab:scale}", r"\end{table}"]
(OUT / "tables" / "scale.tex").write_text("\n".join(Sc) + "\n")

lines = ["% generated by wsa/paper_numbers.py from results/; do not edit"]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
(OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
print(f"{len(M)} macros -> paper/numbers.tex; tables -> paper/tables/{{main,med,frontier,pair,ledger,levels,scale}}.tex")
for k, v in M.items():
    print(f"  {k} = {v}")
