"""Case 1 (the reasoning model's loop): every number the paper prints, read from case1_reasoning/results (override
with WSA_RESULTS). Writes paper/out/numbers_case1.tex (macros, prefix \\w) and the case-1 tables
(paper/out/tables/case1_*.tex). No number is typed by hand. Run case1_reasoning's table scripts first
(python -m wsa.paper_tables; python -m wsa.appendix_tables, from case1_reasoning/): their tables are copied in.

    python paper/case1_numbers.py
"""
import json
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = Path(os.environ.get("WSA_RESULTS", ROOT / "case1_reasoning" / "results"))
OUT = ROOT / "paper" / "out"


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

# ---- the case-1 table: accuracy and answers finished, MATH-500, GSM8K and MedQA
med = {"base": "med_base/think35", "t1": "med_base/card_think", "bf": "med_base/think35_bf8192", "loop": "med_base/think35_loop",
       "off8k": "med_base/nothink35", "edit": "med_edit/think35"}
ROWS = (("Unedited", "base"), ("$T = 1.0$", "t1"), ("Budget forcing", "bf"), ("Loop stop", "loop"),
        ("Thinking off, 8k", "off8k"), ("Thinking off, full budget", "off"), ("Textbook edit", "textbook"),
        ("Agent's edit", "edit"))
best = {ds: max(e3[ds][ARM[k]]["acc"] for _, k in ROWS) for ds in ("math500", "gsm8k")}
best_med = max(p2["test"][v]["official"] for v in med.values())
L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{@{}lrrrrr@{}}",
     r"\toprule", r"& \multicolumn{2}{c}{MATH-500} & \multicolumn{2}{c}{GSM8K} & MedQA \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-6}", r"& Acc & Fin & Acc & Fin & Acc \\", r"\midrule"]
for name, k in ROWS:
    cells = []
    for ds in ("math500", "gsm8k"):
        v = e3[ds][ARM[k]]
        a = f1(v["acc"])
        cells += [rf"\textbf{{{a}}}" if v["acc"] == best[ds] else a, f1(1 - v["cut"])]
    if k in med:
        m = p2["test"][med[k]]["official"]
        cells.append(rf"\textbf{{{f1(m)}}}" if m == best_med else f1(m))
    else:
        cells.append("--")
    L.append(name + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\caption{Qwen3.5-0.8B on the test sets (4 samples per problem; \%). Acc: accuracy, authors' scoring; Fin: answers "
      r"finished. Decoding fixes leave the weights unchanged.}", r"\label{tab:case1}", r"\end{table}"]
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "tables" / "case1_main.tex").write_text("\n".join(L) + "\n")

# ---- appendix: every setting on MATH-500 and GSM8K (accuracy, answers finished, mean tokens)
EXTRA = {"prompt": "prompt/think35_pr2_{}", "rep": "base/think35_rep11_{}"}
ALL = (("Unedited", "base"), ("Prompt only", "prompt"), ("$T = 1.0$", "t1"), ("Repetition penalty", "rep"),
       ("Budget forcing", "bf"), ("Loop stop", "loop"), ("End-of-thinking bias", "eot"), ("Thinking off, 8k", "off8k"),
       ("Thinking off, full budget", "off"), ("Textbook edit", "textbook"), ("Agent's edit", "edit"))
A = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrrrr@{}}",
     r"\toprule", r"& \multicolumn{3}{c}{MATH-500} & \multicolumn{3}{c}{GSM8K} \\", r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
     r"& Acc & Fin & Tok & Acc & Fin & Tok \\", r"\midrule"]
for name, k in ALL:
    cells = []
    for ds in ("math500", "gsm8k"):
        if k in EXTRA:
            acc, fin, tok = score(EXTRA[k].format(ds))
        else:
            v = e3[ds][ARM[k]]
            acc, fin, tok = v["acc"], 1 - v["cut"], v["mean_tokens"]
        cells += [f1(acc), f1(fin), fx(tok / 1000)]
    A.append(name + " & " + " & ".join(cells) + r" \\")
A += [r"\bottomrule", r"\end{tabular}", r"\caption{Every setting on MATH-500 and GSM8K (Qwen3.5-0.8B, 4 samples per "
      r"problem). Acc: accuracy, authors' scoring (\%); Fin: answers finished (\%); Tok: mean generated tokens "
      r"(thousands).}", r"\label{tab:case1-all}", r"\end{table}"]
(OUT / "tables" / "case1_all.tex").write_text("\n".join(A) + "\n")

# ---- appendix: the frontier agents' edits (accuracy, authors' scoring; released minus own, same problems)
F = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
     r"& MATH-500 & GSM8K & MedQA \\", r"\midrule"]
base_acc = {"MATH-500": e3["math500"][ARM["base"]]["acc"], "GSM8K": e3["gsm8k"][ARM["base"]]["acc"],
            "MedQA": p2["test"]["med_base/think35"]["official"]}
edit_acc = {"MATH-500": e3["math500"][ARM["edit"]]["acc"], "GSM8K": e3["gsm8k"][ARM["edit"]]["acc"],
            "MedQA": p2["test"]["med_edit/think35"]["official"]}
F.append("Unedited & " + " & ".join(f1(base_acc[d]) for d in ae) + r" \\")
F.append("Published agent's edit & " + " & ".join(f1(edit_acc[d]) for d in ae) + r" \\")
for a, nm in (("openai_1", "Codex"), ("claude_1", "Claude")):
    F.append(nm + " & " + " & ".join(f1(ae[d][a]["acc"]) for d in ae) + r" \\")
F += [r"\midrule", r"\multicolumn{4}{@{}l}{\emph{Released $-$ own, same problems (points, 95\% interval)}} \\"]
for a, nm in (("openai_1", "Codex"), ("claude_1", "Claude")):
    F.append(nm + " & " + " & ".join(sg(ae[d][a]["within_problem"]["diff"] * 100) + " " +
                                     ci(ae[d][a]["within_problem"]["lo"], ae[d][a]["within_problem"]["hi"]) for d in ae) + r" \\")
F += [r"\bottomrule", r"\end{tabular}", r"\caption{Edits found by two frontier agents redoing the task, Claude Opus 5.5 and Codex (gpt-6.1-sol) (Qwen3.5-0.8B, "
      r"4 samples per problem; accuracy \%, authors' scoring).}", r"\label{tab:frontier}", r"\end{table*}"]
(OUT / "tables" / "case1_frontier.tex").write_text("\n".join(F) + "\n")
# the audit's own generated appendix tables (wsa.paper_tables, wsa.round3_tables), copied unchanged
for t in ("coverage", "covref", "riskcov", "majk"):
    src = R.parent / "paper" / "tables" / f"{t}.tex"
    (OUT / "tables" / f"case1_{t}.tex").write_text(f"% copied from {src.relative_to(R.parent.parent)} (generated there); do not edit\n"
                                                   + src.read_text())

lines = ["% generated by paper/case1_numbers.py from case1_reasoning/results; do not edit"]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
(OUT / "numbers_case1.tex").write_text("\n".join(lines) + "\n")
print(f"{len(M)} macros -> paper/out/numbers_case1.tex; tables -> paper/out/tables/")
for k, v in M.items():
    print(f"  {k} = {v}")
