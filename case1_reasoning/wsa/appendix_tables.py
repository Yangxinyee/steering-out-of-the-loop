"""Appendix tables of the coverage-calibration and majority-vote analyses, written from the result files (no number typed by hand).

    python -m wsa.appendix_tables       (writes paper/tables/covref.tex and paper/tables/majk.tex)
"""
import json
import os
from pathlib import Path

FIX = {"t1": "$T = 1.0$", "bf8k": "Budget forcing", "loop": "Loop stop", "eot_bias": "End-of-thinking bias",
       "rep_penalty": "Repetition penalty", "prompt": "Prompt only", "nothink8k": "Thinking off, 8k",
       "nothink32k": "Thinking off, full budget", "thinking off (both budgets)": "Thinking off, both"}


def ci(v, nd=1):
    return f"{v[0] * 100:.{nd}f} [{v[1] * 100:.{nd}f}, {v[2] * 100:.{nd}f}]"


def covref():
    ref = json.load(open("results/coverage_ref.json"))
    cm = json.load(open("results/coverage_match.json"))
    nu = json.load(open("results/coverage_null.json"))
    sft = json.load(open("results/coverage_sft.json")) if os.path.exists("results/coverage_sft.json") else None
    sets = ("MATH-500", "GSM8K")
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.0pt}",
         r"\begin{tabular}{@{}lllllll@{}}", r"\toprule",
         r"& \multicolumn{3}{c}{MATH-500} & \multicolumn{3}{c}{GSM8K} \\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"\multicolumn{7}{@{}l}{\emph{Against the eight fixes (32 samples per problem)}} \\"]
    row = lambda name, vals: L.append(name + " & " + " & ".join(vals) + r" \\")
    row("Agent's edit", [ci(ref[d]["targets"]["agent's edit"]["per_answer"]) + " & & " for d in sets])
    row("Qwen3.5-2B, all it solves", [ci(ref[d]["targets"]["Qwen3.5-2B"]["per_answer"]) + " & & " for d in sets])
    row("Qwen3.5-2B, most reliable", [ci(ref[d]["targets"]["Qwen3.5-2B, matched"]["per_answer"]) + " & & " for d in sets])
    row("Qwen3.5-2B, random", [ci(cm[d]["references"]["Qwen3.5-2B, random"]["coverage"]["32"]) + " & & " for d in sets])
    row("Qwen3.5-2B, by difficulty", [ci(cm[d]["references"]["Qwen3.5-2B, difficulty"]["coverage"]["32"]) + " & & "
                                       for d in sets])
    if sft:
        row("Distilled 0.8B, matched", [ci(sft[d]["reference"]["coverage"]["32"]) + " & & " for d in sets])
    L[-1] = L[-1]
    L += [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{Each fix against the other seven: the fix's own answers, the edit's, "
          r"Qwen3.5-2B's}} \\", r"Left-out fix & Fix & Edit & 2B & Fix & Edit & 2B \\"]
    for k, name in FIX.items():
        vals = []
        for d in sets:
            t = nu[d]["left_out"][k]["targets"]
            vals += [ci(t["left-out fix"]["coverage"]), f"{t[chr(39).join(['agent', 's edit'])]['coverage'][0] * 100:.1f}",
                     f"{t['Qwen3.5-2B']['coverage'][0] * 100:.1f}"]
        row(name, vals)
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Calibrating coverage on always-looping problems (\%, 95\% intervals): the share of a target's correct "
          r"answers whose problem the unedited 0.8B also solves. Matched references keep as many solved problems as the "
          r"edit (most reliable: highest share of correct samples; random: 4{,}000 draws; by difficulty: within MATH "
          r"levels or GSM8K solution lengths). Thinking off, both: both budgets left out, against the six other fixes.}",
          r"\label{tab:covref}", r"\end{table*}"]
    # fix the empty-cell layout: each target row above spans one value per data set
    L = [l.replace(" & &  & ", " & & & ").replace(" & &  \\\\", " & & \\\\") for l in L]
    Path("paper/tables/covref.tex").write_text("\n".join(L) + "\n")


def majk():
    m = json.load(open("results/majk.json"))
    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}", r"\begin{tabular}{@{}lrrrr@{}}",
         r"\toprule", r"& maj@1 & maj@2 & maj@3 & maj@4 \\"]
    for d in ("MATH-500", "GSM8K"):
        L += [r"\midrule", r"\multicolumn{5}{@{}l}{\emph{" + d + r"}} \\"]
        for k, name in (("agent's edit", "Agent's edit"), ("thinking off, full", "Thinking off, full"),
                        ("thinking off, 8k", "Thinking off, 8k")):
            A = m[d]["arms"][k]
            L.append(name + " & " + " & ".join(f"{A[str(j)]['accuracy'] * 100:.1f} ({A[str(j)]['tokens'] / 1000:.1f})"
                                               for j in range(1, 5)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Majority vote over $j$ of the four samples (accuracy \%, authors' scoring; in parentheses, generated "
          r"tokens per problem, thousands). Ties are split evenly.}", r"\label{tab:majk}", r"\end{table*}"]
    Path("paper/tables/majk.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    covref()
    majk()
