"""Appendix tables of the paper written from the result files (no number typed by hand).

    python -m wsa.paper_tables        (writes paper/tables/riskcov.tex and paper/tables/coverage.tex)
"""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


def r1(x, nd=1):
    """half-up rounding of the value as computed (7.55 -> 7.6), not of its binary float; never -0.0"""
    d = Decimal(f"{x:.10f}").quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return str(abs(d) if d == 0 else d)


def pm(x):
    v = r1(x * 100)
    return ("+" + v if not v.startswith("-") else v).replace("-", "$-$")


def ci(lo, hi):
    return f"[{r1(lo * 100)}, {r1(hi * 100)}]".replace("-", "$-$")


def main():
    rc = json.load(open("results/riskcov.json"))
    cov = json.load(open("results/coverage.json"))
    names = {"t1": "$T = 1.0$", "bf8k": "Budget forcing", "loop": "Loop stop", "eot_bias": "End-of-thinking bias",
             "rep_penalty": "Repetition penalty", "prompt": "Prompt only", "meandiff": "Textbook edit",
             "nothink8k": "Thinking off, 8k", "nothink32k": "Thinking off, full budget"}
    rows = []
    for k, n in names.items():
        cells = []
        for ds in ("MATH-500", "GSM8K"):
            v = rc[ds]["vs_edit"][k]
            at, m = v["error_gap_at_common_coverage"], v["mean_error_gap_over_common_range"]
            kept = round(v["common_coverage"] * 100 + 1e-9, 1)          # halves rounded up, as in Table 1
            cells.append(f"{r1(kept)} & {pm(at[0])} {ci(at[1], at[2])} & {pm(m[0])} {ci(m[1], m[2])}")
        rows.append(n + " & " + " & ".join(cells) + r" \\")
    t1 = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrllrll@{}}
\toprule
& \multicolumn{3}{c}{MATH-500} & \multicolumn{3}{c}{GSM8K} \\
\cmidrule(lr){2-4}\cmidrule(l){5-7}
Fix $G$ & Kept & At that share & Mean over shares & Kept & At that share & Mean over shares \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Error of $G$ minus error of the agent's edit under one selector (Qwen3.5-0.8B, test sets, points, 95\% intervals).
Kept: the lower completion rate of the two (\%); the mean runs over shares kept up to it, in steps of one point.}
\label{tab:riskcov}
\end{table*}
"""
    nm = {"nothink32k": "Thinking off, full budget", "nothink8k": "Thinking off, 8k", "eot_bias": "End-of-thinking bias",
          "bf8k": "Budget forcing", "loop": "Loop stop", "rep_penalty": "Repetition penalty", "t1": "$T = 1.0$",
          "prompt": "Prompt only", "thinking off": "Thinking off, both ($k=8$)", "thinking on": "Thinking on, all",
          "all decodings": "All settings"}
    rows = []
    for k, n in nm.items():
        cells = []
        for ds in ("MATH-500", "GSM8K", "MedQA"):
            c = cov[ds]["coverage"].get(k)
            if c is None or not c["decodings"]:
                cells.append("--")
                continue
            a = c["per_answer"]
            cells.append(f"{r1(a[0] * 100)} {ci(a[1], a[2])}")
        rows.append(n + " & " + " & ".join(cells) + r" \\")
    hdr = " & ".join(f"{ds} ({cov[ds]['always_looping_problems']}; {cov[ds]['edit_correct_answers_on_always_looping']})"
                     for ds in ("MATH-500", "GSM8K", "MedQA"))
    mcc = cov["MedQA"]["chance_corrected_all_decodings"]
    t2 = r"""\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}llll@{}}
\toprule
Settings of the unedited model & """ + hdr + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Share of the agent's edit's correct answers on always-looping problems whose problem the unedited model solves
under the given settings (\%, 95\% intervals; four samples per setting). In parentheses: always-looping problems, and
the edit's correct answers on them. The mathematics sets have the eight fixes and the protocol itself, which finishes
none of these problems (thinking on, all: seven settings, $k = 28$; all settings: nine, $k = 36$); MedQA has six
(thinking on, all: $k = 20$; all settings: $k = 24$). On MedQA, uniform guesses among the options would be covered """ \
        + f"{r1(mcc['guessing'] * 100)}\\% of the time (chance-corrected: {r1(mcc['chance_corrected'][0] * 100)}\\% " \
        + f"{ci(mcc['chance_corrected'][1], mcc['chance_corrected'][2])})." + r"""}
\label{tab:coverage}
\end{table*}
"""
    Path("paper/tables").mkdir(exist_ok=True)
    Path("paper/tables/riskcov.tex").write_text(t1)
    Path("paper/tables/coverage.tex").write_text(t2)
    print(t1, t2)


if __name__ == "__main__":
    main()
