"""Figures of the merged paper (CPU), in the style of both cases (preset B, the "signal" colour roles; no panel letters:
the captions name panels by position). Numbers come from the generated macro files (paper/out/numbers_case1.tex from
paper/case1_numbers.py; case2_world_model/paper/numbers_case2.tex from case 2's number script) or from the result files
in case1_reasoning/results directly.

  fig1_blocks  Figure 1: one block per case, each with the three tests -> paper/figures/fig1_case{1,2}.pdf
  case1       accuracy against answers finished; coverage of the edit's correct answers on always-looping problems;
              released minus own answers, five test sets                     -> paper/figures/case1_tests.pdf

    python paper/figures.py                   (after python paper/case1_numbers.py)
"""
import importlib.util
import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WSA = ROOT / "case1_reasoning"
OUT = ROOT / "paper" / "out" / "figures"
TW, CW = 486, 234           # ICML \textwidth and \columnwidth, bp

# case 1's figure module: its colours, markers and plotting helpers (scatter, mark, place) stay the single source
spec = importlib.util.spec_from_file_location("wsafig", WSA / "paper" / "figures" / "make_figures.py")
wf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf)
fk, plt, np = wf.fk, wf.plt, wf.np
R = Path(wf.R)


def macros(path):
    """\\newcommand{\\name}{value} -> {name: value}, typeset minus and plus turned into plain characters"""
    out = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", Path(path).read_text()):
        out[m.group(1)] = m.group(2).replace("$-$", "\u2212").replace("$+$", "+").replace("{,}", ",")
    return out


def r1(x, nd=1):
    """half-up rounding as in the number macros (7.55 -> 7.6); never -0.0"""
    d = Decimal(f"{x:.10f}").quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return str(abs(d) if d == 0 else d)


def finish(fig, name, axes):
    """no panel letters; the plot areas share top and bottom edges within 1.5 pt; saved at exactly the figure size"""
    fig.canvas.draw()
    r, k = fig.canvas.get_renderer(), 72 / fig.dpi
    boxes = [[v * k for v in (bb.x0, bb.y0, bb.x1, bb.y1)] for bb in (a.get_window_extent(r) for a in axes)]
    dev = max(max(b[i] for b in boxes) - min(b[i] for b in boxes) for i in (1, 3))
    assert dev <= 1.5, (name, dev)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def fig1_blocks():
    """Figure 1: one block per case on the same layout (each one column wide): the model and its failure, the agent's
    edit and its headline result, and the three tests of Section 2, each with one number. Case 1's numbers from
    paper/out/numbers_case1.tex, case 2's from case2_world_model/paper/numbers_case2.tex (both generated from result
    files). Case 2's macro file needs the agents' run folders (released separately); without it only case 1's block is
    drawn."""
    c1 = macros(ROOT / "paper" / "out" / "numbers_case1.tex")
    c2_file = ROOT / "case2_world_model" / "paper" / "numbers_case2.tex"
    c2 = macros(c2_file) if c2_file.exists() else None
    T = fk.use_theme(wf.THEME, width_pt=CW, font="arial")
    INK, MUTED = wf.INK, wf.MUTED
    blocks = {
        "fig1_case1": ("reasoning model", "cycle", "Qwen3.5-0.8B on math problems; its\nthinking loops until it is cut",
                       f"MATH-500: {c1['wBaseCutMath']}% of answers cut", "an agent edits the weights",
                       f"MATH-500 accuracy {c1['wBaseMath']}% \u2192 {c1['wEditMath']}%",
                       [("within reach?", f"yes: the unedited model reaches {c1['wCovMath']}%\nof the edit's correct answers"),
                        ("same information, no agent", f"thinking off recovers {c1['wOffShareMath']}% of the gain"),
                        ("where it should add nothing", f"medicine: released answers {c1['wRelMed']}%\nright, the model's own {c1['wOwnMed']}%")]),
    }
    if c2 is not None:
        blocks["fig1_case2"] = ("patient world model", "patient", "simulated ICU records; the dosing\npolicy cancels the drug's effect",
                       f"insulin \u00d72: {c2['wmplainIx']} of the true change", "agents edit the weights",
                       f"Claude {c2['wmClaudeWIx']}, Codex {c2['wmCodexWIx']} of the true change",
                       [("within reach?", f"no: unedited model {c2['wmplainIx']},\nfine-tuning {c2['wmftZIx']}"),
                        ("same information, no agent", f"a script reaches {c2['wmScriptCIx']}"),
                        ("where it should add nothing", f"no-effect world: the agents' estimators\nimply {c2['wmrepFinImplied']} and {c2['wmcodexRepNull']} mg/dL (band \u00b1{c2['wmBand']})")])
    lh = T.small * 1.24
    nl = [max(fk.nlines(b[6][k][1]) for b in blocks.values()) for k in range(3)]     # same box heights on both sides
    TH = [44 + n * lh + 14 for n in nl]
    for name, (zone, icon, what, flaw, agent, result, tests) in blocks.items():
        Y = 360
        H = int(Y + sum(TH) + 2 * 12 + 4)
        d = fk.Figure(name, H, zone, out=str(OUT))
        d.text(0, 0, 1000, zone, T.zone, True)
        d.box(0, 52, 1000, 150, T.panel, None, T.r_outer)
        fk.tile(d, icon, 26, 74, 104)
        d.text(160, 72, 820, what, T.small)
        d.text(160, 150, 820, flaw, T.small, True, color=T.risk)
        d.line([(500, 208), (500, 238)], INK, arrow=True)
        d.box(0, 244, 1000, 100, T.accent_tint, T.accent, T.r_outer, 1.6 * T.lw_box)
        d.text(26, 256, 950, agent, T.body, True)
        d.text(26, 300, 950, result, T.small, True)
        y = Y
        for k, (head, body) in enumerate(tests):
            d.box(0, y, 1000, TH[k], T.risk_tint if k == 2 else T.panel, None, T.r_outer)
            d.text(26, y + 10, 950, f"{k + 1}  {head}", T.small, True, color=MUTED)
            d.text(26, y + 44, 950, body, T.small)
            y += TH[k] + 12
        d.save()
        fk.check_errors()
        fk.render(str(OUT / f"{name}.pdf"))


def case1():
    """left: strict accuracy against answers finished on MATH-500 (every fix); middle: coverage of the edit's correct
    answers on always-looping problems by the unedited model's decodings, against Qwen3.5-2B matched at random; right:
    released minus own answers on the same problems, five test sets"""
    fig = wf.canvas(TW, 152)
    LAB, EDIT, REF, OTHER, OTHER_EDIT, RAMP = wf.LAB, wf.EDIT, wf.REF, wf.OTHER, wf.OTHER_EDIT, wf.RAMP
    a = wf.place(fig, 30, 178, 28, 142)
    a.set(xlim=(0, 101), ylim=(0, 62))
    # left: strict accuracy (cut answers count as wrong) against answers finished; the dashed line leaves the unedited
    # model with its own accuracy on all its finished answers as slope; the arrow is the edit's gain
    A = json.load(open(R / "abstain.json"))["a"]["MATH-500"]["arms"]
    b0 = A["base"]
    x0, y0, own = b0["coverage"] * 100, b0["accuracy_strict"] * 100, b0["finished_accuracy"]
    a.plot([x0, 101], [y0, y0 + (101 - x0) * own], color=REF, lw=0.8, ls=(0, (4, 2.5)), marker="", zorder=1)
    a.text(52, 61.5, f"unedited model's\naccuracy, {r1(own * 100)}%", fontsize=LAB, color=wf.MUTED, ha="right",
           va="top", linespacing=1.25)
    names = {k: v[0] for k, v in wf.STY.items()}
    names["base"] = "unedited"
    lab = {"base": (5, -2, "left", "top"), "t1": (5, -2, "left", "top"), "meandiff": (5, -6, "left", "top"),
           "bf8k": (6, 0, "left", "top"), "nothink32k": (0, -6, "center", "top"), "authors_edit": (5, 3, "left", "bottom")}
    for k in ("base", "t1", "meandiff", "bf8k", "loop", "nothink32k", "authors_edit"):
        x, y = A[k]["coverage"] * 100, A[k]["accuracy_strict"] * 100
        wf.mark(a, k, x, y)
        if k in lab:
            dx, dy, ha, va = lab[k]
            a.annotate(names[k], (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=LAB,
                       fontweight="bold" if k in ("authors_edit", "nothink32k") else "normal", linespacing=1.25, zorder=2,
                       bbox={"fc": "white", "ec": "none", "pad": 0.3})
    e = A["authors_edit"]
    a.annotate("", (e["coverage"] * 100, e["accuracy_strict"] * 100), xytext=(x0, y0), zorder=2,
               arrowprops={"arrowstyle": "-|>,head_length=0.45,head_width=0.22", "color": EDIT, "lw": 1.0,
                           "shrinkA": 5, "shrinkB": 7, "connectionstyle": "angle,angleA=90,angleB=0,rad=10"})
    a.text(3.5, 30, f"+{r1(e['accuracy_strict'] * 100 - y0)}\npoints", fontsize=LAB, ha="left", va="center",
           fontweight="bold", linespacing=1.25)
    a.text(0.97, 0.03, "MATH-500", transform=a.transAxes, ha="right", va="bottom", fontsize=8.5, fontweight="bold")
    a.set(xlabel="Answers finished (%)", ylabel="Strict accuracy (%)")
    # middle: coverage against the number of samples of the unedited model
    cm = json.load(open(R / "coverage_match.json"))
    b = wf.place(fig, 214, 326, 28, 142)
    ks = [1, 2, 4, 8, 16, 32]
    for ds, ls, m in (("MATH-500", "-", "o"), ("GSM8K", (0, (4, 2)), "s")):
        for t, col in (("agent's edit", EDIT), ("Qwen3.5-2B", REF)):
            v = ([cm[ds]["edit"]["coverage"][str(k)] for k in ks] if t == "agent's edit" else
                 [cm[ds]["references"]["Qwen3.5-2B, random"]["coverage"][str(k)][0] for k in ks])
            b.plot(ks, [x * 100 for x in v], color=col, ls=ls, lw=1.2 if t == "agent's edit" else 1.0, marker=m, ms=3.0,
                   mfc=col if ds == "MATH-500" else "white", mec=col, mew=0.8)
    b.set_xscale("log", base=2)
    b.set(xticks=ks, xticklabels=[str(k) for k in ks], xlim=(0.85, 40), ylim=(30, 101),
          xlabel="Samples of the unedited model", ylabel="Correct answers reached (%)")
    b.text(0.95, 97, "agent's edit", fontsize=LAB, fontweight="bold", va="top")
    b.text(11, 68, "Qwen3.5-2B", fontsize=LAB, ha="left", va="top")
    b.text(36, 37, "solid MATH-500\ndashed GSM8K", fontsize=LAB, color=wf.MUTED, ha="right", va="bottom", linespacing=1.25)
    # right: released minus own on the same problems
    pair = json.load(open(R / "rescue.json"))["sets"]
    more = json.load(open(R / "rescue_more.json"))
    head = json.load(open(R / "headroom.json"))
    pair["MMLU-Pro math"] = {"arms": {"authors_edit": head["MMLU-Pro math, agent's edit"],
                                      **{kk: head[f"MMLU-Pro math, {nm}"] for kk, nm in (
                                          ("loop", "loop stop"), ("bf8k", "budget forcing"), ("t1", "T = 1.0"),
                                          ("nothink", "thinking off, 8k"), ("nothink32k", "thinking off, full budget"))
                                         if f"MMLU-Pro math, {nm}" in head}}}
    pair["MMLU-Pro health"]["arms"]["meandiff"] = head["MMLU-Pro health, medical-direction edit, 8 samples"]
    c = wf.place(fig, 362, 484, 28, 142)
    sets = ("MATH-500", "GSM8K", "MMLU-Pro math", "MedQA", "MMLU-Pro health")
    c.axvline(0, color=REF, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    for i, ds in enumerate(sets):
        y = len(sets) - 1 - i
        A = dict(pair[ds]["arms"], **{k: v[ds] for k, v in more.items() if ds in v})
        c.text(-24.5, y + 0.36, ds.replace(" health", " h.").replace(" math", " m."), fontsize=LAB, fontweight="bold",
               va="center")
        if y:
            c.plot([-25, 9], [y - 0.53] * 2, color=RAMP[1], lw=0.6, marker="", zorder=0)
        for kk in ("t1", "bf8k", "loop", "eot_bias", "rep_penalty", "prompt"):
            if kk in A:
                c.scatter(A[kk]["within_problem"]["diff"] * 100, y, s=12, marker="o", color=OTHER, linewidths=0, zorder=3)
        if "meandiff" in A:
            wf.mark(c, "meandiff", A["meandiff"]["within_problem"]["diff"] * 100, y, s=16)
        for kk, dy in (("authors_edit", 0.0), ("nothink32k", -0.19), ("nothink", -0.36)):
            if kk not in A:
                continue
            w = A[kk]["within_problem"]
            c.plot([w["lo"] * 100, w["hi"] * 100], [y + dy] * 2, color=wf.STY[kk][1], lw=1.1, marker="", zorder=2,
                   solid_capstyle="butt")
            wf.mark(c, kk, w["diff"] * 100, y + dy, s=40 if kk == "authors_edit" else 24)
    w = pair["MATH-500"]["arms"]["authors_edit"]["within_problem"]["diff"] * 100
    c.text(w, len(sets) - 0.5, "agent's edit", fontsize=LAB, fontweight="bold", ha="center", va="bottom",
           bbox={"fc": "white", "ec": "none", "pad": 0.3})
    # key for the markers the labels do not name
    for yk, draw, text in ((-0.85, lambda x, y: c.scatter(x, y, s=12, marker="o", color=OTHER, linewidths=0, clip_on=False),
                            "other fixes"),
                           (-1.45, lambda x, y: wf.mark(c, "meandiff", x, y, s=16), "textbook or medical edit"),
                           (-2.05, lambda x, y: (wf.mark(c, "nothink32k", x - 0.9, y, s=24), wf.mark(c, "nothink", x + 0.9, y, s=24)),
                            "thinking off, full / 8k")):
        draw(-23.2, yk)
        c.text(-20.6, yk, text, fontsize=LAB, ha="left", va="center")
    c.set(xlim=(-25, 9), ylim=(-2.45, len(sets) - 0.15), yticks=[], xlabel="Released \u2212 own (points)")
    c.spines["left"].set_visible(False)
    finish(fig, "case1_tests", [a, b, c])


def case1_search():
    """the agent's search: left, accuracy and cut answers against the strength c of the agent's edit; middle, looping
    tokens whose component along u the edit pushes past zero; right, each edit's direction crossed with the other's 64
    columns (200 practice problems, one sample, strength 6)"""
    d = json.load(open(R / "csweep.json"))
    A, flip = d["arms"], d["flip_share"]
    LAB, EDIT, ALERT, REF, RAMP, MUTED, INK = wf.LAB, wf.EDIT, wf.ALERT, wf.REF, wf.RAMP, wf.MUTED, wf.INK
    cs, keys = [0, 1, 2, 4, 6, 8], ["base", "c1", "c2", "c4", "c6", "c8"]
    acc = [A[k]["acc"] * 100 for k in keys]
    cut = [A[k]["cut"] * 100 for k in keys]
    fig = wf.canvas(TW, 146)
    a = wf.place(fig, 38, 192, 28, 134)
    a.axvline(6, color=RAMP[2], lw=0.8, zorder=0)
    a.text(6.15, 99, "agent's\nstrength", fontsize=LAB, color=MUTED, va="top", linespacing=1.25)
    a.plot(cs, acc, color=EDIT, marker="o", ms=3.6, lw=1.2)
    a.plot(cs, cut, color=ALERT, marker="s", ms=3.2, ls=(0, (4, 2)), lw=1.0)
    a.axhline(acc[0], xmax=0.62, color=REF, lw=0.7, ls=(0, (1.5, 1.5)), zorder=0)
    a.text(2.3, acc[0] + 1.5, "unedited", fontsize=LAB, color=MUTED, va="bottom")
    a.text(8.4, acc[-1], "accuracy", fontsize=LAB, va="center", fontweight="bold")
    a.text(8.4, cut[-1] + 2, "answers cut\nat the budget", fontsize=LAB, va="center", linespacing=1.25)
    a.set(xlabel="Edit strength $c$", ylabel="Practice problems (%)", xticks=cs, ylim=(0, 100), xlim=(-0.3, 13.2))
    a.spines["bottom"].set_bounds(0, 8)
    b = wf.place(fig, 236, 352, 28, 134)
    b.axvspan(4, 8, color=RAMP[0], lw=0, zorder=0)
    b.text(6, 101, "accuracy saturated", fontsize=LAB, color=MUTED, ha="center", va="bottom")
    for key, ls, fill, lab in (("L15_looping", "-", True, "layer 15"), ("L19_looping", (0, (4, 2)), False, "layer 19")):
        v = [flip[key][f"c{c}"] * 100 for c in cs]
        b.plot(cs, v, color=ALERT, ls=ls, lw=1.0, marker="o", ms=3.0, mfc=ALERT if fill else "white", mec=ALERT, mew=0.8)
        b.text(8.4, v[-1], lab, fontsize=LAB, va="center")
    b.set(xlabel="Edit strength $c$", ylabel="Looping tokens\nflipped (%)", xticks=cs, ylim=(0, 100), xlim=(-0.3, 12.2))
    b.spines["bottom"].set_bounds(0, 8)
    c = wf.place(fig, 396, 484, 28, 134)
    e2 = d["e2"]
    vals = {("agent", "agent"): A["c6"]["acc"], ("agent", "text"): A["agentdir_mdcols"]["acc"],
            ("text", "agent"): A["mddir_agentcols"]["acc"], ("text", "text"): A["textbook"]["acc"]}
    for j, nset in enumerate(("agent", "text")):
        for dx, dset, col in ((-0.19, "agent", EDIT), (0.19, "text", wf.OTHER_EDIT)):
            v = vals[(dset, nset)] * 100
            c.bar(j + dx, v, width=0.36, color=col, linewidth=0)
            c.text(j + dx, v + 1.5, r1(v), fontsize=LAB, ha="center", va="bottom")
    c.text(-0.19, 4, "agent's $u$", rotation=90, fontsize=LAB, ha="center", va="bottom", color="white")
    c.text(0.19, 4, "textbook $u$", rotation=90, fontsize=LAB, ha="center", va="bottom", color=INK)
    de, ne = e2["direction_effect"]["d"] * 100, e2["columns_effect"]["d"] * 100
    c.text(0.5, 93, f"direction +{r1(de)}\nneurons +{r1(ne)}", fontsize=LAB, ha="center", va="top", linespacing=1.25)
    c.set(xticks=[0, 1], xticklabels=["agent's\nneurons", "textbook\nneurons"], ylim=(0, 100), xlim=(-0.5, 1.5),
          ylabel="Accuracy (%)")
    c.tick_params(axis="x", length=0)
    finish(fig, "case1_search", [a, b, c])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_blocks()
    case1()
    case1_search()
    print("fig1_case1.pdf, fig1_case2.pdf, case1_tests.pdf, case1_search.pdf ->", OUT)
