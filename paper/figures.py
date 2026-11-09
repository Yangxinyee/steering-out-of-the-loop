"""Figures of the merged paper (CPU), in the style of both cases (preset B, the "signal" colour roles; no panel letters:
the captions name panels by position). Numbers come from the generated macro file (paper/out/numbers_case1.tex, written
by paper/case1_numbers.py) or from the result files in case1_reasoning/results directly.

  case1_schematic  case 1's block of the intro's two-column schematic (case 2's block comes with case 2)
                                                                            -> paper/figures/case1_schematic.pdf
  case1       accuracy against answers finished; coverage of the edit's correct answers on always-looping problems;
              released minus own answers, five test sets                     -> paper/figures/case1_tests.pdf

    python paper/figures.py                   (after python paper/case1_numbers.py)
"""
import importlib.util
import json
import os
import re
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


def case1_schematic():
    """case 1's block of the intro's two-column schematic, drawn on the geometry of case 2's block (case 2's figure
    script, fig_schematic): a small reasoning model whose thinking loops; an agent
    edits its weights from practice problems and a forward-pass tool, without the test sets; the edit is judged against
    what the unedited model already reaches and in a domain the model barely knows"""
    c1 = macros(ROOT / "paper" / "out" / "numbers_case1.tex")
    T = fk.use_theme(wf.THEME, width_pt=CW, font="arial")
    INK, MUTED = wf.INK, wf.MUTED
    H = 640
    d = fk.Figure("case1_schematic", H, "Case 1: a reasoning model that loops", out=str(OUT))
    d.text(0, 0, 1000, "reasoning model", T.zone, True)
    d.box(0, 52, 1000, 150, T.panel, None, T.r_outer)
    fk.tile(d, "cycle", 26, 74, 104)
    d.text(160, 72, 820, "Qwen3.5-0.8B on math problems; its\nthinking repeats until it is cut", T.small)
    d.text(160, 150, 820, f"MATH-500: {c1['wBaseCutMath']}% of answers cut", T.small, True, color=T.risk)
    d.line([(500, 210), (500, 252)], INK, arrow=True)
    d.box(0, 260, 1000, 132, T.accent_tint, T.accent, T.r_outer, 1.6 * T.lw_box)
    d.text(26, 276, 950, "agent edits the weights", T.body, True)
    d.text(26, 322, 950, "two MLP matrices; practice problems + forward passes", T.small, color=MUTED)
    d.line([(250, 400), (250, 442)], INK, arrow=True)
    d.line([(750, 400), (750, 442)], INK, arrow=True)
    for x, head, sub, tint in ((0, "reach", "other decodings,\na larger model", T.panel),
                               (520, "medicine", "a domain the model\nbarely knows", T.risk_tint)):
        d.box(x, 450, 480, 182, tint, None, T.r_outer)
        d.text(x + 24, 466, 440, head, T.body, True)
        d.text(x + 24, 514, 440, sub, T.small, color=MUTED)
    d.save()
    fk.check_errors()
    fk.render(str(OUT / "case1_schematic.pdf"))


def case1():
    """left: strict accuracy against answers finished on MATH-500 (every fix); middle: coverage of the edit's correct
    answers on always-looping problems by the unedited model's decodings, against Qwen3.5-2B matched at random; right:
    released minus own answers on the same problems, five test sets"""
    fig = wf.canvas(TW, 152)
    LAB, EDIT, REF, OTHER, OTHER_EDIT, RAMP = wf.LAB, wf.EDIT, wf.REF, wf.OTHER, wf.OTHER_EDIT, wf.RAMP
    a = wf.place(fig, 30, 178, 28, 142)
    a.set(xlim=(0, 101), ylim=(0, 62))
    wf.scatter(a, "MATH-500", ("base", "t1", "meandiff", "bf8k", "loop", "nothink32k", "authors_edit"),
               {"base": (5, -2, "left", "top"), "t1": (5, -2, "left", "top"), "meandiff": (5, -6, "left", "top"),
                "bf8k": (6, 0, "left", "top"), "nothink32k": (0, -6, "center", "top"), "authors_edit": (5, 3, "left", "bottom")},
               (47, 61.5), (3.5, 30))
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
    c.text(-24.5, -0.56, "thinking off: dark full\nbudget, light 8k", fontsize=LAB, ha="left", va="top", linespacing=1.25,
           bbox={"fc": "white", "ec": "none", "pad": 0.3}, zorder=5)
    c.set(xlim=(-25, 9), ylim=(-1.45, len(sets) - 0.15), yticks=[], xlabel="Released \u2212 own (points)")
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
            c.text(j + dx, v + 1.5, f"{v:.1f}", fontsize=LAB, ha="center", va="bottom")
    c.text(-0.19, 4, "agent's $u$", rotation=90, fontsize=LAB, ha="center", va="bottom", color="white")
    c.text(0.19, 4, "textbook $u$", rotation=90, fontsize=LAB, ha="center", va="bottom", color=INK)
    de, ne = e2["direction_effect"]["d"] * 100, e2["columns_effect"]["d"] * 100
    c.text(0.5, 93, f"direction +{de:.1f}\nneurons +{ne:.1f}", fontsize=LAB, ha="center", va="top", linespacing=1.25)
    c.set(xticks=[0, 1], xticklabels=["agent's\nneurons", "textbook\nneurons"], ylim=(0, 100), xlim=(-0.5, 1.5),
          ylabel="Accuracy (%)")
    c.tick_params(axis="x", length=0)
    finish(fig, "case1_search", [a, b, c])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    case1_schematic()
    case1()
    case1_search()
    print("case1_schematic.pdf, case1_tests.pdf, case1_search.pdf ->", OUT)
