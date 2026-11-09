"""The paper's figures, from the result files only (CPU). Style: preset B of figlib (vendored in paper/figures/figlib).

    python paper/figures/make_figures.py
    (writes paper/figures/fig1_teaser*.pdf, fig2_completion, fig3_reach, fig4_medicine, fig5_search, fig6_deferral (.pdf) and alignment.json)
"""
import dataclasses
import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

sys.path.insert(0, os.environ.get("PAPER_FIGURES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figlib")))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import figkit as fk  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotkit as pk  # noqa: E402
from themes import THEMES  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
OUT = os.path.dirname(os.path.abspath(__file__))
TW, CW = 454, 218          # ACL \textwidth and \columnwidth in bp (455.24 and 219.09 TeX pt)
B = THEMES["B"]
INK, MUTED, RAMP = B.ink, B.muted, B.ramp
# colour roles, one per kind of setting, the same in every figure (saturated hero, quiet baselines)
EDIT = "#E8590C"           # the agent's edit, the object of the audit (vivid orange)
STRONG = "#0F4D92"         # thinking off at the full budget, the strongest cheap baseline (deep blue)
STRONG8K = "#4A90D9"       # thinking off at 8,192 tokens (light blue)
OTHER_EDIT = "#F2A900"     # the textbook and medical-direction edits (amber)
ALERT = "#C2185B"          # loops and answers cut at the budget (crimson)
OTHER = RAMP[4]            # every other decoding fix (slate)
REF = "#000000"            # references: the model's own accuracy, zero, fitted lines
LAB = 8                    # direct labels, pt
THEME = dataclasses.replace(B, accent=EDIT, accent_tint="#FCE4D6", risk=ALERT, risk_tint="#F9DDE8", ok=INK)
plt.rcParams.update(pk.mpl_style("B"))
plt.rcParams["figure.constrained_layout.use"] = False     # plot areas are placed in bp (place())
ALIGN = {}


def r1(x, nd=1):
    """half-up rounding as in the paper's number macros (7.55 -> 7.6); never -0.0"""
    d = Decimal(f"{x:.10f}").quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return str(abs(d) if d == 0 else d)


def canvas(w, h):
    return plt.figure(figsize=(w / 72, h / 72), layout="none")


def place(fig, x0, x1, y0, y1):
    """an axes whose plot area is the rectangle x0..x1, y0..y1 (bp from the figure's lower left)"""
    W, H = fig.get_size_inches() * 72
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


def finish(fig, name, axes):
    """no panel letters (the lab's default: panels are named by position in captions and text); the plot areas of the
    row, measured after the final draw, must share top and bottom edges within 1.5 pt (recorded in alignment.json);
    saved at exactly the figure size"""
    fig.canvas.draw()
    r, k = fig.canvas.get_renderer(), 72 / fig.dpi
    boxes = [[v * k for v in (bb.x0, bb.y0, bb.x1, bb.y1)] for bb in (a.get_window_extent(r) for a in axes)]
    dev = max(max(b[i] for b in boxes) - min(b[i] for b in boxes) for i in (1, 3))
    ALIGN[name] = {"plot_areas_bp": [[round(v, 2) for v in b] for b in boxes], "max_edge_deviation_pt": round(dev, 3),
                   "tolerance_pt": 1.5, "pass": bool(dev <= 1.5)}
    assert dev <= 1.5, (name, dev)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)


def score(stem):
    s = json.load(open(os.path.join(R, "runs", stem + ".scores.json")))
    return s["acc"], 1 - s["cut_share"]


def fig1():
    """teaser: the unedited model loops and is cut; three parallel fixes end the loop; the answers they release are as
    accurate as the model's own on the same problems, high in mathematics and low in medicine"""
    T = fk.use_theme(THEME, width_pt=TW, font="arial")
    pair = json.load(open(os.path.join(R, "rescue.json")))["sets"]
    rows = [(dom, pair[ds]["arms"]["authors_edit"]["within_problem"]) for dom, ds in
            (("mathematics", "MATH-500"), ("medicine", "MedQA"))]
    fin = {k: score(s)[1] for k, s in (("base", "base/think35_math500"), ("edit", "edit_authors/think35_math500"),
                                       ("t1", "base/card_think_math500"), ("bf", "base/think35_bf8192_math500"))}
    H, Y0, Y1 = 262, 34, 258
    d = fk.Figure("fig1_teaser", H, "Released answers", out=OUT)
    zones = [(0, 258, "unedited model"), (300, 254, "fixes"), (596, 404, "released answers")]
    for x, w, head in zones:
        d.text(x + 2, 4, w, head, T.zone, True)
    # unedited model: the loop, the cut, and how few answers finish
    x, w = zones[0][:2]
    d.box(x, Y0, w, Y1 - Y0, T.panel, None, T.r_outer)
    fk.tile(d, "cycle", x + 16, Y0 + 16, 46, T.risk)
    d.ctext(x + 74, Y0 + 16, 170, 46, "thinking loop", T.body, True, align="left")
    for i in range(6):
        d.box(x + 16 + i * 31, Y0 + 86, 25, 16, T.risk_tint, T.risk, T.r_inner, 0.8 * T.lw_box)
    d.line([(x + 210, Y0 + 76), (x + 210, Y0 + 112)], INK, 2.2 * T.scale)
    d.text(x + 16, Y0 + 118, w - 20, "cut at the token budget", T.body)
    d.text(x + 16, Y0 + 160, 90, f"{r1(fin['base'] * 100, 0)}%", 2.1 * T.body, True)
    d.text(x + 92, Y0 + 162, w - 96, "of MATH-500\nanswers finished", T.small, color=MUTED)
    # fixes: parallel cards, no arrows between them
    x, w = zones[1][:2]
    cards = [("agent's weight edit", "weights", fin["edit"]), ("temperature 1.0", "sampling", fin["t1"]),
             ("budget forcing", "token budget", fin["bf"])]
    ch, gap = 64, (Y1 - Y0 - 3 * 64) / 2
    for i, (name, where, f) in enumerate(cards):
        y = Y0 + i * (ch + gap)
        d.box(x, y, w, ch, T.accent_tint if i == 0 else T.panel, T.accent if i == 0 else None, T.r_outer,
              1.6 * T.lw_box if i == 0 else None)
        d.text(x + 14, y + 10, w - 20, name, T.body, True)
        d.text(x + 14, y + 36, w - 20, f"{where} · {r1(f * 100, 0)}% finished", T.small, color=MUTED)
    ym = (Y0 + Y1) / 2
    d.line([(zones[0][0] + zones[0][1] + 6, ym), (x - 8, ym)], INK, arrow=True)
    d.line([(x + w + 6, ym), (zones[2][0] - 8, ym)], INK, arrow=True)
    # released answers against the model's own on the same problems
    x, w = zones[2][:2]
    d.box(x, Y0, w, Y1 - Y0, T.panel, None, T.r_outer)
    d.text(x + 16, Y0 + 10, w - 24, "same problems, agent's edit", T.small, color=MUTED)
    S = 190                                       # bar length for 100%
    for i, (dom, p) in enumerate(rows):
        y = Y0 + 42 + i * 92
        d.text(x + 16, y, 200, dom, T.body, True)
        for j, (lab, v, c) in enumerate((("released", p["rescued_acc"], EDIT),
                                          ("model's own", p["base_finished_acc"], T.ramp[3]))):
            yy = y + 28 + j * 24
            d.box(x + 16, yy, S, 15, T.ramp[1], None, 1.5 * T.scale)
            d.box(x + 16, yy, S * v, 15, c, None, 1.5 * T.scale)
            d.text(x + 26 + S, yy - 1, w - S - 30, f"{lab} {r1(v * 100)}%", T.small)
    d.save()
    fk.check_errors()
    fk.render(os.path.join(OUT, "fig1_teaser.pdf"))


def fig1_examples():
    """teaser, real-example version: one MATH-500 and one MedQA problem drawn at random by wsa/teaser_examples.py
    (a sample the base model cut and the agent's edit finished, on a problem the base model sometimes finishes; the
    medical one among wrong released answers). Left: the base model's loop as one chip per output line (list markers
    dropped), one cycle shown twice, then the line it was cut on; right: the edit's answer. Every text is verbatim
    from the outputs and '…' marks every cut. Right edge: released against own accuracy over all such problems."""
    T = fk.use_theme(THEME, width_pt=TW, font="arial")
    ex = json.load(open(os.path.join(R, "teaser_examples.json")))
    pair = json.load(open(os.path.join(R, "rescue.json")))["sets"]
    pick = {"MATH-500": ("mathematics", "problems", "Their sum is $85 + 58 = 143$.", "\\boxed{143}"),
            "MedQA": ("medicine", "questions", "So the TSH is low.", "The answer is (D)")}
    S, ELL, CH = T.small, "…", 24

    def lines(t):
        return [re.sub(r"^\*\s+", "", x.strip()) for x in t.split("\n") if x.strip()]

    def cycle(t):
        L = lines(t)
        tail = L[:-1]
        for p in range(1, 100):
            n = 0
            while n + p < len(tail) and tail[-1 - n] == tail[-1 - n - p]:
                n += 1
            if n >= 3 * p:
                return tail[-p:], (n + p) // p, L[-1]

    def fit(t, w, lead=False):
        t = (ELL + " " if lead else "") + t
        words = t.split(" ")
        while fk.tw(" ".join(words), S) > w:
            words = words[:-2] + [ELL] if words[-1] == ELL else words[:-1] + [ELL]
        return " ".join(words)

    def shown(t):                                            # display only: LaTeX math delimiters dropped
        return t.replace("$", "")

    def chip(x, y, t, tint, bar):
        m = re.fullmatch(r"\\boxed\{(.*)\}", t)               # \boxed{...} drawn as a box
        t = shown(m.group(1) if m else t)
        w = fk.tw(t, S) + (29 if m else 15)
        d.box(x, y, w, CH, tint, None, 2 * T.scale)
        d.box(x, y, 3.5, CH, bar, None, 0)
        if m:
            d.box(x + 9, y + 1.5, fk.tw(t, S) + 12, CH - 3, None, INK, 0, 0.9 * T.lw_box)
        d.text(x + (15 if m else 9), y + 2.5, w, t, S)
        return w

    def mark(cx, cy, ok):
        if ok:
            fk.checkmark(d, cx, cy, 8)
        else:
            d.circle(cx, cy, 8, T.risk, T.risk, T.scale)
            for s in (1, -1):
                d.line([(cx - 3.6, cy - 3.6 * s), (cx + 3.6, cy + 3.6 * s)], fk.WHITE, 1.6 * T.scale)

    XL, WL, XR, WR, XT = 0, 506, 522, 290, 838
    d = fk.Figure("fig1_teaser_examples", 320, "Released answers, two examples", out=OUT)
    for i, (ds, (dom, unit, reason, final)) in enumerate(pick.items()):
        r, y0 = ex[ds], i * 166
        base, edit = r["base"], r["edit"]
        assert base["finish"] == "length" and edit["finish"] == "stop" and reason in edit["text"] and final in edit["text"]
        if i:
            d.line([(0, y0 - 7), (1000, y0 - 7)], T.ramp[1], 0.7 * T.lw_box)
        q = r["question"].get("problem") or r["question"]["prompt"].split("Question: ", 1)[1]
        dw = fk.tw(dom, T.body, True) + 12
        d.text(0, y0, dw, dom, T.body, True)
        d.text(dw, y0 + 1, 1000 - dw, fit(shown(" ".join(q.split())), 995 - dw), S, color=MUTED)
        cyc, reps, cut = cycle(base["text"])
        head = f"unedited · {base['n_tok']:,} tokens · cut"
        d.text(XL, y0 + 26, WL, head, S, True)
        mark(XL + fk.tw(head, S, True) + 13, y0 + 37, False)
        d.text(XR, y0 + 26, WR, f"agent's edit · {edit['n_tok']:,} tokens", S, True)
        e0 = fk.tw(ELL, S) + 8
        for row in (0, 1):                                   # the same cycle, twice
            x, y = XL, y0 + 52 + row * (CH + 4)
            if row == 0:
                d.text(x, y + 2, e0, ELL, S, color=MUTED)
            x += e0
            for ln in cyc:
                if x + fk.tw(shown(ln), S) + 15 > XL + WL - 22:
                    break
                x += chip(x, y, ln, T.risk_tint, T.risk) + 5
            d.text(x + 1, y + 2, 30, ELL, S, color=MUTED)
        d.text(XL, y0 + 108, WL, f"{ELL}  the same {len(cyc)} lines, {reps} times", S, color=MUTED)
        w = chip(XL + e0, y0 + 130, cut, T.risk_tint, T.risk)
        d.line([(XL + e0 + w + 7, y0 + 127), (XL + e0 + w + 7, y0 + 130 + CH + 3)], INK, 2.2 * T.scale)
        d.text(XR, y0 + 54, e0, ELL, S, color=MUTED)
        chip(XR + e0, y0 + 52, fit(shown(reason), WR - e0 - 16), T.accent_tint, T.accent)
        d.text(XR, y0 + 108, 30, ELL, S, color=MUTED)
        w = chip(XR + e0, y0 + 130, final, T.accent_tint, T.accent)
        mark(XR + e0 + w + 14, y0 + 130 + CH / 2, edit["correct"])
        if not edit["correct"]:
            d.text(XR + e0 + w + 28, y0 + 132, WR, f"correct: ({r['question']['answer']})", S, color=MUTED)
        if ds == "MATH-500":                                 # mathematics: coverage on always-looping problems
            cv = json.load(open(os.path.join(R, "coverage.json")))[ds]
            two = json.load(open(os.path.join(R, "coverage_match.json")))[ds]["references"]["Qwen3.5-2B, random"]["coverage"]["32"]
            d.text(XT, y0 + 26, 1000 - XT, f"{cv['always_looping_problems']} always looping", S, color=MUTED)
            bars = (("edit covered", cv["coverage"]["all decodings"]["per_answer"][0], EDIT), ("2B covered", two[0], OTHER))
        else:                                                # medicine: released against own on the same questions
            p = pair[ds]["arms"]["authors_edit"]["within_problem"]
            d.text(XT, y0 + 26, 1000 - XT, f"{p['problems']} {unit} like it", S, color=MUTED)
            bars = (("released", p["rescued_acc"], EDIT), ("model's own", p["base_finished_acc"], OTHER))
        for j, (lab, v, c) in enumerate(bars):
            yy = y0 + 52 + j * 52
            d.text(XT, yy, 1000 - XT, f"{lab} {r1(v * 100)}%", S)
            d.box(XT, yy + 25, 160, 9, T.ramp[1], None, T.scale)
            d.box(XT, yy + 25, 160 * v, 9, c, None, T.scale)
    d.save()
    fk.check_errors()
    fk.render(os.path.join(OUT, "fig1_teaser_examples.pdf"), dpi=300)


STY = {"base": ("base", REF, "o", 24), "t1": ("T = 1.0", OTHER, "v", 24), "bf8k": ("budget forcing,\nloop stop", OTHER, "s", 22),
       "loop": (None, OTHER, "D", 16), "meandiff": ("textbook edit", OTHER_EDIT, "^", 28),
       "nothink": ("thinking off, 8k", STRONG8K, "X", 28), "nothink32k": ("thinking off,\nfull budget", STRONG, "h", 40),
       "authors_edit": ("agent's edit", EDIT, "P", 56), "eot_bias": ("end-of-thinking\nbias", OTHER, "<", 24),
       "rep_penalty": ("repetition\npenalty", OTHER, ">", 24), "prompt": ("prompt", OTHER, "*", 34)}


def mark(ax, k, x, y, s=None, hollow=False, **kw):
    _, c, m, s0 = STY[k]
    face = {"facecolor": "white", "edgecolor": c, "linewidths": 0.9} if k == "base" or hollow else {"color": c, "linewidths": 0}
    ax.scatter(x, y, s=s or s0, marker=m, zorder=4 if k == "authors_edit" else 3, clip_on=False, **face, **kw)


def scatter(ax, ds, arms, lab, own_lab, gain_at):
    """strict accuracy against answers finished on one test set. The dashed line leaves the base model with the base
    model's own finished accuracy as slope, so a fix on it releases answers as accurate as the model's own; the arrow
    (up, then across) is the agent's edit's gain"""
    A = dict(json.load(open(os.path.join(R, "abstain.json")))["a"][ds]["arms"])
    if ds == "MATH-500":                                 # the settings wsa.abstain does not cover (wsa.completion_points)
        cp = json.load(open(os.path.join(R, "completion_points.json")))[ds]
        A.update({k: {"coverage": v["finished"], "accuracy_strict": v["strict"]} for k, v in cp.items() if k not in A})
    b = A["base"]
    x0, y0, own = b["coverage"] * 100, b["accuracy_strict"] * 100, b["finished_accuracy"]
    xr = ax.get_xlim()
    ax.plot([x0, xr[1]], [y0, y0 + (xr[1] - x0) * own], color=REF, lw=0.8, ls=(0, (4, 2.5)), marker="", zorder=1)
    ax.text(*own_lab, f"model's own\naccuracy, {r1(own * 100)}%", fontsize=LAB, color=MUTED, ha="right", va="top",
            linespacing=1.25)
    for k in arms:
        x, y = A[k]["coverage"] * 100, A[k]["accuracy_strict"] * 100
        mark(ax, k, x, y)
        if k in lab:
            dx, dy, ha, va = lab[k][:4]
            name = lab[k][5] if len(lab[k]) > 5 else STY[k][0]
            lead = {"arrowprops": {"arrowstyle": "-", "lw": 0.5, "color": MUTED, "shrinkA": 1, "shrinkB": 3}} \
                if len(lab[k]) > 4 and lab[k][4] else {}
            ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=LAB,
                        fontweight="bold" if k in ("authors_edit", "nothink32k") else "normal", linespacing=1.25, zorder=2,
                        bbox={"fc": "white", "ec": "none", "pad": 0.3}, **lead)
    e = A["authors_edit"]
    ax.annotate("", (e["coverage"] * 100, e["accuracy_strict"] * 100), xytext=(x0, y0), zorder=2,
                arrowprops={"arrowstyle": "-|>,head_length=0.45,head_width=0.22", "color": EDIT, "lw": 1.0,
                            "shrinkA": 5, "shrinkB": 7, "connectionstyle": "angle,angleA=90,angleB=0,rad=10"})
    ax.text(*gain_at, f"+{r1(e['accuracy_strict'] * 100 - y0)}\npoints", fontsize=LAB, ha="left", va="center",
            fontweight="bold", linespacing=1.25)
    ax.text(0.97, 0.03, ds, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set(xlabel="Answers finished (%)", ylabel="Strict accuracy (%)")


def fig2():
    """Fig. 2, one claim: in mathematics the edit's gain is completion. a (primary): strict accuracy against answers finished
    on MATH-500, every fix; b (generalisation): 32 textbook edits lie on one line; c (control): on the same problems the
    released answers are no more accurate than the model's own, on all five test sets"""
    fig = canvas(TW, 154)
    a = place(fig, 30, 190, 28, 142)
    a.set(xlim=(0, 101), ylim=(0, 62))
    scatter(a, "MATH-500", ("base", "prompt", "t1", "rep_penalty", "meandiff", "bf8k", "loop", "eot_bias", "nothink",
                            "nothink32k", "authors_edit"),
            {"base": (5, -2, "left", "top", False, "base, prompt"), "t1": (5, -2, "left", "top"),
             "rep_penalty": (-5, 0, "right", "center"), "meandiff": (5, -6, "left", "top"),
             "bf8k": (6, 0, "left", "top"), "eot_bias": (22, -26, "left", "top", True),
             "nothink": (-5, 1, "right", "bottom", False, "8k"), "nothink32k": (0, -6, "center", "top"),
             "authors_edit": (5, 3, "left", "bottom")},
            (47, 61.5), (8.6, 40))
    # b: the 32 textbook edits on practice problems
    g = json.load(open(os.path.join(R, "part1b_grid.json")))
    c = g["cells"]
    f = np.array([1 - v["cut"] for v in c.values()]) * 100
    acc = np.array([v["acc"] for v in c.values()]) * 100
    one = np.array([k.startswith("l15|") for k in c])
    k, b0 = np.polyfit(f, acc, 1)
    r2 = 1 - ((acc - (k * f + b0)) ** 2).sum() / ((acc - acc.mean()) ** 2).sum()
    bx = place(fig, 226, 318, 28, 142)
    x = np.array([5, 51])
    bx.plot(x, k * x + b0, color=REF, lw=0.8, marker="", zorder=1)
    bx.scatter(f[~one], acc[~one], s=16, marker="^", color=OTHER_EDIT, linewidths=0, zorder=3)
    bx.scatter(f[one], acc[one], s=15, marker="^", facecolor="white", edgecolor=OTHER_EDIT, linewidths=0.9, zorder=3)
    bf, ba = (1 - g["base"]["cut"]) * 100, g["base"]["acc"] * 100
    mark(bx, "base", bf, ba, s=22)
    bx.annotate("base", (bf, ba), xytext=(0, 6), textcoords="offset points", fontsize=LAB, ha="center", va="bottom")
    bx.text(2, 54, f"slope {k:.3f}\nR² = {r2:.3f}", fontsize=LAB, color=MUTED, va="top", linespacing=1.25)
    bx.scatter([24], [12.5], s=16, marker="^", color=OTHER_EDIT, linewidths=0, clip_on=False)
    bx.text(26, 12.5, "layers 15 + 19", fontsize=LAB, va="center")
    bx.scatter([24], [6.5], s=15, marker="^", facecolor="white", edgecolor=OTHER_EDIT, linewidths=0.9, clip_on=False)
    bx.text(26, 6.5, "layer 15", fontsize=LAB, va="center")
    bx.text(0.97, 0.0, "practice", transform=bx.transAxes, ha="right", va="bottom", fontsize=8.5, fontweight="bold")
    bx.set(xlabel="Answers finished (%)", ylabel="Accuracy (%)", xlim=(0, 52), ylim=(0, 55))
    # c: released minus own on the same problems; intervals for the agent's edit and thinking off at either budget, the
    # other fixes (T = 1.0, budget forcing, loop stop, end-of-thinking bias, repetition penalty, prompt) as grey dots
    pair = json.load(open(os.path.join(R, "rescue.json")))["sets"]
    more = json.load(open(os.path.join(R, "rescue_more.json")))
    # MMLU-Pro math (the mathematics set with headroom: the agent's edit and the decoding fixes) and
    # the medical-direction edit on MMLU-Pro health with eight samples per question
    head = json.load(open(os.path.join(R, "headroom.json")))
    pair["MMLU-Pro math"] = {"arms": {"authors_edit": head["MMLU-Pro math, agent's edit"],
                                      **{kk: head[f"MMLU-Pro math, {nm}"] for kk, nm in (
                                          ("loop", "loop stop"), ("bf8k", "budget forcing"), ("t1", "T = 1.0"),
                                          ("nothink", "thinking off, 8k"), ("nothink32k", "thinking off, full budget"))
                                         if f"MMLU-Pro math, {nm}" in head}}}
    pair["MMLU-Pro health"]["arms"]["meandiff"] = head["MMLU-Pro health, medical-direction edit, 8 samples"]
    cx = place(fig, 380, 452, 28, 142)
    lab_t = cx.get_yaxis_transform()                        # x in axes units, y in data: labels in the left margin
    sets = ("MATH-500", "GSM8K", "MMLU-Pro math", "MedQA", "MMLU-Pro health")
    cx.axvline(0, color=REF, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    for i, ds in enumerate(sets):
        y = len(sets) - 1 - i
        A = dict(pair[ds]["arms"], **{k: v[ds] for k, v in more.items() if ds in v})
        cx.text(-0.04, y - 0.1, ds.replace(" health", " h.").replace(" math", " m."), fontsize=LAB, fontweight="bold",
                va="center", ha="right", transform=lab_t)
        if y:                                                 # a rule under each data set's rows
            cx.plot([-25, 9], [y - 0.53] * 2, color=RAMP[1], lw=0.6, marker="", zorder=0)
        for kk in ("t1", "bf8k", "loop", "eot_bias", "rep_penalty", "prompt"):
            if kk in A:
                cx.scatter(A[kk]["within_problem"]["diff"] * 100, y, s=12, marker="o", color=OTHER, linewidths=0, zorder=3)
        if "meandiff" in A:
            mark(cx, "meandiff", A["meandiff"]["within_problem"]["diff"] * 100, y, s=16)
        for kk, dy in (("authors_edit", 0.0), ("nothink32k", -0.19), ("nothink", -0.36)):
            if kk not in A:
                continue
            w = A[kk]["within_problem"]
            cx.plot([w["lo"] * 100, w["hi"] * 100], [y + dy] * 2, color=STY[kk][1], lw=1.1, marker="", zorder=2,
                    solid_capstyle="butt")
            mark(cx, kk, w["diff"] * 100, y + dy, s=40 if kk == "authors_edit" else 24)
    w = pair["MATH-500"]["arms"]["authors_edit"]["within_problem"]["diff"] * 100
    cx.text(w, len(sets) - 0.5, "agent's edit", fontsize=LAB, fontweight="bold", ha="center", va="bottom",
            bbox={"fc": "white", "ec": "none", "pad": 0.3})
    # key for the markers the labels do not name
    for yk, draw, text in ((-0.8, lambda x, y: cx.scatter(x, y, s=12, marker="o", color=OTHER, linewidths=0, clip_on=False,
                                                           transform=lab_t), "other fixes"),
                           (-1.35, lambda x, y: mark(cx, "meandiff", x, y, s=16, transform=lab_t), "other edits"),
                           (-1.9, lambda x, y: (mark(cx, "nothink32k", x - 0.035, y, s=24, transform=lab_t),
                                                mark(cx, "nothink", x + 0.035, y, s=24, transform=lab_t)),
                            "thinking off, full / 8k")):
        draw(-0.68, yk)
        cx.text(-0.6, yk, text, fontsize=LAB, ha="left", va="center", zorder=5, transform=lab_t,
                bbox={"fc": "white", "ec": "none", "pad": 0.3})
    cx.set(xlim=(-25, 9), ylim=(-2.2, len(sets) - 0.15), yticks=[], xlabel="Released − own")
    cx.spines["left"].set_visible(False)
    finish(fig, "fig2_completion", [a, bx, cx])
    return k, b0, r2


def fig3():
    """Fig. 3, one claim: where the model always loops, the edit's correct answers are ones the unedited model reaches, and
    at equal completion they are no more accurate than the fixes that keep thinking on. a (primary): coverage of the
    edit's correct answers on always-looping problems by the unedited model's decodings; b, c (control): risk-coverage
    curves under one selector, MATH-500 and GSM8K"""
    cm = json.load(open(os.path.join(R, "coverage_match.json")))       # the 2B matched at random (wsa.coverage_match)
    rc = json.load(open(os.path.join(R, "riskcov.json")))
    fig = canvas(TW, 150)
    a = place(fig, 34, 176, 28, 138)
    ks = [1, 2, 4, 8, 16, 32]
    for ds, ls, m in (("MATH-500", "-", "o"), ("GSM8K", (0, (4, 2)), "s")):
        curves = {"agent's edit": [cm[ds]["edit"]["coverage"][str(k)] for k in ks],
                  "Qwen3.5-2B": [cm[ds]["references"]["Qwen3.5-2B, random"]["coverage"][str(k)][0] for k in ks]}
        for t, col in (("agent's edit", EDIT), ("Qwen3.5-2B", REF)):
            a.plot(ks, [v * 100 for v in curves[t]], color=col, ls=ls, lw=1.2 if t == "agent's edit" else 1.0,
                   marker=m, ms=3.0, mfc=col if ds == "MATH-500" else "white", mec=col, mew=0.8)
    a.set_xscale("log", base=2)
    a.set(xticks=[1, 2, 4, 8, 16, 32], xticklabels=["1", "2", "4", "8", "16", "32"], xlim=(0.85, 40), ylim=(30, 101),
          xlabel="Samples of the 0.8B per problem", ylabel="Answers covered (%)")
    a.text(0.95, 97, "agent's edit", fontsize=LAB, fontweight="bold", va="top")
    a.text(1.6, 33, "Qwen3.5-2B", fontsize=LAB, va="bottom")
    a.text(36, 37, "solid MATH-500\ndashed GSM8K", fontsize=LAB, color=MUTED, ha="right", va="bottom", linespacing=1.25)
    sty = (("t1", OTHER, "-", 0.8), ("bf8k", OTHER, "-", 0.8), ("loop", OTHER, "-", 0.8), ("meandiff", OTHER_EDIT, "-", 1.0),
           ("nothink8k", STRONG8K, (0, (3, 1.5)), 1.1), ("nothink32k", STRONG, "-", 1.2), ("authors_edit", EDIT, "-", 1.6))
    axes = [a]
    labs = {"MATH-500": {"authors_edit": (72, 18, "left"), "nothink32k": (84, 45.5, "right"), "nothink8k": (66, 38.5, "right"),
                         "meandiff": (2, 13, "left")}, "GSM8K": {}}
    names = {"authors_edit": "agent's\nedit", "nothink32k": "thinking off", "nothink8k": "8k", "meandiff": "textbook"}
    for x0, x1, ds in ((222, 322, "MATH-500"), (352, 446, "GSM8K")):
        ax = place(fig, x0, x1, 28, 138)
        C = rc[ds]["curves"]
        for k, col, ls, lw in sty:
            e = C[k]["error"]
            ax.plot([float(c) * 100 for c in e], [v * 100 for v in e.values()], color=col, ls=ls, lw=lw, marker="",
                    zorder=4 if k == "authors_edit" else 2)
            if k in labs[ds]:
                x, y, ha = labs[ds][k]
                ax.text(x, y, names[k], fontsize=LAB, ha=ha, va="center", fontweight="bold" if k == "authors_edit" else "normal",
                        linespacing=1.25)
        ax.text(0.97, 0.03, ds, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="right", va="bottom")
        ax.set(xlim=(0, 100), ylim=(0, 46), xlabel="Answers kept (%)")
        axes.append(ax)
    axes[1].set_ylabel("Error of kept answers (%)")
    axes[2].text(2, 10.5, "other fixes", fontsize=LAB, color=MUTED, va="bottom")
    finish(fig, "fig3_reach", axes)


def fig4():
    """Fig. 4, one claim: where the model knows little, releasing answers brings little. a (boundary): the MedQA scatter;
    b (decomposition): each fix's released answers are right about as often as the model's own in that domain, so the
    domain sets the level; c (mechanism): temperature 1.0 ends medical loops at every loop period, mathematical ones rarely"""
    fig = canvas(TW, 154)
    a = place(fig, 30, 178, 28, 142)
    a.set(xlim=(74, 102), ylim=(34, 51))
    scatter(a, "MedQA", ("base", "t1", "bf8k", "loop", "nothink", "authors_edit"),
            {"base": (0, -6, "center", "top"), "t1": (-6, 1, "right", "center"), "bf8k": (5, -3, "left", "top"),
             "nothink": (0, -6, "center", "top"), "authors_edit": (-8, 3, "right", "bottom")},
            (91, 50.8), (77.6, 41.6))
    a.set(yticks=[35, 40, 45, 50])
    # b: accuracy of released answers in each domain (filled MATH-500, open MedQA), against the model's own
    resc = json.load(open(os.path.join(R, "rescue.json")))["sets"]
    ab = json.load(open(os.path.join(R, "abstain.json")))["a"]
    bx = place(fig, 206, 330, 28, 142)
    rows = (("t1", "T = 1.0"), ("bf8k", "budget forcing"), ("loop", "loop stop"), ("authors_edit", "agent's edit"),
            ("nothink", "thinking off, 8k"))
    for ds, ha in (("MATH-500", "right"), ("MedQA", "left")):
        own = ab[ds]["arms"]["base"]["finished_accuracy"] * 100
        bx.axvline(own, color=REF, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
        bx.text(own + (-1.5 if ha == "right" else 1.5), -0.62, f"own {r1(own)}", fontsize=LAB, color=MUTED, ha=ha,
                va="bottom")
        bx.text(own - 1.5, 5.05, ds, fontsize=LAB, fontweight="bold", ha="right", va="top")
    for i, (kk, name) in enumerate(rows):
        y = len(rows) - 1 - i
        m, md = (resc[ds]["arms"][kk]["rescued_accuracy"] * 100 for ds in ("MATH-500", "MedQA"))
        bx.plot([md, m], [y, y], color=RAMP[1], lw=1.6, marker="", zorder=1, solid_capstyle="butt")
        mark(bx, kk, m, y, s=32 if kk == "authors_edit" else 22)
        mark(bx, kk, md, y, s=32 if kk == "authors_edit" else 22, hollow=True)
        bx.text(52, y + 0.36, name, fontsize=LAB, va="center",
                fontweight="bold" if kk == "authors_edit" else "normal")
    bx.set(xlim=(25, 101), ylim=(-0.7, 5.1), yticks=[], xlabel="Released answers right (%)")
    bx.spines["left"].set_visible(False)
    # c: share of looping samples that finish at temperature 1.0, by loop period
    br = json.load(open(os.path.join(R, "abstain.json")))["b"]["t1_break_by_period_words"]
    cx = place(fig, 372, 448, 28, 142)
    bins = list(br["MedQA"])
    for ds, col, fill in (("MedQA", REF, False), ("MMLU-Pro health", REF, False), ("GSM8K", OTHER, True),
                          ("MATH-500", OTHER, True)):                # filled: mathematics, open: medicine (as in b)
        v = [br[ds][b][1] * 100 for b in bins]
        cx.plot(range(4), v, color=col, lw=1.0, marker="o", ms=3.4, mfc=col if fill else "white", mec=col, mew=0.9)
    cx.text(-0.1, 87, "MedQA,\nMMLU-Pro h.", fontsize=LAB, va="top", linespacing=1.25)
    cx.text(-0.1, 41, "GSM8K", fontsize=LAB, va="bottom", color=MUTED)
    cx.text(-0.1, 18, "MATH-500", fontsize=LAB, va="bottom", color=MUTED)
    cx.set(xticks=range(4), xticklabels=["≤" + b.split("-")[1] for b in bins], xlim=(-0.3, 3.3), ylim=(0, 102),
           xlabel="Loop period (words)", ylabel="Loops ended at T = 1.0 (%)")
    finish(fig, "fig4_medicine", [a, bx, cx])


def fig5():
    """Fig. 5, one claim: the agent's search found a better direction, and damping the loop along it suffices. a (primary):
    accuracy and cut answers against strength c; b (mechanism): looping tokens pushed past zero along u; c (decomposition):
    direction against neurons"""
    d = json.load(open(os.path.join(R, "csweep.json")))
    A, flip = d["arms"], d["flip_share"]
    cs, keys = [0, 1, 2, 4, 6, 8], ["base", "c1", "c2", "c4", "c6", "c8"]
    acc = [A[k]["acc"] * 100 for k in keys]
    cut = [A[k]["cut"] * 100 for k in keys]
    fig = canvas(TW, 144)
    a = place(fig, 36, 180, 28, 132)
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
    # b: looping tokens whose component along u the edit flips (pushes past zero), layers 15 and 19
    bx = place(fig, 222, 330, 28, 132)
    bx.axvspan(4, 8, color=RAMP[0], lw=0, zorder=0)
    bx.text(6, 101, "accuracy saturated", fontsize=LAB, color=MUTED, ha="center", va="bottom")
    for key, ls, fill, lab in (("L15_looping", "-", True, "layer 15"), ("L19_looping", (0, (4, 2)), False, "layer 19")):
        v = [flip[key][f"c{c}"] * 100 for c in cs]
        bx.plot(cs, v, color=ALERT, ls=ls, lw=1.0, marker="o", ms=3.0, mfc=ALERT if fill else "white", mec=ALERT, mew=0.8)
        bx.text(8.4, v[-1], lab, fontsize=LAB, va="center")
    bx.set(xlabel="Edit strength $c$", ylabel="Looping tokens\nflipped (%)", xticks=cs, ylim=(0, 100), xlim=(-0.3, 12.2))
    bx.spines["bottom"].set_bounds(0, 8)
    # c: direction against neurons, strength 6
    cx = place(fig, 370, 452, 28, 132)
    e2 = d["e2"]
    vals = {("agent", "agent"): A["c6"]["acc"], ("agent", "text"): A["agentdir_mdcols"]["acc"],
            ("text", "agent"): A["mddir_agentcols"]["acc"], ("text", "text"): A["textbook"]["acc"]}
    for j, nset in enumerate(("agent", "text")):
        for dx, dset, col in ((-0.19, "agent", EDIT), (0.19, "text", OTHER_EDIT)):
            v = vals[(dset, nset)] * 100
            cx.bar(j + dx, v, width=0.36, color=col, linewidth=0)
            cx.text(j + dx, v + 1.5, r1(v), fontsize=LAB, ha="center", va="bottom")
    cx.text(-0.19, 4, "agent's $u$", rotation=90, fontsize=LAB, ha="center", va="bottom", color="white")
    cx.text(0.19, 4, "textbook $u$", rotation=90, fontsize=LAB, ha="center", va="bottom", color=INK)
    de, ne = e2["direction_effect"]["d"] * 100, e2["columns_effect"]["d"] * 100
    cx.text(0.5, 93, f"direction +{r1(de)}\nneurons +{r1(ne)}", fontsize=LAB, ha="center", va="top", linespacing=1.25)
    cx.set(xticks=[0, 1], xticklabels=["agent's\nneurons", "textbook\nneurons"], ylim=(0, 100), xlim=(-0.5, 1.5),
           ylabel="Accuracy (%)")
    cx.tick_params(axis="x", length=0)
    finish(fig, "fig5_search", [a, bx, cx])


def fig_deferral():
    """Fig. 6: MedQA accuracy against the share of questions sent to Qwen3.5-9B: the length threshold, random deferral,
    the loop signal and deferring cut answers, and the agent's edit with or without deferral"""
    d = json.load(open(os.path.join(R, "defer_control.json")))
    fig = canvas(CW, 150)
    ax = place(fig, 34, 206, 28, 142)
    xs, ys = zip(*d["length_curve"])
    ax.plot([0, 100], [d["small_alone"] * 100, d["strong_alone"] * 100], color=REF, lw=0.8, ls=(0, (4, 2.5)), marker="")
    ax.plot([x * 100 for x in xs], [y * 100 for y in ys], color=OTHER, lw=1.0, marker="")
    ax.axhline(d["agents_edit"] * 100, xmax=0.5, color=EDIT, lw=1.0, ls=(0, (1.5, 1.5)))
    L, C = d["triggers"]["loop"], d["triggers"]["cap"]
    F = d["fix_then_defer"]
    r = F["rate"] * 100
    ax.scatter(L["defer_rate"] * 100, L["acc"] * 100, s=26, marker="o", color=ALERT, zorder=4)
    ax.scatter(C["defer_rate"] * 100, C["acc"] * 100, s=22, marker="o", facecolor="white", edgecolor=ALERT, linewidths=0.9, zorder=4)
    ax.scatter(r, F["edit, random"]["acc"] * 100, s=30, marker="P", color=EDIT, linewidths=0, zorder=5)
    ax.text(L["defer_rate"] * 100 - 2, L["acc"] * 100 + 6, "loop signal", fontsize=LAB, ha="right", va="bottom")
    ax.text(C["defer_rate"] * 100 - 2.5, C["acc"] * 100, "cut\nanswers", fontsize=LAB, ha="right", va="center", linespacing=1.25)
    ax.text(r + 2.5, F["edit, random"]["acc"] * 100 - 2, "edit, then random", fontsize=LAB, va="top")
    ax.text(48, 84, "length threshold", fontsize=LAB, color=MUTED, ha="right", va="bottom")
    ax.text(72, 73.5, "random", fontsize=LAB, ha="left", va="top")
    ax.text(51.5, d["agents_edit"] * 100, "agent's edit alone", fontsize=LAB, ha="left", va="center")
    ax.set(xlim=(0, 100), ylim=(35, 95), xlabel="Questions sent to the 9B model (%)", ylabel="MedQA accuracy (%)")
    fig.savefig(os.path.join(OUT, "fig6_deferral.pdf"))
    fig.savefig(os.path.join(OUT, "fig6_deferral.png"), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    fig1()
    fig1_examples()
    k, b, r2 = fig2()
    fig3()
    fig4()
    fig5()
    fig_deferral()
    json.dump(ALIGN, open(os.path.join(OUT, "alignment.json"), "w"), indent=1)
    print(f"fig2b grid fit: accuracy = {k:.3f} x finished + {b:.2f}, R2 {r2:.3f}")
    print({n: v["max_edge_deviation_pt"] for n, v in ALIGN.items()})
