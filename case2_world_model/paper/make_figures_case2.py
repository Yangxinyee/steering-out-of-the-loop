"""Case 2 figures, from the result files only (CPU), in case 1's style (preset B layout, 'signal' colour roles, Arial
metrics, direct labels, no legends, no panel letters). Needs the figure toolkit shared with case 1 (figkit, plotkit,
themes): set PAPER_FIGURES to its directory, or have it on the Python path.

    PAPER_FIGURES=<toolkit dir> python <this folder>/make_figures_case2.py
writes figures/case2_readouts.pdf beside this script (main figure), case2_schematic.pdf (block for the two-column intro schematic),
case2_dose.pdf, case2_cost.pdf, case2_estimator.pdf (appendix), each with a .png beside it.

Colour roles, mapped from case 1 (STYLE.md):
  agent with weight access (the hero)      vivid orange #E8590C, plus P (Claude filled, Codex hollow)
  information-matched script (strongest baseline)  deep blue #0F4D92 hexagon h; fully scripted light blue #4A90D9 X
  input-side (black-box) agent edits       amber #F2A900, triangle ^ (Claude filled, Codex hollow)
  other corrections (fine-tuning, steering) slate, v / s / D
  invented effects, the replayed estimators crimson #C2185B
  references (truth, zero), the unedited model black, open circle for the unedited model
"""
import dataclasses
import json
import os
import re
import sys
from pathlib import Path

if os.environ.get("PAPER_FIGURES"):
    sys.path.insert(0, os.environ["PAPER_FIGURES"])
else:                                                   # the toolkit shipped with case 1 in this repository
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "case1_reasoning" / "paper" / "figures" / "figlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import figkit as fk  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotkit as pk  # noqa: E402
from themes import THEMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TW, CW = 486, 234          # the two-column template's \textwidth and \columnwidth in bp
B = THEMES["B"]
INK, MUTED, RAMP = B.ink, B.muted, B.ramp
EDIT = "#E8590C"           # agent with weight access
STRONG = "#0F4D92"         # information-matched script, Claude's estimate
STRONG2 = "#4A90D9"        # fully scripted pipeline
OTHER_EDIT = "#F2A900"     # input-side agent edits
ALERT = "#C2185B"          # invented effects
OTHER = RAMP[4]            # fine-tuning, steering
REF = "#000000"
LAB = 8
THEME = dataclasses.replace(B, accent=EDIT, accent_tint="#FCE4D6", risk=ALERT, risk_tint="#F9DDE8", ok=INK)
plt.rcParams.update(pk.mpl_style("B"))
plt.rcParams["figure.constrained_layout.use"] = False


def canvas(w, h):
    return plt.figure(figsize=(w / 72, h / 72), layout="none")


def place(fig, x0, x1, y0, y1):
    W, H = fig.get_size_inches() * 72
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


readout = [json.loads(x) for x in open(R / "readout.jsonl")]
agents = {json.loads(x)["run"]: json.loads(x) for x in open(R / "agent_eval.jsonl")}
gen = json.loads((R / "generalize.json").read_text())
NUM = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (HERE / "numbers_case2.tex").read_text()))


def num(k):
    return float(NUM[k].replace("$-$", "-").replace("$+$", ""))


def ctl(key, method, seed=0):
    return next(r for r in readout if r["key"] == key and r["method"] == method and r["seed"] == seed)


# one row per correction, top to bottom: label, colour, marker, filled, size, main-world source, no-effect source,
# generalize key, replayed-estimator macro
ROWS = [
    ("unedited model", REF, "o", False, 22, ("ctl", "es_register"), ("ctl", "es_register"), "e03:plain model", None),
    ("steering, mean difference", OTHER, "v", True, 22, ("ctl", "steer_a0.25"), ("ctl", "steer_a0.1"),
     "e03:steering, mean difference", None),
    ("steering, probe direction", OTHER, "s", True, 18, ("ctl", "steerp_a32"), ("ctl", "steerp_a8"),
     "e03:steering, probe direction", None),
    ("fine-tuning", OTHER, "D", True, 18, ("ctl", "ft_pnp"), ("ctl", "ft_pnp"), "e03:fine-tuning", None),
    ("fine-tuning, sized to estimate", OTHER, "D", False, 18, ("ctl", "ftC"), None,
     "e03:fine-tuning sized to Claude's estimate", None),
    ("Codex, input side", OTHER_EDIT, "^", False, 24, ("agent", "openai_c"), None, "e03:Codex, black box", None),
    ("Claude, input side", OTHER_EDIT, "^", True, 24, ("agent", "claude_c"), None, "e03:Claude, black box", None),
    ("Codex, weight access", EDIT, "P", False, 40, ("agent", "openai_a"), ("agent", "openai_b"), "e03:Codex, white box",
     "wmcodexRepNull"),
    ("Claude, weight access", EDIT, "P", True, 52, ("agent", "claude_a"), ("agent", "claude_b"), "e03:Claude, white box",
     "wmrepFinImplied"),
    ("script, Claude's estimate", STRONG, "h", True, 34, ("ctl", "scriptC"), None,
     "e03:scripted install, Claude's estimate", None),
    ("script, scripted estimate", STRONG2, "X", True, 30, ("ctl", "scriptS"), ("ctl", "scriptS"),
     "e03:scripted install, scripted estimate", None),
]


def value(src, world):
    kind, name = src
    if kind == "ctl":
        r = ctl(world, name)["insulin_x2"]
    else:
        r = agents[name]["insulin_x2"]
    return r["ratio"] if world == "e03" else r["mean_shift"]


def mark(ax, x, y, col, mk, filled, s, z=4):
    face = {"color": col, "linewidths": 0} if filled else {"facecolor": "white", "edgecolor": col, "linewidths": 0.9}
    ax.scatter([x], [y], s=s, marker=mk, zorder=z, clip_on=False, **face)


def fig_readouts():
    """Main figure, one claim: weight access and the information-matched script recover the queried effect; the
    effect's time course and the no-effect world show what that readout hides. Left (hero): insulin x2 readout over
    truth, held-out stays, 95% intervals. Middle: the same readout in hours 13-16 and 21-24. Right: the shift in the
    no-effect world, with the agents' main-world estimators replayed (crosses)."""
    n = len(ROWS)
    fig = canvas(TW, 196)
    a = place(fig, 132, 260, 30, 182)
    b = place(fig, 278, 364, 30, 182)
    c = place(fig, 382, 476, 30, 182)
    ys = np.arange(n)[::-1]
    for (lab, col, mk, fill, s, main, null, gk, rep), y in zip(ROWS, ys):
        x, lo, hi = value(main, "e03")
        a.plot([lo, hi], [y, y], color=col, lw=1.1, marker="", solid_capstyle="butt", zorder=2)
        mark(a, x, y, col, mk, fill, s)
        h = gen[gk]["detail"]["insulin_x2"]["horizon"]
        e, l = h["hours 13-16"][0], h["hours 21-24"][0]
        b.plot([e, l], [y, y], color=col, lw=1.0, marker="", zorder=2)
        b.scatter([e], [y], s=14, marker="o", facecolor="white", edgecolor=col, linewidths=0.9, zorder=3)
        mark(b, l, y, col, mk, fill, s * 0.8)
        if null is None:
            c.text(2.35, y, "not run", fontsize=LAB, color=MUTED, ha="right", va="center")
            continue
        x, lo, hi = value(null, "n03")
        c.plot([lo, hi], [y, y], color=col, lw=1.1, marker="", solid_capstyle="butt", zorder=2)
        mark(c, x, y, col, mk, fill, s)
        if rep:
            r = num(rep)
            c.scatter([r], [y], s=30, marker="x", color=ALERT, linewidths=1.4, zorder=5)
    top = n - 0.35
    for ax in (a, b):
        ax.axvline(1.0, color=REF, lw=0.8, zorder=1)
        ax.text(1.0, top, "truth", fontsize=LAB, ha="center", va="bottom")
    a.set(xlim=(-0.02, 1.12), ylim=(-0.6, n - 0.4), xticks=[0, 0.5, 1.0], yticks=ys,
          xlabel="Insulin ×2, readout / truth")
    a.set_yticklabels([r[0] for r in ROWS])
    for t, r in zip(a.get_yticklabels(), ROWS):
        if r[1] == EDIT:
            t.set_fontweight("bold")
    b.set(xlim=(0, 1.35), ylim=a.get_ylim(), xticks=[0, 0.5, 1.0], yticks=[], xlabel="Readout / truth by hour")
    b.text(1.33, n - 1.25, "open: hours 13–16\nfilled: 21–24", fontsize=LAB, color=MUTED, ha="right", va="center",
           linespacing=1.25, bbox={"fc": "white", "ec": "none", "pad": 0.3}, zorder=6)
    band = num("wmBand")                       # a tenth of the true insulin x2 change
    c.axvspan(-band, band, color=RAMP[0], lw=0, zorder=0)
    c.axvspan(-3.2, -band, color=THEME.risk_tint, lw=0, zorder=0)
    c.axvline(0, color=REF, lw=0.8, zorder=1)
    c.text(0, top, "truth", fontsize=LAB, ha="center", va="bottom")
    c.text(-3.1, 0.2, "invented\neffect", fontsize=LAB, color=ALERT, ha="left", va="center", linespacing=1.25)
    yc = ys[[i for i, r in enumerate(ROWS) if r[0] == "Codex, weight access"][0]]
    c.text(num("wmcodexRepNull"), yc + 0.42, "estimator,\nreplayed", fontsize=LAB, color=ALERT, ha="center", va="bottom",
           linespacing=1.15, zorder=6)
    c.set(xlim=(-3.2, 2.4), ylim=a.get_ylim(), xticks=[-2, 0, 2], yticks=[], xlabel="No-effect shift (mg/dL)")
    for ax in (a, b, c):
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    save(fig, "case2_readouts")


def fig_schematic():
    """Block for the merged paper's two-column intro schematic: a patient world model whose drug effect the dosing
    policy cancels; an agent edits its weights from the records and a forward-pass tool, without the truth; the edit
    is judged on held-out stays with known truth and in a world where the drugs are inert."""
    T = fk.use_theme(THEME, width_pt=CW, font="arial")
    plain = num("wmplainIx")
    H = 640
    d = fk.Figure("case2_schematic", H, "Case 2: drug effects in a patient world model", out=OUT)
    d.text(0, 0, 1000, "patient world model", T.zone, True)
    d.box(0, 52, 1000, 150, T.panel, None, T.r_outer)
    fk.tile(d, "patient", 26, 74, 104)
    d.text(160, 72, 820, "simulated ICU records; the dosing\npolicy cancels the drug's effect", T.small)
    d.text(160, 150, 820, f"insulin ×2: {plain:.2f} of the true change", T.small, True, color=T.risk)
    d.line([(500, 210), (500, 252)], INK, arrow=True)
    d.box(0, 260, 1000, 132, T.accent_tint, T.accent, T.r_outer, 1.6 * T.lw_box)
    d.text(26, 276, 950, "agent edits the weights", T.body, True)
    d.text(26, 322, 950, "Claude or Codex; records + forward-pass tool, no truth", T.small, color=MUTED)
    d.line([(250, 400), (250, 442)], INK, arrow=True)
    d.line([(750, 400), (750, 442)], INK, arrow=True)
    for x, head, sub, tint in ((0, "known truth", "held-out stays,\nunqueried doses", T.panel),
                               (520, "no-effect world", "same doctors,\ninert drugs", T.risk_tint)):
        d.box(x, 450, 480, 182, tint, None, T.r_outer)
        d.text(x + 24, 466, 440, head, T.body, True)
        d.text(x + 24, 514, 440, sub, T.small, color=MUTED)
    d.save()
    fk.check_errors()
    fk.render(OUT / "case2_schematic.pdf")


GEN_ROWS = [("unedited model", "e03:plain model", REF, "o", False), ("fine-tuning", "e03:fine-tuning", OTHER, "D", True),
            ("Codex", "e03:Codex, white box", EDIT, "P", False), ("Claude", "e03:Claude, white box", EDIT, "P", True),
            ("script", "e03:scripted install, Claude's estimate", STRONG, "h", True)]


def fig_dose():
    """Appendix: predicted and true change against the dose factor from hour 12; only x1/2 and x2 were queried."""
    fig = canvas(TW, 150)
    axes = [place(fig, 42, 182, 30, 140), place(fig, 294, 420, 30, 140)]
    for ax, drug, unit in zip(axes, ("insulin", "ne"), ("Glucose change (mg/dL)", "MAP change (mmHg)")):
        name = {0.0: f"{drug}_x0@12", 0.5: f"{drug}_half", 1.5: f"{drug}_x1.5@12", 2.0: f"{drug}_x2", 3.0: f"{drug}_x3@12"}
        for q in (0.5, 2.0):
            ax.axvspan(q - 0.06, q + 0.06, color=RAMP[1], lw=0, zorder=0)
        truth = None
        ends = {}
        for lab, key, col, mk, fill in GEN_ROWS:
            r = gen[key]
            xs, ys, tr = [], [], []
            for f, e in sorted(name.items()):
                src = r.get("original", {}).get(e) if "@" not in e else r.get("doses", {}).get(e)
                xs.append(f); ys.append(src["model_shift"][0]); tr.append(src["true_shift"][0])
            truth = (xs, tr)
            ax.plot(xs, ys, color=col, lw=1.0, marker=mk, ms=4 if mk != "P" else 5.5, mfc=col if fill else "white",
                    mec=col, mew=0.9, zorder=3)
            ends[lab] = ys[-1]
        ax.plot(*truth, color=REF, lw=1.1, ls=(0, (4, 2.5)), marker="", zorder=4)
        ends["truth"] = truth[1][-1]
        ax.axhline(0, color=RAMP[2], lw=0.6, zorder=0)
        ax.set(xticks=[0, 0.5, 1.5, 2, 3], xticklabels=["×0", "×½", "×1.5", "×2", "×3"], xlim=(-0.1, 3.0),
               xlabel=("Insulin" if drug == "insulin" else "Norepinephrine") + " dose factor from hour 12", ylabel=unit)
        # direct labels at the right end, spread so they do not overlap
        order = sorted(ends.items(), key=lambda kv: kv[1])
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.085
        placed = []
        for lab, v in order:
            yv = v if not placed or v - placed[-1] >= gap else placed[-1] + gap
            placed.append(yv)
            ax.text(3.06, yv, lab, fontsize=LAB, va="center", ha="left",
                    fontweight="bold" if lab == "Claude" else "normal", clip_on=False)
    axes[0].text(0.5, axes[0].get_ylim()[0], "queried", fontsize=LAB, color=MUTED, ha="center", va="bottom")
    save(fig, "case2_dose")


HEADS = [("value_bin", "reading value"), ("code", "next event type"), ("action_code", "next drug"),
         ("time", "time to next bundle"), ("count", "readings per bundle"), ("action_count", "drugs per bundle"),
         ("cont", "record continues"), ("kl", "latent (KL)"), ("rate_bin", "infusion rate")]


def fig_cost():
    """Appendix: change in held-out loss on every output head after each agent's main-world edit."""
    parts = {json.loads(x)["run"]: json.loads(x) for x in open(R / "agent_parts.jsonl")}
    runs = [("openai_a", "Codex, weight access", EDIT, "P", False), ("claude_a", "Claude, weight access", EDIT, "P", True),
            ("openai_c", "Codex, input side", OTHER_EDIT, "^", False), ("claude_c", "Claude, input side", OTHER_EDIT, "^", True)]
    fig = canvas(CW, 200)
    ax = place(fig, 92, 226, 30, 150)
    ys = np.arange(len(HEADS))[::-1]
    offs = np.linspace(-0.27, 0.27, len(runs))
    for (run, lab, col, mk, fill), dy in zip(runs, offs):
        for (h, _), y in zip(HEADS, ys):
            p = parts[run]["parts"][h]
            pct = 100 * p["change"] / abs(p["original"])
            ax.plot([0, pct], [y + dy] * 2, color=col, lw=0.7, marker="", zorder=2)
            mark(ax, pct, y + dy, col, mk, fill, 20 if mk == "P" else 14)
    ax.axvline(0, color=REF, lw=0.7, marker="")
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.6)
    ax.set(xlim=(-1.2, 600), xticks=[0, 1, 10, 100], xticklabels=["0", "1%", "10%", "100%"], yticks=ys,
           xlabel="Held-out loss change (symlog)", ylim=(-0.6, len(HEADS) - 0.4))
    ax.set_yticklabels([h[1] for h in HEADS])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for i, (run, lab, col, mk, fill) in enumerate(runs):            # key above the plot, as direct labels
        x0, y0 = 6 + (i % 2) * 112, 188 - (i // 2) * 14
        fx = fig.add_axes([x0 / CW, (y0 - 5) / 200, 10 / CW, 10 / 200])
        fx.set_axis_off()
        mark(fx, 0.5, 0.5, col, mk, fill, 20 if mk == "P" else 14)
        fx.set(xlim=(0, 1), ylim=(0, 1))
        fig.text((x0 + 13) / CW, y0 / 200, lab, fontsize=LAB, va="center")
    save(fig, "case2_cost")


def coef(world, run_no, col):
    """the replay's per-lag table (Claude's run printed it): coefficient and SE of column `col` (0 = finger-stick or
    arterial) at lags 1..12, before the feedback of later doses is removed"""
    out = [json.loads(x) for x in open(ROOT / "agent_runs" / "replay" / f"claude_a_{world}" / "outputs.jsonl")]
    text = next(o["output"] for o in out if o["run"] == run_no)
    est, se = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and 1 <= int(parts[0]) <= 12 and "±" in parts[1 + col]:
            v, e = parts[1 + col].split("±")
            est.append(float(v)); se.append(float(e))
    return np.array(est[:12]), np.array(se[:12])


def fig_estimator():
    """Appendix: Claude's estimator replayed unchanged on each world's records: coefficient by lag, +-2 SE, before the
    feedback of later doses is removed."""
    fig = canvas(TW, 140)
    axes = [place(fig, 44, 182, 30, 128), place(fig, 294, 420, 30, 128)]
    lags = np.arange(1, 13)
    for ax, (run_no, unit) in zip(axes, ((15, "Glucose, mg/dL per U/h"), (17, "MAP, mmHg per µg/kg/min"))):
        ax.axhline(0, color=RAMP[2], lw=0.6, zorder=0)
        for world, col, fill, lab in (("e03", REF, True, "main world"), ("n03", ALERT, False, "no-effect world")):
            e, se = coef(world, run_no, 0)
            ax.fill_between(lags, e - 2 * se, e + 2 * se, color=col, alpha=0.12, lw=0)
            ax.plot(lags, e, color=col, lw=1.0, marker="o", ms=3.4, mfc=col if fill else "white", mec=col, mew=0.9)
            ax.text(12.6, e[-1], lab, fontsize=LAB, va="center", color=ALERT if col == ALERT else INK, clip_on=False)
        ax.set(xlabel="Hours after the dose", ylabel=unit, xlim=(0.5, 12.5), xticks=[1, 4, 8, 12])
    save(fig, "case2_estimator")


if __name__ == "__main__":
    fig_readouts()
    fig_schematic()
    fig_dose()
    fig_cost()
    fig_estimator()
    print("figures written to", OUT)
