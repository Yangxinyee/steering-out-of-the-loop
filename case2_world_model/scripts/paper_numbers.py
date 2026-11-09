"""Write paper/numbers.tex: every number the draft quotes, as LaTeX macros read from the result files.

    python3 scripts/paper_numbers.py
A run whose evaluation is missing gets "??" so the gap shows in the PDF.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
readout = [json.loads(x) for x in open(R / "readout.jsonl")]
agents = {json.loads(x)["run"]: json.loads(x) for x in open(R / "agent_eval.jsonl")}
parts = {json.loads(x)["run"]: json.loads(x) for x in open(R / "agent_parts.jsonl")}
EDITS = {"insulin_x2": "Ix", "insulin_half": "Ih", "ne_x2": "Nx", "ne_half": "Nh"}
RUNS = {"claude_a": "ClaudeW", "openai_a": "CodexW", "claude_c": "ClaudeB", "openai_c": "CodexB",
        "claude_b": "ClaudeN", "openai_b": "CodexN"}
out = []


def m(name, value):
    out.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def num(x, d=2, sign=False):
    x = 0.0 if round(x, d) == 0 else x
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def ctl(key, method, seed):
    return next(r for r in readout if r["key"] == key and r["method"] == method and r["seed"] == seed)


# truth and the plain model
t = ctl("e03", "es_register", 0)
for e, k in EDITS.items():
    m(f"truth{k}", num(t[e]["true_shift"][0], 1, sign=True))
plain = [ctl("e03", "es_register", s)["insulin_x2"]["ratio"][0] for s in range(3)]
m("plainIx", num(plain[0]))
for e, k in EDITS.items():
    v = [ctl("e03", "es_register", s)[e]["ratio"][0] for s in range(3)]
    m(f"plain{k}Lo", num(min(v))); m(f"plain{k}Hi", num(max(v)))
# seed 0, the agents' model: the plain model, fine-tuning and steering
for meth, tag in (("es_register", "plainZ"), ("ft_pnp", "ftZ"), ("steer_a0.25", "steerZ"), ("steerp_a32", "probeZ")):
    r0 = ctl("e03", meth, 0)
    for e, k in EDITS.items():
        x, lo, hi = r0[e]["ratio"]
        m(f"{tag}{k}", num(x)); m(f"{tag}{k}Lo", num(lo)); m(f"{tag}{k}Hi", num(hi))
# steering and fine-tuning, three seeds
for meth, key in (("steer_a0.25", "steer"), ("ft_pnp", "ft")):
    vals = {e: [ctl("e03", meth, s)[e]["ratio"][0] for s in range(3)] for e in EDITS}
    for e, k in EDITS.items():
        m(f"{key}{k}Lo", num(min(vals[e]))); m(f"{key}{k}Hi", num(max(vals[e])))
pr = {e: [ctl("e03", "steerp_a32", s)[e]["ratio"][0] for s in range(3)] for e in EDITS}
for e, k in EDITS.items():
    m(f"probe{k}Lo", num(min(pr[e]))); m(f"probe{k}Hi", num(max(pr[e])))
null_pr = [ctl("n03", mm, s)["insulin_x2"]["mean_shift"][0] for mm, s in (("steerp_a8", 0), ("steerp_a32", 1), ("steerp_a32", 2))]
m("probeNullLo", num(min(null_pr), 2, True)); m("probeNullHi", num(max(null_pr), 2, True))
null_ft = [ctl("n03", "ft_pnp", s)["insulin_x2"]["mean_shift"][0] for s in range(3)]
m("ftNullLo", num(min(null_ft), 2, True)); m("ftNullHi", num(max(null_ft), 2, True)); m("ftNullZ", num(null_ft[0], 2, True))
null_plain = [ctl("n03", "es_register", s)["insulin_x2"]["mean_shift"][0] for s in range(3)]
m("plainNullLo", num(min(null_plain), 2, True))
m("plainNullHi", num(max(null_plain), 2, True)); m("plainNull", num(null_plain[0], 2, True))
null_st = [ctl("n03", "steer_a0.1", s)["insulin_x2"]["mean_shift"][0] for s in range(3)]
m("steerNullZ", num(null_st[0], 2, True)); m("probeNullZ", num(null_pr[0], 2, True))
m("steerNullHi", num(max(null_st), 2, True)); m("steerNullLo", num(min(null_st), 2, True))
# agents
for run, k in RUNS.items():
    a = agents.get(run)
    if a is None:
        for e, ek in EDITS.items():
            m(f"{k}{ek}", "??"); m(f"{k}{ek}Lo", "??"); m(f"{k}{ek}Hi", "??")
        m(f"{k}Nll", "??")
        continue
    for e, ek in EDITS.items():
        x, lo, hi = (a[e]["ratio"] if a["world"] == "e03" else a[e]["mean_shift"])
        m(f"{k}{ek}", num(x)); m(f"{k}{ek}Lo", num(lo)); m(f"{k}{ek}Hi", num(hi))
    x = a["value_nll_change"]
    m(f"{k}Nll", num(x, 5 if abs(x) < 0.0005 else 4, True))
# information-matched baselines without an agent (scripts/scripted_*.py), seed 0
BASE = {"scriptC": "ScriptC", "scriptS": "ScriptS", "ftC": "FtSized"}
for meth, k in BASE.items():
    r0 = ctl("e03", meth, 0)
    for e, ek in EDITS.items():
        x, lo, hi = r0[e]["ratio"]
        m(f"{k}{ek}", num(x)); m(f"{k}{ek}Lo", num(lo)); m(f"{k}{ek}Hi", num(hi))
x, lo, hi = ctl("n03", "scriptS", 0)["insulin_x2"]["mean_shift"]
m("ScriptSNull", num(x, 2, True)); m("ScriptSNullLo", num(lo, 2, True)); m("ScriptSNullHi", num(hi, 2, True))
for w, wk in (("e03", "Main"), ("n03", "Null")):
    est = json.loads((R / "scripted" / f"estimate_{w}.json").read_text())["pairs"]
    for pair, pk in (("I->glucose_finger", "I"), ("N->map_arterial", "N")):
        m(f"scrEst{wk}{pk}", num(est[pair]["sum8"], 1, True)); m(f"scrEst{wk}{pk}Se", num(est[pair]["sum8_se"], 1))
tc = json.loads((R / "scripted" / "tune_scriptC_e03.json").read_text())
m("scrCRounds", str(len(tc["rounds"])))
for e, ek in EDITS.items():
    m(f"scrCTune{ek}", num(tc["rounds"][-1]["readouts"][e], 1, True))
# per-head losses (percent change)
for run, k in RUNS.items():
    p = parts.get(run)
    if p is not None and "monitor" in p["parts"]:
        m(f"{k}MonitorAbs", num(1000 * p["parts"]["monitor"]["change"], 2, True) if abs(p["parts"]["monitor"]["change"]) >= 5e-6 else "0.00")
    else:
        m(f"{k}MonitorAbs", "??")
    for h in ("value_bin", "code", "action_code", "action_count", "time", "count", "cont", "kl", "monitor", "rate_bin"):
        hk = re.sub(r"[^a-z]", "", h.replace("_bin", "").replace("action_code", "drug"))
        if p is None or h not in p["parts"]:
            m(f"{k}Head{hk}", "??"); continue
        q = p["parts"][h]
        pct = 100 * q["change"] / abs(q["original"])
        m(f"{k}Head{hk}", "0.0" if abs(pct) < 0.05 else num(pct, 1, True))
# cost and run statistics
tot, runs_all = 0.0, 0
for run, k in RUNS.items():
    d = ROOT / "agent_runs" / run
    s = json.loads((d / "state.json").read_text())
    usd = s.get("usd", 0.0)
    tot += usd
    runs_all += s["runs"]
    m(f"{k}Usd", f"{usd:.2f}"); m(f"{k}Turns", str(s["turn"])); m(f"{k}Runs", str(s["runs"]))
    m(f"{k}Passes", str(s["passes"])); m(f"{k}Refused", str(s.get("refusals", 0)))
m("totalUsd", f"{tot:.0f}"); m("totalRuns", f"{runs_all:,}".replace(",", "{,}"))
import numpy as np  # noqa: E402


def psi(world, run_no, key):
    rows = [json.loads(x) for x in open(ROOT / "agent_runs" / "replay" / f"claude_a_{world}" / "outputs.jsonl")]
    text = next(o["output"] for o in rows if o["run"] == run_no)
    return np.array([float(v) for v in re.search(rf"psi {key}\s*\[([^\]]*)\]", text).group(1).split()])


fa = (ROOT / "agent_runs" / "claude_a" / "final_answer.md").read_text()
own = {k: float(re.search(rf"{pat}[^|]*\|[^|]*\|[^|]*\|\s*([−+\-0-9.]+)", fa).group(1).replace("−", "-"))
       for k, pat in (("ins", r"Insulin x2"), ("ne", r"Norepinephrine x2"))}
m("ownIx", num(own["ins"], 1, True)); m("ownNx", num(own["ne"], 1, True))
for key, run_no, tag in (("finger", 16, "Fin"), ("lab", 16, "Lab"), ("art", 17, "Art")):
    a_, b_ = psi("e03", run_no, key), psi("n03", run_no, key)
    m(f"rep{tag}Main", num(a_.sum(), 1, True)); m(f"rep{tag}Null", num(b_.sum(), 1, True))
    ratio = np.cumsum(b_).mean() / np.cumsum(a_).mean()          # the readout averages the cumulative effect over hours 13-24
    m(f"rep{tag}Implied", num((own["ins"] if tag != "Art" else own["ne"]) * ratio, 1, True))
# generalization: interventions no agent queried (results/generalize.json, scripts/generalize_summary.py)
GEN_MODELS = {"Plain": "e03:plain model", "Ft": "e03:fine-tuning", "SteerM": "e03:steering, mean difference",
              "SteerP": "e03:steering, probe direction", "CodexB": "e03:Codex, black box", "ClaudeB": "e03:Claude, black box",
              "CodexW": "e03:Codex, white box", "ClaudeW": "e03:Claude, white box", "Erase": "e03:erasure (diagnostic)",
              "ScriptC": "e03:scripted install, Claude's estimate", "ScriptS": "e03:scripted install, scripted estimate",
              "FtSized": "e03:fine-tuning sized to Claude's estimate"}
GEN_KEYS = {"Izero": ("doses", "insulin_x0@12"), "Ionefive": ("doses", "insulin_x1.5@12"), "Ithree": ("doses", "insulin_x3@12"),
            "Ilab": ("doses", "insulin_x2_lab@12"), "Isix": ("starts", "insulin_x2@6"), "Isixteen": ("starts", "insulin_x2@16"),
            "Ihsix": ("starts", "insulin_half@6"), "Ihsixteen": ("starts", "insulin_half@16"),
            "Nzero": ("doses", "ne_x0@12"), "Nonefive": ("doses", "ne_x1.5@12"), "Nthree": ("doses", "ne_x3@12"),
            "Ncuff": ("doses", "ne_x2_cuff@12"), "Nsix": ("starts", "ne_x2@6"), "Nsixteen": ("starts", "ne_x2@16"),
            "Nhsix": ("starts", "ne_half@6"), "Nhsixteen": ("starts", "ne_half@16")}
gpath = ROOT / "results" / "generalize.json"
gen = json.loads(gpath.read_text()) if gpath.is_file() else {}
for mk, gk in GEN_MODELS.items():
    r = gen.get(gk, {})
    for kk, (design, e) in GEN_KEYS.items():
        v = (r.get(design) or {}).get(e)
        m(f"gen{mk}{kk}", num(v["ratio"][0]) if v and "ratio" in v else "??")
    d = (r.get("detail") or {}).get("insulin_x2")
    m(f"gen{mk}Early", num(d["horizon"]["hours 13-16"][0]) if d else "??")
    m(f"gen{mk}Late", num(d["horizon"]["hours 21-24"][0]) if d else "??")
    m(f"gen{mk}Corr", num(d["per_stay"]["corr"][0]) if d else "??")
    m(f"gen{mk}CorrLo", num(d["per_stay"]["corr"][1]) if d else "??")
    m(f"gen{mk}CorrHi", num(d["per_stay"]["corr"][2]) if d else "??")
    m(f"gen{mk}Rmse", f"{d['per_stay']['rmse']:.1f}" if d else "??")
    dn = (r.get("detail") or {}).get("ne_x2")
    m(f"gen{mk}NEarly", num(dn["horizon"]["hours 13-16"][0]) if dn else "??")
    m(f"gen{mk}NLate", num(dn["horizon"]["hours 21-24"][0]) if dn else "??")
# x0 against each edit's own response at x1/2 (a response linear in the dose gives twice the x1/2 shift at x0)
for mk, gk in (("ClaudeW", "e03:Claude, white box"), ("CodexW", "e03:Codex, white box"), ("Ft", "e03:fine-tuning")):
    r = gen.get(gk, {})
    z, h = (r.get("doses") or {}).get("insulin_x0@12"), (r.get("original") or {}).get("insulin_half")
    if z and h:
        m(f"gen{mk}IzeroShift", num(z["model_shift"][0], 1, True)); m(f"gen{mk}IhalfLin", num(2 * h["model_shift"][0], 1, True))
        m(f"gen{mk}IzeroFrac", f"{100 * z['model_shift'][0] / (2 * h['model_shift'][0]):.0f}")
        if mk == "ClaudeW":
            m("genTruthIzero", num(z["true_shift"][0], 1, True))
d0 = (gen.get("e03:plain model", {}).get("detail") or {}).get("insulin_x2")
m("genTruthSd", f"{d0['per_stay']['truth_sd']:.1f}" if d0 else "??")
for wk, mk in (("n03:plain model", "PlainN"), ("n03:fine-tuning", "FtN"), ("n03:Codex, white box", "CodexN"), ("n03:Claude, white box", "ClaudeN"),
               ("n03:scripted install, scripted estimate", "ScriptSN")):
    r = gen.get(wk, {}).get("doses") or {}
    for kk, e in (("Izero", "insulin_x0@12"), ("Ionefive", "insulin_x1.5@12"), ("Ithree", "insulin_x3@12"), ("Ilab", "insulin_x2_lab@12"),
                  ("Nzero", "ne_x0@12"), ("Nonefive", "ne_x1.5@12"), ("Nthree", "ne_x3@12"), ("Ncuff", "ne_x2_cuff@12")):
        v = r.get(e)
        m(f"gen{mk}{kk}", num(v["model_shift"][0], 2, True) if v else "??")
# a stopped drug given to the pathway as rate 0 with the whole expected dose subtracted (scripted install, Claude's estimate)
z = gen.get("e03:scripted install, Claude's estimate, stopped dose as rate 0", {}).get("doses") or {}
for kk, e in (("Izero", "insulin_x0@12"), ("Nzero", "ne_x0@12")):
    v = z.get(e)
    m(f"genStopZero{kk}", num(v["ratio"][0]) if v else "??")
    m(f"genStopZero{kk}Shift", num(v["model_shift"][0], 1, True) if v else "??")
    m(f"genStopZero{kk}Truth", num(v["true_shift"][0], 1, True) if v else "??")
# per-stay agreement on the queried x2 questions, with a dose-only reference (results/per_stay.json, scripts/per_stay.py)
ps_path = ROOT / "results" / "per_stay.json"
ps = json.loads(ps_path.read_text())["methods"] if ps_path.is_file() else {}
for lab, mk in (("plain model", "Plain"), ("fine-tuning", "Ft"), ("Codex, white box", "CodexW"), ("Claude, white box", "ClaudeW"),
                ("dose only (Claude's kernel)", "Dose"), ("Codex, black box", "CodexB"), ("Claude, black box", "ClaudeB"),
                ("steering, mean difference", "SteerM"), ("steering, probe direction", "SteerP"),
                ("scripted install, Claude's estimate", "ScriptC"), ("scripted install, scripted estimate", "ScriptS"),
                ("fine-tuning sized to Claude's estimate", "FtSized")):
    for q, qk in (("insulin_x2", "I"), ("ne_x2", "N")):
        r = ps.get(lab, {}).get(q)
        m(f"ps{mk}{qk}Corr", num(r["corr"][0]) if r else "??")
        m(f"ps{mk}{qk}Rmse", f"{r['rmse']:.1f}" if r else "??")
        m(f"ps{mk}{qk}Sf", f"{r['rmse_scale_free']:.1f}" if r else "??")
        m(f"ps{mk}{qk}Scale", num(r["best_scale"]) if r else "??")
# Codex's main-world estimator replayed unchanged on each world's records (agent_runs/replay/openai_a_<world>, its runs 1-24)
for w, wk in (("e03", "Main"), ("n03", "Null")):
    f = ROOT / "agent_runs" / "replay" / f"openai_a_{w}" / "outputs.jsonl"
    rows = [json.loads(x) for x in open(f)] if f.is_file() else []
    o23 = next((o["output"] for o in rows if o["run"] == 23), "")
    o24 = next((o["output"] for o in rows if o["run"] == 24), "")
    fit = re.search(r"RD dynamic fit \[effect, persistence\]: \[\s*([-0-9.e]+)\s+([-0-9.e]+)\]", o23)
    tg = re.search(r"train / first 100 tuning: \(([-0-9.e]+), \d+\) \(([-0-9.e]+), \d+\)", o24)
    m(f"codexRep{wk}Beta", num(float(fit.group(1)), 2, True) if fit else "??")
    m(f"codexRep{wk}Rho", num(float(fit.group(2)), 2) if fit else "??")
    m(f"codexRep{wk}", num(float(tg.group(1)), 1, True) if tg else "??")
# Codex's white-box readouts on the 400 tuning stays, final weights (final_answer.md, from its tool passes)
fo = (ROOT / "agent_runs" / "openai_a" / "final_answer.md").read_text()
for pat, mk in ((r"Insulin ×2", "Ix"), (r"Insulin ×½", "Ih"), (r"Norepinephrine ×2", "Nx"), (r"Norepinephrine ×½", "Nh")):
    v = re.search(rf"{pat}\s*\|\s*([−+\-0-9.]+)", fo)
    m(f"codexTune{mk}", num(float(v.group(1).replace("−", "-")), 1, True) if v else "??")
# total held-out NLL per stay, every head (results/model_parts.jsonl, scripts/model_parts.py)
mp_path = ROOT / "results" / "model_parts.jsonl"
mp, mp_parts = {}, {}
if mp_path.is_file():
    for x in open(mp_path):
        j = json.loads(x)
        mp[(j["world"], j["label"])] = j["total_nll_per_stay"]
        mp_parts[(j["world"], j["label"])] = j["parts"]["value_bin"]
for (w, lab), mk in {("e03", "steering, mean difference"): "SteerM", ("e03", "steering, probe direction"): "SteerP",
                     ("e03", "fine-tuning"): "Ft", ("e03", "erasure (diagnostic)"): "Erase", ("e03", "openai_c"): "CodexB",
                     ("e03", "claude_c"): "ClaudeB", ("e03", "openai_a"): "CodexW", ("e03", "claude_a"): "ClaudeW",
                     ("n03", "fine-tuning"): "FtN", ("n03", "openai_b"): "CodexN", ("n03", "claude_b"): "ClaudeN",
                     ("e03", "scriptC"): "ScriptC", ("e03", "scriptS"): "ScriptS", ("e03", "ftC"): "FtSized",
                     ("n03", "scriptS"): "ScriptSN"}.items():
    t = mp.get((w, lab))
    m(f"tot{mk}", (num(t["change"], 2, True) if abs(t["change"]) < 0.095 else num(t["change"], 1, True)) if t else "??")
    if mk == "SteerM" and t:
        m("totPlain", f"{t['plain']:.1f}")
for (w, lab), mk in {("e03", "scriptC"): "ScriptC", ("e03", "scriptS"): "ScriptS", ("e03", "ftC"): "FtSized",
                     ("n03", "scriptS"): "ScriptSN"}.items():
    vb = mp_parts.get((w, lab))
    m(f"{mk}Nll", num(vb["edited"] - vb["plain"], 3, True) if vb else "??")
t = mp.get(("n03", "fine-tuning"))
m("totPlainN", f"{t['plain']:.1f}" if t else "??")
dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "numbers.tex"     # default: the standalone paper's
dest.write_text("% generated by scripts/paper_numbers.py; do not edit\n" + "\n".join(out) + "\n")
print(len(out), "macros")
