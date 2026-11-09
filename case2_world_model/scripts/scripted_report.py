"""Collect the information-matched baselines beside Claude, Codex and fine-tuning in one place.

    python3 scripts/scripted_report.py   -> results/scripted_baseline.json and a markdown table on stdout. CPU.
Sources: results/scripted/tune_<variant>.json (tuning-stay sizing and the factual check), results/readout.jsonl (held-out
queried readouts), results/generalize.json (unqueried interventions; run generalize_summary.py first), results/per_stay.json,
results/model_parts.jsonl (total held-out NLL).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
ROWS = [  # label, readout method (e03), generalize key, per-stay label, model_parts (world, label), tune variant
    ("fine-tuning", "ft_pnp", "e03:fine-tuning", "fine-tuning", ("e03", "fine-tuning"), None),
    ("Codex, white box", "agent_openai_a", "e03:Codex, white box", "Codex, white box", ("e03", "openai_a"), None),
    ("Claude, white box", "agent_claude_a", "e03:Claude, white box", "Claude, white box", ("e03", "claude_a"), None),
    ("scripted install, Claude's estimate", "scriptC", "e03:scripted install, Claude's estimate",
     "scripted install, Claude's estimate", ("e03", "scriptC"), "scriptC_e03"),
    ("fine-tuning sized to Claude's estimate", "ftC", "e03:fine-tuning sized to Claude's estimate",
     "fine-tuning sized to Claude's estimate", ("e03", "ftC"), "ftC_e03"),
    ("scripted install, scripted estimate", "scriptS", "e03:scripted install, scripted estimate",
     "scripted install, scripted estimate", ("e03", "scriptS"), "scriptS_e03"),
]
QUERIED = ("insulin_x2", "insulin_half", "ne_x2", "ne_half")
UNQUERIED = ("insulin_x0@12", "insulin_x1.5@12", "insulin_x3@12", "insulin_x2_lab@12", "insulin_x2@6", "insulin_x2@16",
             "ne_x0@12", "ne_x1.5@12", "ne_x3@12", "ne_x2_cuff@12", "ne_x2@6", "ne_x2@16")


def last_readout(method: str) -> dict | None:
    rec = None
    for x in open(R / "readout.jsonl"):
        j = json.loads(x)
        if j["key"] == "e03" and j["method"] == method and j["seed"] == 0:
            rec = j
    return rec


def main() -> None:
    gen = json.loads((R / "generalize.json").read_text())
    ps = json.loads((R / "per_stay.json").read_text())["methods"]
    mp = {}
    for x in open(R / "model_parts.jsonl"):
        j = json.loads(x)
        mp[(j["world"], j["label"])] = j["total_nll_per_stay"]["change"]
    out = {}
    for label, method, gk, pk, mk, tv in ROWS:
        r = {"queried": {}, "unqueried": {}}
        ro = last_readout(method)
        g = gen.get(gk, {})
        for e in QUERIED:
            if ro and ro.get(e, {}).get("ratio"):
                r["queried"][e] = ro[e]["ratio"]
            elif (g.get("original") or {}).get(e):
                r["queried"][e] = g["original"][e]["ratio"]
        for e in UNQUERIED:
            d = "starts" if e.endswith(("@6", "@16")) else "doses"
            v = (g.get(d) or {}).get(e)
            if v and "ratio" in v:
                r["unqueried"][e] = v["ratio"][0]
        det = (g.get("detail") or {}).get("insulin_x2")
        if det:
            r["hours 13-16"], r["hours 21-24"] = det["horizon"]["hours 13-16"][0], det["horizon"]["hours 21-24"][0]
        dn = (g.get("detail") or {}).get("ne_x2")
        if dn:
            r["ne hours 13-16"], r["ne hours 21-24"] = dn["horizon"]["hours 13-16"][0], dn["horizon"]["hours 21-24"][0]
        if pk in ps:
            r["per_stay_insulin"] = ps[pk].get("insulin_x2")
        r["total_nll_change"] = mp.get(mk)
        if tv and (R / "scripted" / f"tune_{tv}.json").is_file():
            t = json.loads((R / "scripted" / f"tune_{tv}.json").read_text())
            r["tuning"] = {"readouts": t["rounds"][-1]["readouts"], "targets": t["targets"], "gains": t.get("final_gains"),
                           "factual_value_nll_change": (t.get("factual") or {}).get("diff")}
        out[label] = r
    for wk in ("n03:scripted install, scripted estimate",):
        g = gen.get(wk, {})
        out[wk] = {e: v["model_shift"][0] for e, v in (g.get("doses") or {}).items()}
    z = gen.get("e03:scripted install, Claude's estimate, stopped dose as rate 0", {})
    out["x0 with the stopped dose as rate 0"] = {e: v.get("ratio", [None])[0] for e, v in (z.get("doses") or {}).items()}
    (R / "scripted_baseline.json").write_text(json.dumps(out, indent=1) + "\n")
    f = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    print("| method | ins ×2 | ins ×½ | NE ×2 | NE ×½ | " + " | ".join(UNQUERIED) + " | h13-16 | h21-24 | ΔNLL/stay |")
    for label, r in out.items():
        if "queried" not in r:
            continue
        q = [f(r["queried"].get(e, [None])[0]) for e in QUERIED]
        u = [f(r["unqueried"].get(e)) for e in UNQUERIED]
        print(f"| {label} | " + " | ".join(q + u) + f" | {f(r.get('hours 13-16'))} | {f(r.get('hours 21-24'))} | "
              f"{f(r.get('total_nll_change'))} |")
    for k in ("n03:scripted install, scripted estimate", "x0 with the stopped dose as rate 0"):
        print(k, out[k])


if __name__ == "__main__":
    main()
