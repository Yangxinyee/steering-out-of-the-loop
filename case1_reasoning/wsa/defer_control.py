"""Deferral against random deferral (CPU, existing outputs). On all 1,273 MedQA questions, the small model's
first sample (Qwen3.5-0.8B, authors' protocol) and Qwen3.5-9B's first sample are read the authors' way. Random deferral
at rate r has expected accuracy (1 - r) * small + r * 9B per question; the loop trigger (recomputed exactly as in
wsa.route_vs_fix, which needs the tokenizer) and the cut trigger are compared with it at their own rates (paired
bootstrap over questions). The accuracy-against-call-rate curve of the length threshold comes from
results/route_vs_fix.json.

    .venv-fwd/bin/python -m wsa.defer_control        (writes results/defer_control.json)
"""
import json
from pathlib import Path

import numpy as np

RUNS = Path("results/runs")


def first(stem):
    return {r["id"]: bool(r["correct"][0]) for r in json.load(open(RUNS / f"{stem}.per_problem.json"))}


def main():
    ids = [json.loads(l)["id"] for l in open("results/data/medqa.jsonl")]
    small, big = first("med_base/think35_medqa"), first("medqa9b/think35_medqa")
    rv = json.load(open("results/route_vs_fix.json"))["domains"]["medqa_full"]
    S = rv["strong"]["9B"]
    s = np.array([small[i] for i in ids], float)
    b = np.array([big[i] for i in ids], float)
    assert abs(s.mean() - rv["small_alone"]["acc"]) < 1e-9 and abs(b.mean() - S["alone_acc"]) < 1e-9
    edit = np.array([first("med_edit/think35_medqa")[i] for i in ids], float)
    assert abs(edit.mean() - rv["fixes"]["authors' edit"]["acc"]) < 1e-9
    from transformers import AutoTokenizer
    from wsa.route_vs_fix import CAP, gens, loop_onset
    tok = AutoTokenizer.from_pretrained("ckpt/base")
    g = gens("med_base/think35_medqa", set(ids))
    fires = {i for i in ids if loop_onset(g[(i, 0)]["text"], tok)[0]}
    cut = {i for i in ids if g[(i, 0)]["finish"] == "length"}
    out = {"small_alone": s.mean(), "strong_alone": b.mean(), "agents_edit": edit.mean(), "triggers": {}}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), (10000, len(ids)))
    for t, D in (("loop", fires), ("cap", cut)):
        e = S["triggers"][t]
        v = np.array([b[n] if i in D else s[n] for n, i in enumerate(ids)], float)
        r = len(D) / len(ids)
        assert abs(r - e["defer_rate"]) < 1e-9 and abs(v.mean() - e["acc"]) < 1e-9, (t, r, v.mean())
        rand = (1 - r) * s + r * b
        d = v - rand
        de = v - edit
        bs, bse = d[idx].mean(1), de[idx].mean(1)
        out["triggers"][t] = {"defer_rate": r, "acc": v.mean(), "random_acc": rand.mean(),
                              "minus_random": [d.mean(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                              "random_minus_edit": [(rand - edit).mean()],
                              "minus_edit": [de.mean(), float(np.percentile(bse, 2.5)), float(np.percentile(bse, 97.5))]}
        o = out["triggers"][t]
        print(f"{t}: rate {r*100:.1f}% acc {v.mean()*100:.1f} random {rand.mean()*100:.1f} "
              f"trigger-random {o['minus_random'][0]*100:+.1f} [{o['minus_random'][1]*100:.1f}, {o['minus_random'][2]*100:.1f}] "
              f"random-edit {(rand-edit).mean()*100:+.1f} trigger-edit {o['minus_edit'][0]*100:+.1f}")
    # fix, then defer, at the loop signal's call rate: the edited model answers; the deferred questions are drawn at
    # random, chosen by the base model's loop signal, or the edited model's longest answers (its own length threshold)
    r = len(fires) / len(ids)
    ge = gens("med_edit/think35_medqa", set(ids))
    elen = np.array([ge[(i, 0)]["n_tok"] for i in ids], float)
    n_def = len(fires)
    longest = set(np.array(ids)[np.argsort(-elen, kind="stable")[:n_def]])
    loopv = np.array([b[n] if i in fires else s[n] for n, i in enumerate(ids)], float)
    arms = {"edit, random": (1 - r) * edit + r * b,
            "edit, base loop signal": np.array([b[n] if i in fires else edit[n] for n, i in enumerate(ids)], float),
            "edit, its longest answers": np.array([b[n] if i in longest else edit[n] for n, i in enumerate(ids)], float)}
    out["fix_then_defer"] = {"rate": r}
    for name, v in arms.items():
        d = loopv - v
        bs = d[idx].mean(1)
        out["fix_then_defer"][name] = {"acc": float(v.mean()), "loop_signal_minus": [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        print(f"fix then defer, {name}: {v.mean()*100:.1f}; loop signal minus it {d.mean()*100:+.1f} [{np.percentile(bs, 2.5)*100:.1f}, {np.percentile(bs, 97.5)*100:.1f}]")
    out["length_curve"] = [[v["defer_rate"], v["acc"]] for v in sorted(S["length_curve"].values(), key=lambda v: v["defer_rate"])]
    # where random deferral matches the agent's edit, and the length threshold at the loop trigger's rate
    out["random_rate_to_match_edit"] = (edit.mean() - s.mean()) / (b.mean() - s.mean())
    print("small", s.mean(), "9B", b.mean(), "edit", edit.mean(), "random rate matching the edit",
          round(out["random_rate_to_match_edit"] * 100, 1))
    Path("results/defer_control.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
