"""The c sweep and E2 on the 200 practice problems (sel200, one sample, think35; Qwen3.5-0.8B), CPU only.

  c sweep   the authors' u and 64 columns per layer (15, 19) at W = W0 - c u (uᵀW0): c 1 removes the u component, c 2
            reflects it, c 6 is their edit (bit for bit), c 4 and 8 either side; separates "flip" from "amplify"
            (wsa.svd_edit)
  E2        one source's direction on the other's 64 columns, strength 6: with the authors' edit (their direction,
            their columns = c 6) and part 1's textbook edit (our mean-difference direction, our top-64 columns =
            meandiff_top64_a6) it makes a 2 x 2: which of the two explains the gap between the edits
Base and the textbook edit are the 200-problem subset of their 500-problem practice runs (part 1). Paired bootstrap over
problems, 10,000 resamples. Cut = the answer reached the 32,768-token cap.

    python -m wsa.csweep --runs results/runs --rows results/data/sel200.jsonl --out results/csweep \
        --loopwrite results/runs/loopwrite/loop_write.npz
"""
import argparse
import json
import os

import numpy as np

ARMS = {  # name -> (run directory, file stem)
    "base": ("base", "think35_sel"),
    "c1": ("csweep/c1", "think35_sel200"),
    "c2": ("csweep/c2", "think35_sel200"),
    "c4": ("csweep/c4", "think35_sel200"),
    "c6": ("csweep/c6", "think35_sel200"),
    "c8": ("csweep/c8", "think35_sel200"),
    "textbook": ("meandiff_top64_a6", "think35_sel"),
    "agentdir_mdcols": ("e2/e2_agentdir_meandiffcols", "think35_sel200"),
    "mddir_agentcols": ("e2/e2_meandiffdir_agentcols", "think35_sel200"),
}


def load(runs, d, stem, ids):
    pp = os.path.join(runs, d, f"{stem}.per_problem.json")
    if not os.path.exists(pp):
        return None
    acc = {r["id"]: float(np.mean(r["correct"])) for r in json.load(open(pp))}
    cut, tok = {}, {}
    for l in open(os.path.join(runs, d, f"{stem}.jsonl")):
        r = json.loads(l)
        cut.setdefault(r["id"], []).append(r["finish"] == "length")
        tok.setdefault(r["id"], []).append(r["n_tok"])
    if not all(i in acc for i in ids):
        return None
    return {k: np.array([np.mean(v[i]) for i in ids]) for k, v in (("acc", acc), ("cut", cut), ("tok", tok))}


def boot(d, n=10000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def diff(x, y):
    d = x - y
    return {"d": float(d.mean()), "ci95": boot(d)}


def main(a):
    ids = [json.loads(l)["id"] for l in open(a.rows)]
    A = {k: v for k, (d, s) in ARMS.items() if (v := load(a.runs, d, s, ids)) is not None}
    out = {"n_problems": len(ids), "arms": {}, "vs": {}}
    for k, v in A.items():
        out["arms"][k] = {"acc": float(v["acc"].mean()), "cut": float(v["cut"].mean()), "mean_tokens": float(v["tok"].mean())}
    pairs = [(f"c{c}", "base") for c in (1, 2, 4, 6, 8)] + [(f"c{c}", "c6") for c in (1, 2, 4, 8)]
    pairs += [("c2", "c1"), ("c4", "c2"), ("textbook", "base"), ("c6", "textbook")]
    pairs += [("agentdir_mdcols", "textbook"), ("agentdir_mdcols", "c6"), ("mddir_agentcols", "textbook"), ("mddir_agentcols", "c6")]
    for x, y in pairs:
        if x in A and y in A:
            out["vs"][f"{x} - {y}"] = {"acc": diff(A[x]["acc"], A[y]["acc"]), "cut": diff(A[x]["cut"], A[y]["cut"])}
    if all(k in A for k in ("c6", "textbook", "agentdir_mdcols", "mddir_agentcols")):
        # 2 x 2: direction (agent / ours) x columns (agent / ours); main effects averaged over the other factor
        ag, md = A["c6"]["acc"], A["textbook"]["acc"]
        ad_mc, md_ac = A["agentdir_mdcols"]["acc"], A["mddir_agentcols"]["acc"]
        out["e2"] = {"direction_effect": diff((ag + ad_mc) / 2, (md_ac + md) / 2),
                     "columns_effect": diff((ag + md_ac) / 2, (ad_mc + md) / 2),
                     "interaction": diff(ag - ad_mc, md_ac - md)}
    if a.loopwrite:
        # first order, from the base model's own activations: the edit at c changes the layer's output along u by
        # -c s, so the share of tokens whose residual ends on the -u side is the share with h·u - c s < 0
        z = np.load(a.loopwrite)
        out["flip_share"] = {f"L{L}_{st}": {f"c{c}": float(((z[f"L{L}_{st}_hu"] - c * z[f"L{L}_{st}_s"]) < 0).mean())
                                            for c in (0, 1, 2, 4, 6, 8)}
                             for L in (15, 19) for st in ("productive", "looping", "closing")}
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--loopwrite", help="wsa.loop_write's .npz: share of tokens the edit at each c pushes to -u")
    main(ap.parse_args())
