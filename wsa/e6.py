"""E6: the authors' protocol with an 81,920-token cap (think35_81k) on MATH-500, base Qwen3.5-0.8B (1 sample) and the authors' edit (4 samples); CPU only. Paired comparisons use sample 0 of both (the edit's 4-sample
numbers are reported beside). "Cut at 32,768" grades the same answers truncated at 32,768 tokens (the authors' cap),
so the 32k and 81k numbers differ only by the extra budget, not by sampling.

    python -m wsa.e6 --runs results/runs --rows results/data/math500.jsonl --tok ckpt/base --out results/e6
"""
import argparse
import json

import numpy as np

from wsa.csweep import boot
from wsa.grade import grade
from wsa.score import offsets

C32 = 32768


def arm(path, rows, tok, k=0):
    """per problem (sample k): correct at 81,920, correct when cut at 32,768, cut at 81,920, reached 32,768, tokens,
    finished without an answer"""
    out = {}
    for line in open(path):
        g = json.loads(line)
        if g["k"] != k or g["id"] not in rows:
            continue
        r = rows[g["id"]]
        pred, good = grade(r, g["text"])
        if g["n_tok"] > C32:
            offs, _ = offsets(tok, g["text"])
            good32 = grade(r, g["text"][:offs[C32 - 1][1]])[1] if len(offs) >= C32 else good
        else:
            good32 = good
        out[g["id"]] = {"ok": bool(good), "ok32": bool(good32), "cut": g["finish"] == "length", "reach32": g["n_tok"] >= C32,
                        "tok": g["n_tok"], "noans": pred is None and g["finish"] != "length"}
    return out


def main(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    rows = {r["id"]: r for r in (json.loads(l) for l in open(a.rows))}
    B = arm(f"{a.runs}/e6/base/think35_81k_math500.jsonl", rows, tok)
    E = arm(f"{a.runs}/e6/edit_authors/think35_81k_math500.jsonl", rows, tok)
    ids = sorted(set(B) & set(E))
    v = lambda X, f: np.array([float(X[i][f]) for i in ids])
    out = {"n_problems": len(ids), "arms": {}, "paired": {}}
    for name, X in (("base", B), ("edit", E)):
        out["arms"][name] = {"acc_81920": v(X, "ok").mean(), "acc_cut_at_32768": v(X, "ok32").mean(),
                             "cut_81920": v(X, "cut").mean(), "reached_32768": v(X, "reach32").mean(),
                             "mean_tokens": v(X, "tok").mean(), "finished_no_answer": int(v(X, "noans").sum())}
        r32 = [i for i in ids if X[i]["reach32"]]
        out["arms"][name]["of_those_reaching_32768"] = {
            "n": len(r32), "finished_by_81920": int(sum(not X[i]["cut"] for i in r32)),
            "right_at_81920": int(sum(X[i]["ok"] for i in r32)), "right_when_cut_at_32768": int(sum(X[i]["ok32"] for i in r32))}
    for label, x, y in (("base: 81,920 - cut at 32,768", v(B, "ok"), v(B, "ok32")),
                        ("edit: 81,920 - cut at 32,768", v(E, "ok"), v(E, "ok32")),
                        ("at 81,920: edit - base", v(E, "ok"), v(B, "ok")),
                        ("cut at 32,768: edit - base", v(E, "ok32"), v(B, "ok32")),
                        ("base at 81,920 - edit cut at 32,768", v(B, "ok"), v(E, "ok32"))):
        d = x - y
        out["paired"][label] = {"d": float(d.mean()), "ci95": boot(d)}
    s = json.load(open(f"{a.runs}/e6/edit_authors/think35_81k_math500.scores.json"))
    out["edit_4_samples"] = {k: s[k] for k in ("acc", "cut_share", "mean_tokens", "no_answer_finished", "total")}
    out = json.loads(json.dumps(out, default=float))
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
