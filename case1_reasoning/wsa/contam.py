"""Overlap between the practice problems (the authors' decontaminated MATH / GSM8K train files, which the agent and our
screens drew from) and the test sets: the authors' 13-word window, then nearest neighbours by character n-gram TF-IDF
cosine on normalised text. Writes the flagged test ids, so the Part 1 gains can be recomputed without them.

    python -m wsa.contam --practice <math_train.jsonl> <gsm8k_train.jsonl> --bench <math500.jsonl> <gsm8k.jsonl> \
        --out results/contam.json
"""
import argparse
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

WINDOW = 13
NEAR = 0.8          # cosine at or above which a pair is flagged as a near-duplicate


def norm(t):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t.lower()).split())


def grams(t, n=WINDOW):
    w = t.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main(a):
    prac = [json.loads(l) for f in a.practice for l in open(f)]
    out = {}
    for f in a.bench:
        test = [json.loads(l) for l in open(f)]
        name = f.rsplit("/", 1)[-1].split(".")[0]
        pn, tn = [norm(r["problem"]) for r in prac], [norm(r["problem"]) for r in test]
        index = {}
        for j, t in enumerate(pn):
            for g in grams(t):
                index.setdefault(g, j)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(pn + tn)
        P, T = vec.transform(pn), vec.transform(tn)
        flagged, top = [], []
        for i, (r, t) in enumerate(zip(test, tn)):
            sims = (P @ T[i].T).toarray().ravel()
            j = int(np.argmax(sims))
            hit13 = next((index[g] for g in grams(t) if g in index), None)
            rec = {"id": r["id"], "nearest": prac[j]["id"], "cos": float(sims[j]),
                   "same_answer": str(prac[j].get("answer", "")).strip() == str(r.get("answer", "")).strip(),
                   "window13": prac[hit13]["id"] if hit13 is not None else None}
            top.append(rec)
            if rec["cos"] >= NEAR or rec["window13"]:
                flagged.append(rec)
        top.sort(key=lambda x: -x["cos"])
        out[name] = {"n": len(test), "flagged": flagged, "top20": top[:20],
                     "cos_quantiles": {q: float(np.quantile([x["cos"] for x in top], q)) for q in (0.5, 0.9, 0.99)}}
        print(f"{name}: {len(flagged)} of {len(test)} flagged (cos >= {NEAR} or a shared {WINDOW}-word window); "
              f"nearest-neighbour cosine median {out[name]['cos_quantiles'][0.5]:.2f}, "
              f"99th pct {out[name]['cos_quantiles'][0.99]:.2f}")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--practice", nargs="+", required=True)
    ap.add_argument("--bench", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
