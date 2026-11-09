"""Data for the same-model reference that adds ability (the distilled 0.8B).

pool    (workstation) training problems for the teacher: the authors' decontaminated MATH and GSM8K train files (the
        agent's practice pool), minus every problem that shares a 13-word window with a MATH-500 or GSM8K test problem
        or has character n-gram TF-IDF cosine >= 0.8 with one (wsa.contam's rule, applied from the training side);
        3,000 MATH and 1,000 GSM8K problems drawn with seed 0. Each row carries max_tokens, the teacher's cap.
        Also writes the rows of GSM8K's always-looping test problems (all four base samples cut), where the reference
        is evaluated on GSM8K.
filter  (node) keeps the teacher's finished, correct solutions as SFT rows {id, problem, answer, text}; if the teacher's
        cell ran out of time before scoring, it scores the solutions written so far first (wsa.score).

    python -m wsa.sft_data pool --practice <math_train.jsonl> <gsm8k_train.jsonl> --bench results/data/math500.jsonl \
        results/data/gsm8k.jsonl --out results/data/sft_pool.jsonl --always-out results/data/gsm8k_always.jsonl
    python -m wsa.sft_data filter --gen <teacher .jsonl> --out data/sft/train.jsonl --rows data/sft/pool.jsonl
"""
import argparse
import json
import random

N_MATH, N_GSM, TEACHER_CAP = 3000, 1000, 12288


def pool(a):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    from wsa.contam import NEAR, grams, norm
    prac = [json.loads(l) for f in a.practice for l in open(f)]
    test = [json.loads(l) for f in a.bench for l in open(f)]
    pn, tn = [norm(r["problem"]) for r in prac], [norm(r["problem"]) for r in test]
    tgrams = set().union(*(grams(t) for t in tn))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(pn + tn)
    P, T = vec.transform(pn), vec.transform(tn)
    best = np.zeros(len(prac))
    for i in range(0, len(prac), 2000):
        best[i:i + 2000] = (P[i:i + 2000] @ T.T).max(1).toarray().ravel()
    drop = [best[j] >= NEAR or bool(grams(pn[j]) & tgrams) for j in range(len(prac))]
    keep = [r for r, d in zip(prac, drop) if not d]
    rng = random.Random(0)
    math_ = [r for r in keep if r["source"] == "math"]
    gsm = [r for r in keep if r["source"] == "gsm8k"]
    rows = rng.sample(math_, N_MATH) + rng.sample(gsm, N_GSM)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "source": r["source"], "level": r.get("level"), "problem": r["problem"],
                                "answer": r["answer"], "max_tokens": TEACHER_CAP}, ensure_ascii=False) + "\n")
    print(f"{len(prac)} training problems, {sum(drop)} near-duplicates of a test problem dropped; "
          f"drew {N_MATH} of {len(math_)} MATH and {N_GSM} of {len(gsm)} GSM8K -> {a.out}")
    if a.always_out:
        from wsa.rescue import load
        base = load(__import__("pathlib").Path("results/runs"), "base/think35_gsm8k", "math")
        always = {p for p in {p for p, _ in base} if all(not base[(p, k)][1] for k in range(4))}
        gs = [json.loads(l) for l in open(a.bench[1])]
        with open(a.always_out, "w") as f:
            for r in gs:
                if r["id"] in always:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(always)} always-looping GSM8K test problems -> {a.always_out}")


def filt(a):
    import os
    import subprocess
    import sys
    rows = {r["id"]: r for r in (json.loads(l) for l in open(a.rows))}
    pp = a.gen.replace(".jsonl", ".per_problem.json")
    if not os.path.exists(pp):              # the teacher's cell ran out of time: score what it wrote (wsa.score marks
        print(f"{pp} missing: scoring the {sum(1 for _ in open(a.gen))} solutions written", flush=True)
        subprocess.run([sys.executable, "-m", "wsa.score", "--rows", a.rows, "--gen", a.gen, "--samples", "1"])
    corr = {}
    for r in json.load(open(pp)):
        corr[r["id"]] = r["correct"][0]
    n, kept = 0, 0
    with open(a.out, "w") as f:
        for line in open(a.gen):
            g = json.loads(line)
            n += 1
            if g["finish"] == "length" or not corr.get(g["id"]) or "</think>" not in g["text"]:
                continue
            r = rows[g["id"]]
            f.write(json.dumps({"id": g["id"], "source": r["source"], "problem": r["problem"], "answer": r["answer"],
                                "text": g["text"], "n_tok": g["n_tok"]}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept {kept} of {n} teacher solutions (finished, correct, thinking closed) -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--practice", nargs=2, required=True)
    p.add_argument("--bench", nargs=2, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--always-out")
    p = sub.add_parser("filter")
    p.add_argument("--gen", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    pool(a) if a.cmd == "pool" else filt(a)
