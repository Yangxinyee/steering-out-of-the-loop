"""Low-temperature loop screen: R1-Distill-Qwen-1.5B and Qwen3-1.7B on the 200
MATH level-5 practice problems, one sample, at temperatures 0 and 0.2 (the two lowest of Pipis et al.'s grid, arXiv
2512.12895 v3 §2) beside the cards' 0.6 from the hard screen. Per model and temperature:

  acc, cut           the authors' grading; the share of answers cut at 32,768 tokens
  true_loop          answers cut at the cap whose end repeats by the authors' rule (wsa.grade.looping on the tail):
                     the "loops on > 20% of problems" of the pre-set rule is this share
  pipis_words/tokens answers (cut or not) with some 30-gram at least 20 times, Pipis et al.'s rule; the paper does not
                     say whether its n-grams are words or tokens (its public code covers the synthetic task only), so
                     both are given

    python -m wsa.loop_screen --runs results/runs --out results/loop_screen
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from wsa.grade import TAIL_CHARS, looping

MODELS = {"r1d_qwen15b": ("DeepSeek-R1-Distill-Qwen-1.5B", {"r1d_t0": 0.0, "r1d_t02": 0.2, "r1d_card": 0.6}),
          "qwen3_17b": ("Qwen3-1.7B", {"qwen3_t0": 0.0, "qwen3_t02": 0.2, "qwen3_card": 0.6})}
SET = "math_l5_200"
N, K = 30, 20


def has_ngram(seq, n=N, k=K):
    if len(seq) < n * k:
        return False
    c = Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
    return max(c.values()) >= k


def main(a):
    from transformers import AutoTokenizer
    runs = Path(a.runs)
    out = {}
    for m, (name, protos) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(a.ckpt_root + "/" + m)
        for p, temp in protos.items():
            sub = "screen_hard" if p.endswith("_card") else "lowtemp"
            f = runs / sub / m / f"{p}_{SET}.jsonl"
            if not f.exists():
                continue
            s = json.load(open(f.with_name(f"{p}_{SET}.scores.json")))
            gens = [json.loads(l) for l in open(f)]
            cut = [g for g in gens if g["finish"] == "length"]
            loops = sum(looping(g["text"][-TAIL_CHARS:]) for g in cut)
            pw = sum(has_ngram(g["text"].split()) for g in gens)
            pt = sum(has_ngram(tok(g["text"], add_special_tokens=False)["input_ids"]) for g in gens)
            n = len(gens)
            out[f"{m}/{p}"] = {"model": name, "temperature": temp, "n": n, "missing": s.get("missing"), "acc": s["acc"],
                               "cut": len(cut) / n, "true_loop": loops / n, "true_loop_count": loops, "n_cut": len(cut),
                               "pipis_words": pw / n, "pipis_tokens": pt / n, "mean_tokens": s.get("mean_tokens"),
                               "over_20pct": loops / n > 0.20}
            r = out[f"{m}/{p}"]
            print(f"{name:32s} T={temp:.1f} n={n} acc={100 * r['acc']:5.1f}% cut={100 * r['cut']:5.1f}% "
                  f"true loop={100 * r['true_loop']:5.1f}% ({loops}/{len(cut)} cut) pipis words={100 * r['pipis_words']:5.1f}% "
                  f"tokens={100 * r['pipis_tokens']:5.1f}%")
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--ckpt-root", default="ckpt", help="the two models' folders (their tokenizers)")
    ap.add_argument("--out", default="results/loop_screen")
    main(ap.parse_args())
