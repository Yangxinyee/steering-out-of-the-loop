"""The two real examples in the paper's teaser, chosen by a fixed rule, not by eye (CPU, existing outputs only).

Rule, per row: among samples (problem, sample index) that the base model cut at the token budget and the agent's edit
finished, on problems where the base model finishes at least one other sample (the same-problem pairing set of the
released-answer analysis), keep those whose released answer is right (MATH-500 row) or wrong (MedQA row), and draw one
with random.Random(SEED). The wrong medical answer is a choice of illustration; the teaser prints the overall
released and own accuracies beside it. The verbatim texts of both outputs are saved so the figure rebuilds from the
repository (the run files themselves are not tracked).

    python -m wsa.teaser_examples        (writes results/teaser_examples.json)
"""
import json
import random
from pathlib import Path

from wsa.rescue import load

RUNS, DATA = Path("results/runs"), Path("results/data")
SEED = 2026
ROWS = {"MATH-500": ("math", "base/think35_math500", "edit_authors/think35_math500", "math500.jsonl", True),
        "MedQA": ("mc", "med_base/think35_medqa", "med_edit/think35_medqa", "medqa.jsonl", False)}


def texts(stem, key):
    for line in open(RUNS / f"{stem}.jsonl"):
        r = json.loads(line)
        if (r["id"], r["k"]) == key:
            return r


def main():
    out = {}
    for name, (kind, base_stem, edit_stem, data, right) in ROWS.items():
        B, E = load(RUNS, base_stem, kind), load(RUNS, edit_stem, kind)
        own = {p for (p, _), v in B.items() if v[1]}
        cand = sorted(k for k in B if k in E and not B[k][1] and E[k][1] and k[0] in own and E[k][0] == right)
        key = random.Random(SEED).choice(cand)
        q = next(r for r in map(json.loads, open(DATA / data)) if r["id"] == key[0])
        b, e = texts(base_stem, key), texts(edit_stem, key)
        out[name] = {"rule": f"seed {SEED} over {len(cand)} candidates", "id": key[0], "k": key[1], "question": q,
                     "base": {"n_tok": b["n_tok"], "finish": b["finish"], "text": b["text"]},
                     "edit": {"n_tok": e["n_tok"], "finish": e["finish"], "text": e["text"], "correct": E[key][0]}}
        print(name, key, len(cand), "candidates; base", b["n_tok"], b["finish"], "edit", e["n_tok"], e["finish"], E[key][0])
    Path("results/teaser_examples.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
