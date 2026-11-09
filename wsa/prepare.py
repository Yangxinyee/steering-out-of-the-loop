"""Benchmarks and practice subsets (all public data), as jsonl rows {id, problem, answer}.

  bench/math500.jsonl   HuggingFaceH4/MATH-500 test, id = unique_id (the authors' ids)
  bench/gsm8k.jsonl     openai/gsm8k main test, id = gsm8k_test_<i> (the authors' ids)
  practice/sel.jsonl    250 MATH train + 250 GSM8K train problems from the authors' decontaminated practice files:
                        where decoding settings and edit strengths are chosen (never the test sets)
  practice/dir.jsonl    another 256 + 256: where the mean-difference loop direction is measured

    python -m wsa.prepare <data_dir> <authors_practice_dir>
"""
import json
import os
import random
import sys


def dump(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(path, len(rows))


def main(data, practice):
    from datasets import load_dataset
    dump(os.path.join(data, "bench/math500.jsonl"),
         [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
          for r in load_dataset("HuggingFaceH4/MATH-500", split="test")])
    dump(os.path.join(data, "bench/gsm8k.jsonl"),
         [{"id": f"gsm8k_test_{i}", "problem": r["question"],
           "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
          for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))])
    rng = random.Random(0)
    pools = {}
    for name in ("math_train", "gsm8k_train"):
        rows = [json.loads(l) for l in open(os.path.join(practice, f"{name}.jsonl"))]
        rng.shuffle(rows)
        pools[name] = [{"id": r["id"], "problem": r["problem"], "answer": r["answer"]} for r in rows]
    dump(os.path.join(data, "practice/sel.jsonl"), pools["math_train"][:250] + pools["gsm8k_train"][:250])
    dump(os.path.join(data, "practice/dir.jsonl"), pools["math_train"][250:506] + pools["gsm8k_train"][250:506])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
