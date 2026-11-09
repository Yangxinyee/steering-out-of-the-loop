"""Part 2 benchmarks (public data), as jsonl rows {id, kind: "mc", prompt, answer} in the multiple-choice format the
authors used for MMLU-Pro and GPQA ("the official MMLU-Pro prompt and answer extraction"): the official instruction
zero-shot (there are no worked medical examples to put in front), options lettered, answer read by wsa.grade.mc_extract.

  bench/medqa.jsonl        GBaker/MedQA-USMLE-4-options test, 1,273 questions, options A-D
  bench/pubmedqa.jsonl     qiaojin/PubMedQA pqa_labeled restricted to the official 500-question test split
                           (pubmedqa/pubmedqa data/test_ground_truth.json); the abstract without its conclusion
                           (long_answer) as context, options A yes / B no / C maybe
  bench/<set>_s200.jsonl   200 of each, drawn with random.Random(0): the screen
  practice/medqa_dir.jsonl 768 MedQA train questions: where the mean-difference loop direction is measured
  practice/medqa_sel.jsonl another 500 train questions: where the direction's coefficient is picked (never the test set)
  bench/mmlupro_health.jsonl     TIGER-Lab/MMLU-Pro test, category "health" (818 questions, up to 10 options A-J)
  bench/medxpertqa.jsonl         TsinghuaC3I/MedXpertQA, Text test (2,450 questions, 10 options A-J); its question field
                                 ends in the options ("Answer Choices: (A) ..."), which are cut off and put in our format
  (both MIT, revisions pinned below; --extra writes these two and their _s200 draws, as for the first two sets)
  bench/mmlupro_math.jsonl       TIGER-Lab/MMLU-Pro test, category "math" (1,351 questions, up to 10 options A-J): a
                                 mathematics set with headroom for the released-against-own test; the agent's practice data never included MMLU-Pro (--math)

    python -m wsa.prepare_med <data_dir> [--train | --extra | --math]
"""
import json
import os
import random
import re
import sys
import urllib.request

MMLU_PRO = ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be")
MEDXPERTQA = ("TsinghuaC3I/MedXpertQA", "7e7c465a68eb2b866926bfa59c8c9d17a8daba65")
TEST_IDS = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
HEAD = ('The following are multiple choice questions (with answers) about {subject}. Think step by step and then '
        'output the answer in the format of "The answer is (X)" at the end.\n\n')


def mc_prompt(subject, question, options, context=None):
    ctx = f"Context: {context}\n" if context else ""
    opts = "\n".join(f"{k}. {v}" for k, v in options)
    return HEAD.format(subject=subject) + f"{ctx}Question: {question}\nOptions: {opts}\nAnswer: Let's think step by step."


def dump(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(path, len(rows))


def medqa_rows(split):
    from datasets import load_dataset
    return [{"id": f"medqa_{split}_{i}", "kind": "mc", "answer": r["answer_idx"],
             "prompt": mc_prompt("medicine", r["question"], sorted(r["options"].items()))}
            for i, r in enumerate(load_dataset("GBaker/MedQA-USMLE-4-options", split=split))]


def extra_rows():
    from datasets import load_dataset
    letters = "ABCDEFGHIJ"
    mmlu = [{"id": f"mmlupro_{r['question_id']}", "kind": "mc", "answer": r["answer"],
             "prompt": mc_prompt("health", r["question"], list(zip(letters, r["options"])))}
            for r in load_dataset(MMLU_PRO[0], revision=MMLU_PRO[1], split="test") if r["category"] == "health"]
    assert all(r["answer"] in letters for r in mmlu)
    mx = []
    for r in load_dataset(MEDXPERTQA[0], "Text", revision=MEDXPERTQA[1], split="test"):
        # the options as the question field shows them (what MedXpertQA's own evaluation puts in front of a model); in
        # Text-2230 and Text-2270 one wrong option there differs from the options field, the answer never does
        stem, choices = r["question"].split("\nAnswer Choices: ")
        parts = re.split(r"(?:^| )\(([A-J])\) ", choices.strip())
        opts = list(zip(parts[1::2], (p.strip() for p in parts[2::2])))
        assert parts[0] == "" and [k for k, _ in opts] == list(letters), r["id"]
        assert dict(opts)[r["label"]] == r["options"][r["label"]], r["id"]
        mx.append({"id": f"medxpertqa_{r['id']}", "kind": "mc", "answer": r["label"],
                   "prompt": mc_prompt("medicine", stem, opts)})
    assert len(mmlu) == 818 and len(mx) == 2450, (len(mmlu), len(mx))
    return {"mmlupro_health": mmlu, "medxpertqa": mx}


def math_rows():
    from datasets import load_dataset
    letters = "ABCDEFGHIJ"
    rows = [{"id": f"mmlupro_{r['question_id']}", "kind": "mc", "answer": r["answer"],
             "prompt": mc_prompt("math", r["question"], list(zip(letters, r["options"])))}
            for r in load_dataset(MMLU_PRO[0], revision=MMLU_PRO[1], split="test") if r["category"] == "math"]
    assert all(r["answer"] in letters for r in rows) and len(rows) == 1351, len(rows)
    return rows


def main(data, train=False, extra=False, math=False):
    if math:
        dump(os.path.join(data, "bench/mmlupro_math.jsonl"), math_rows())
        return
    if extra:
        for name, rows in extra_rows().items():
            dump(os.path.join(data, f"bench/{name}.jsonl"), rows)
            dump(os.path.join(data, f"bench/{name}_s200.jsonl"), random.Random(0).sample(rows, 200))
        return
    if train:
        rows = random.Random(0).sample(medqa_rows("train"), 768 + 500)
        dump(os.path.join(data, "practice/medqa_dir.jsonl"), rows[:768])
        dump(os.path.join(data, "practice/medqa_sel.jsonl"), rows[768:])
        return
    from datasets import load_dataset
    medqa = medqa_rows("test")
    test = json.load(urllib.request.urlopen(TEST_IDS, timeout=60))
    letter = {"yes": "A", "no": "B", "maybe": "C"}
    pubmed = [{"id": f"pubmedqa_{r['pubid']}", "kind": "mc", "answer": letter[r["final_decision"]],
               "prompt": mc_prompt("biomedical research", r["question"], [("A", "yes"), ("B", "no"), ("C", "maybe")],
                                   " ".join(r["context"]["contexts"]))}
              for r in load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train") if str(r["pubid"]) in test]
    assert len(medqa) == 1273 and len(pubmed) == 500, (len(medqa), len(pubmed))
    assert all(test[r["id"].split("_", 1)[1]] == {"A": "yes", "B": "no", "C": "maybe"}[r["answer"]] for r in pubmed)
    for name, rows in (("medqa", medqa), ("pubmedqa", pubmed)):
        dump(os.path.join(data, f"bench/{name}.jsonl"), rows)
        dump(os.path.join(data, f"bench/{name}_s200.jsonl"), random.Random(0).sample(rows, 200))


if __name__ == "__main__":
    main(sys.argv[1], "--train" in sys.argv[2:], "--extra" in sys.argv[2:], "--math" in sys.argv[2:])
