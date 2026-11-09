"""Diagnostic: does the edited model generate <|endoftext|> (token 248044)? The authors' text-only checkpoint lists it
as eos in generation_config.json, so their vLLM stopped there; the public release has no generation_config.json, so
our server (and the independent re-run) did not stop and the text went on. From one run with special tokens kept,
score each answer both ways: as we ran (special tokens dropped) and as the authors' engine would have stopped (text cut
at the first <|endoftext|>).

    python -m wsa.eot --rows data/bench/math500.jsonl --gen runs/diag/think35_raw_math500.jsonl
"""
import argparse
import json

from wsa.grade import extract, is_correct

EOT = "<|endoftext|>"
SPECIAL = ("<|endoftext|>", "<|im_end|>")


def main(a):
    gold = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(a.rows)}
    n = with_eot = ours_ok = auth_ok = changed = 0
    where = {"thinking": 0, "after </think>": 0}
    for line in open(a.gen):
        r = json.loads(line)
        t = r["text"]
        full = t
        for s in SPECIAL:
            full = full.replace(s, "")
        cut = t.split(EOT)[0].replace("<|im_end|>", "")
        o = is_correct(extract(full)[0], gold[r["id"]])
        u = is_correct(extract(cut)[0], gold[r["id"]])
        n += 1
        ours_ok += o
        auth_ok += u
        changed += o != u
        if EOT in t:
            with_eot += 1
            where["after </think>" if "</think>" in t.split(EOT)[0] else "thinking"] += 1
    print(json.dumps({"answers": n, "with_endoftext": with_eot, "where_first_endoftext": where,
                      "acc_as_we_ran": ours_ok / n, "acc_if_stopped_at_endoftext": auth_ok / n,
                      "answers_whose_grade_changes": changed}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--gen", required=True)
    main(ap.parse_args())
