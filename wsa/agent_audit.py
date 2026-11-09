"""Leakage audit of the two frontier-agent runs (CPU, the runs' own logs).

1. The task prompt: the authors' message (agent/assets/message.txt), verbatim; which sentences name repetition.
2. Problems: every practice id the agents generated on is in the practice pool given to them (no test id), and how
   many of them are training-side near-duplicates of a test problem (wsa.contam's rule from the training side, the
   343 problems wsa.sft_data drops).
3. Text the agents wrote or sent: teacher-forced texts and chats (inline or from the files in their /work), every
   Python program, and every reply. A text leaks a test problem when it shares a 13-word window (normalised as in
   wsa.contam) with a MATH-500, GSM8K or MedQA test problem; a window that the practice problem of the same request
   also contains is a practice near-duplicate, not a leak.

    python -m wsa.agent_audit        (writes results/agent_audit.json)
"""
import glob
import json
import re
from pathlib import Path

from wsa.contam import grams, norm

RUNS = Path("agent_runs")
AGENTS = {"claude_1": "Claude Opus 5.5", "openai_1": "gpt-6.1-sol"}
TESTS = {"MATH-500": "results/data/math500.jsonl", "GSM8K": "results/data/gsm8k.jsonl", "MedQA": "results/data/medqa.jsonl"}


def text_of(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(text_of(y) for y in x)
    if isinstance(x, dict):
        return "\n".join(text_of(v) for k, v in x.items() if k in ("content", "text"))
    return ""


def main():
    msg = open("agent/assets/message.txt").read()
    rep = [s.strip() for s in re.split(r"(?<=[.;])\s+", msg) if re.search(r"repeat|repetition|loop", s, re.I)]
    prac = {}
    for f in glob.glob("data/practice/*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            prac[r["id"]] = r
    pool_neardup = set()
    test_grams = {}
    for name, f in TESTS.items():
        for line in open(f):
            r = json.loads(line)
            q = r.get("problem") or r.get("question") or r.get("prompt") or ""
            for g in grams(norm(q)):
                test_grams.setdefault(g, set()).add((name, r["id"]))
    for pid, r in prac.items():
        if any(g in test_grams for g in grams(norm(r["problem"]))):
            pool_neardup.add(pid)
    import numpy as np                      # and by cosine (>= 0.8) to a MATH-500 or GSM8K problem, as wsa.sft_data
    from sklearn.feature_extraction.text import TfidfVectorizer
    from wsa.contam import NEAR
    pid_list = [i for i in prac if not i.startswith("aime")]
    pn = [norm(prac[i]["problem"]) for i in pid_list]
    tn = [norm(json.loads(l)["problem"]) for f in (TESTS["MATH-500"], TESTS["GSM8K"]) for l in open(f)]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(pn + tn)
    P, T = vec.transform(pn), vec.transform(tn)
    for i in range(0, len(pn), 2000):
        best = (P[i:i + 2000] @ T.T).max(1).toarray().ravel()
        pool_neardup.update(pid_list[i + j] for j in np.flatnonzero(best >= NEAR))
    out = {"task_prompt_sentences_naming_repetition": rep, "agents": {}}
    for a, label in AGENTS.items():
        reqs = [json.load(open(f)) for f in sorted(glob.glob(str(RUNS / a / "requests" / "fwd_*.json")))
                if not f.endswith(".done.json")]
        ids, texts, prompts = set(), [], []
        for r in reqs:
            if "request_file" in r:
                p = RUNS / a / "work" / Path(r["request_file"]).name
                r = json.load(open(p)) if p.exists() else r
            ids.update(r.get("problems", []))
            prompts += r.get("prompts", [])
            if "texts" in r:
                texts += [text_of(t) for t in r["texts"]]
            if "chats" in r:
                texts += [text_of(c) for c in r["chats"]]
            if "texts_file" in r:
                p = RUNS / a / "work" / Path(r["texts_file"]).name
                if p.exists():
                    d = json.load(open(p))
                    d = d.get("texts", d.get("chats", d)) if isinstance(d, dict) else d
                    texts += [text_of(t) for t in d]
        calls = [json.loads(l) for l in open(RUNS / a / "tool_calls.jsonl")]
        code = [c.get("code", "") for c in calls]
        log = open(RUNS / a / "driver.log").read()          # the harness's own count: failed requests are not counted
        used = re.findall(r"passes (\d+)/(\d+)", log)
        replies = [open(f).read() for f in sorted(glob.glob(str(RUNS / a / "turns" / "*" / "answer.md")))]
        prac_grams = set().union(*(grams(norm(prac[i]["problem"])) for i in ids if i in prac)) if ids else set()

        def hits(items):
            leak, near = set(), set()
            for t in items:
                for g in grams(norm(t)):
                    if g in test_grams:
                        (near if g in prac_grams else leak).update(test_grams[g])
            return sorted(leak), sorted(near)
        R = {"pool_neardup_of_a_test_problem": len(pool_neardup), "forward_requests": len(reqs), "practice_ids": len(ids), "ids_not_in_practice_pool": sorted(i for i in ids if i not in prac),
             "practice_ids_neardup_of_a_test_problem": len(ids & pool_neardup), "own_prompts": len(prompts),
             "teacher_forced_texts": len(texts), "python_programs": len(code), "replies": len(replies), "windows": {},
             "tool_calls": {"python": sum(c.get("type") == "python" for c in calls),
                            "forward": sum(c.get("type") == "forward" for c in calls),
                            "forward_failed": len(re.findall(r"forward request \d+ failed", log)),
                            "passes_used": int(used[-1][0]) if used else None, "pass_budget": int(used[-1][1]) if used else None}}
        for name, items in (("own prompts", prompts), ("teacher-forced texts", texts), ("python programs", code),
                            ("replies", replies)):
            leak, near = hits(items)
            R["windows"][name] = {"test_problems_leaked": leak, "via_practice_near_duplicates": near}
        R["mentions_of_test_sets"] = sum(len(re.findall(r"MATH-?500|math500|gsm8k_test|test/[a-z_]+/\d+\.json", t))
                                         for t in code + replies)
        out["agents"][label] = R
        print(label, {k: v for k, v in R.items() if k != "windows"})
        for k, v in R["windows"].items():
            print(f"   {k:22s} leaked {len(v['test_problems_leaked'])}  via practice near-duplicates {len(v['via_practice_near_duplicates'])}")
    print("task prompt, sentences naming repetition:", *rep, sep="\n   ")
    Path("results/agent_audit.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
