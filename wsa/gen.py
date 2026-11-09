"""Generate answers through a running vLLM OpenAI-compatible server and append every finished sample to a jsonl file
at once, so a stopped run resumes where it stopped (a sample already in the file is never asked again).

Each sample has its own seed from (problem id, sample index), the same in every arm, so paired arms share random
numbers as far as their logits allow.

    python -m wsa.gen --url http://127.0.0.1:8000 --ckpt <dir for the tokenizer> --rows bench/math500.jsonl \
        --samples 4 --protocol think35 --out runs/<arm>/math500.jsonl [--concurrency 384]

Rows may carry "prefix" (text appended after the chat template's generation prompt, e.g. a forced end of thinking),
"max_tokens" (overrides the protocol's cap) and "k" (only that sample); budget forcing and loop breaking use these.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time

PROMPT_STOP = "If you notice that you are repeating the same reasoning, stop and give your final answer."
PROMPT_STOP_FIRM = ("Important: do not repeat yourself. As soon as you notice that you are going over the same steps or "
                    "phrases again, stop thinking and give your final answer.")
PROTOCOLS = {
    # the authors' evaluation (code/evaluate.py SAMPLING_THINK35)
    "think35": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                "max_tokens": 32768},
    # E3's anti-repetition family after the no-repeat sampler proved too slow: think35 plus vLLM's
    # own multiplicative repetition_penalty
    **{f"think35_rp{t}": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                          "repetition_penalty": v, "max_tokens": 32768} for t, v in (("105", 1.05), ("11", 1.1), ("12", 1.2))},
    # Qwen3.5-0.8B model card, thinking mode
    "card_think": {"thinking": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                   "max_tokens": 32768},
    "card_think_81k": {"thinking": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                       "max_tokens": 81920},
    # diagnostic: the authors' protocol with special tokens kept in the text (is <|endoftext|> ever generated?)
    "think35_raw": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                    "max_tokens": 32768, "skip_special_tokens": False},
    # stronger repetition penalties at the authors' temperature
    "think35_pp2": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 2.0,
                    "max_tokens": 32768},
    "think35_rep11": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                      "repetition_penalty": 1.1, "max_tokens": 32768},
    # thinking off: the authors' own coded protocol (SAMPLING_NOTHINK35) and the model card's non-thinking setting
    "nothink35": {"thinking": False, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
                  "max_tokens": 8192},
    "card_nothink": {"thinking": False, "temperature": 1.0, "top_p": 1.0, "top_k": 20, "presence_penalty": 2.0,
                     "max_tokens": 8192},
    # Gemma 3 270M IT (no thinking mode, 32k context): the authors' Gemma follow-up evaluation (exp8_math/gemma_followup:
    # the model card's sampling, at most 8,192 new tokens)
    "gemma3_card": {"thinking": False, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 8192},
    # each model card's own thinking-mode sampling, for the second screen: DeepSeek-R1-Distill
    # (temperature 0.6, top_p 0.95), Qwen3 (thinking mode: 0.6, 0.95, top_k 20, min_p 0), Gemma 4 (1.0, 0.95, top_k 64;
    # thinking through the chat template's enable_thinking); Qwen3.5-4B and 9B use card_think, their cards' general
    # thinking-mode setting; Gemma 3 270M's card setting is gemma3_card. None of these adds a presence penalty except
    # card_think, as the cards say.
    "r1d_card": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "max_tokens": 32768},
    "qwen3_card": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 32768},
    "gemma4_card": {"thinking": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 32768},
    # E3: cheap decoding baselines, each the authors' think35 with one change; the bias and the window
    # are picked on the practice set (sel200) only, then the picks answer the test sets. </think> is token 248069 of
    # Qwen3.5's vocabulary.
    **{f"think35_tb{b}": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                          "max_tokens": 32768, "logit_bias": {"248069": float(b)}} for b in (2, 4, 8)},
    # no 4-gram repeated within the last w generated tokens (0 = the whole answer; wsa.norepeat, so the server needs
    # --logits-processors wsa.norepeat:NoRepeatNGram)
    **{f"think35_nr4_{n}": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                            "max_tokens": 32768, "vllm_xargs": {"ngram_size": 4, "window_size": w}}
       for n, w in (("w256", 256), ("w1024", 1024), ("all", 0))},
    # thinking off at the thinking runs' cap (nothink35 stops at 8,192, so its budget differs)
    "nothink35_32k": {"thinking": False, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
                      "max_tokens": 32768},
    # E6: the authors' protocol at the model card's cap for hard problems
    "think35_81k": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                    "max_tokens": 81920},
    # low-temperature loop screen: the two lowest temperatures of Pipis et al.'s
    # grid (arXiv 2512.12895 v3, §2 and App. A: 0, 0.2, ..., 1.0, each model's default top_p/top_k, no repetition
    # penalty), with the cards' other settings; their 0.6 is r1d_card / qwen3_card above, already run (hard screen)
    "r1d_t0": {"thinking": True, "temperature": 0.0, "top_p": 0.95, "max_tokens": 32768},
    "r1d_t02": {"thinking": True, "temperature": 0.2, "top_p": 0.95, "max_tokens": 32768},
    "qwen3_t0": {"thinking": True, "temperature": 0.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 32768},
    "qwen3_t02": {"thinking": True, "temperature": 0.2, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 32768},
    # prompt-only arm: think35 plus one instruction, no weight change; the wording is picked on
    # the practice set (sel200) and only the picked one answers the test sets
    "think35_pr1": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                    "max_tokens": 32768, "suffix": PROMPT_STOP},
    "think35_pr2": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                    "max_tokens": 32768, "system": PROMPT_STOP},
    "think35_pr3": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                    "max_tokens": 32768, "suffix": PROMPT_STOP_FIRM},
}


def seed_of(pid, k):
    return int(hashlib.sha256(f"{pid}|{k}".encode()).hexdigest()[:8], 16)


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:      # a line cut by a kill: drop it, the sample is asked again
                    continue
                keys.add((r["id"], r["k"]))
    return keys


async def one(session, url, body, retries=3):
    for a in range(retries):
        try:
            async with session.post(url + "/v1/completions", json=body) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
                return await resp.json()
        except Exception as e:  # noqa: BLE001
            if a == retries - 1:
                raise
            print(f"retry after {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            await asyncio.sleep(10)


async def run(a):
    import aiohttp
    from transformers import AutoTokenizer
    from wsa.grade import user_prompt
    proto = dict(PROTOCOLS[a.protocol])
    thinking = proto.pop("thinking")
    suffix, system = proto.pop("suffix", None), proto.pop("system", None)
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    rows = [json.loads(l) for l in open(a.rows)]
    if a.limit:
        rows = rows[:a.limit]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    have = done_keys(a.out)
    todo = []
    for r in rows:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
            [{"role": "user", "content": user_prompt(r) + ("\n\n" + suffix if suffix else "")}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        for k in ([r["k"]] if "k" in r else range(a.samples)):
            if (r["id"], k) not in have:
                todo.append((r, k, prompt + r.get("prefix", "")))
    print(f"{a.out}: {len(have)} done, {len(todo)} to do ({a.protocol})", flush=True)
    if not todo:
        return 0
    sem = asyncio.Semaphore(a.concurrency)
    out = open(a.out, "a")
    t0, n_done, n_tok, failed = time.time(), 0, 0, 0
    timeout = aiohttp.ClientTimeout(total=None, sock_read=4 * 3600)

    async def task(session, r, k, prompt):
        nonlocal n_done, n_tok, failed
        body = {"model": "m", "prompt": prompt, "seed": seed_of(r["id"], k), "skip_special_tokens": True, **proto}
        if "max_tokens" in r:
            body["max_tokens"] = r["max_tokens"]
        async with sem:
            try:
                res = await one(session, a.url, body)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAILED {r['id']} {k}: {e}", file=sys.stderr, flush=True)
                return
        c = res["choices"][0]
        rec = {"id": r["id"], "k": k, "text": c["text"], "finish": c["finish_reason"],
               "n_tok": res["usage"]["completion_tokens"], "protocol": a.protocol}
        if "prefix" in r:
            rec["prefix_tok"] = r.get("prefix_tok")
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        n_done += 1
        n_tok += rec["n_tok"]
        if n_done % 100 == 0:
            dt = time.time() - t0
            print(f"{n_done}/{len(todo)} done, {n_tok / dt:,.0f} tok/s, {dt / 60:.1f} min", flush=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=a.concurrency)) as s:
        await asyncio.gather(*(task(s, r, k, p) for r, k, p in todo))
    out.close()
    dt = time.time() - t0
    print(f"finished {n_done}/{len(todo)} in {dt / 60:.1f} min, {n_tok / max(dt, 1):,.0f} tok/s, failed {failed}",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0)
    sys.exit(asyncio.run(run(ap.parse_args())))
