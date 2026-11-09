"""Budget forcing and loop breaking from an existing thinking run, without regenerating the thinking.

A run that is stopped at thinking token B (or at the first point a loop detector fires) produces exactly the first B
tokens of the unstopped run, so we cut the saved answer there, close the thinking ("\\n</think>\\n\\n", as the chat
template writes it) and let the model write only the answer (at most ANSWER_TOKENS new tokens). Answers whose
thinking already closed before the cut are kept as they are.

One difference from forcing inside a single request: vLLM's presence penalty counts only the tokens generated in the
current request, so the forced answer is not penalised for words used during the thinking.

    python -m wsa.force plan  --rows R --gen SRC.jsonl --samples n --ckpt C --mode budget4096|loop --out ROWS.jsonl
    (generate ROWS.jsonl with wsa.gen into CONT.jsonl, same protocol)
    python -m wsa.force merge --gen SRC.jsonl --plan ROWS.jsonl --cont CONT.jsonl --out ARM.jsonl
"""
import argparse
import json

from wsa.grade import TAIL_CHARS, looping

ANSWER_TOKENS = 4096
LOOP_STEP = 512          # the loop detector looks every 512 thinking tokens
CLOSE = "\n</think>\n\n"


def load(path):
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[(r["id"], r["k"])] = r
    return out


def cut_point(text, enc, mode):
    """token index at which to close the thinking, or None to keep the answer as it is."""
    offs = enc["offset_mapping"]
    n = len(offs)
    close_at = text.find("</think>")
    close_tok = None
    if close_at >= 0:
        close_tok = next((i for i, (s, e) in enumerate(offs) if e > close_at), n)
    if mode.startswith("budget"):
        b = int(mode[len("budget"):])
        if (close_tok is not None and close_tok <= b) or n <= b:
            return None
        return b
    if mode == "loop":
        limit = close_tok if close_tok is not None else n
        for t in range(LOOP_STEP, limit, LOOP_STEP):
            end = offs[t - 1][1]
            if looping(text[max(0, end - TAIL_CHARS):end]):
                return t
        return None
    raise ValueError(mode)


def plan(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    rows = {r["id"]: r for r in (json.loads(l) for l in open(a.rows))}
    src = load(a.gen)
    n_forced = 0
    with open(a.out, "w") as f:
        for (pid, k), g in sorted(src.items()):
            enc = tok(g["text"], add_special_tokens=False, return_offsets_mapping=True)
            t = cut_point(g["text"], enc, a.mode)
            if t is None:
                continue
            prefix = g["text"][:enc["offset_mapping"][t - 1][1]]
            r = rows[pid]
            keep = {x: r[x] for x in ("problem", "answer", "kind", "prompt") if x in r}     # math or multiple choice
            f.write(json.dumps({"id": pid, "k": k, **keep,
                                "prefix": prefix + CLOSE, "prefix_tok": t, "max_tokens": ANSWER_TOKENS},
                               ensure_ascii=False) + "\n")
            n_forced += 1
    print(f"{a.mode}: {n_forced} of {len(src)} answers forced -> {a.out}")


def merge(a):
    src, cont = load(a.gen), load(a.cont)
    planned = load(a.plan)
    n = 0
    with open(a.out, "w") as f:
        for key, g in sorted(src.items()):
            if key in planned:
                c = cont[key]
                p = planned[key]
                rec = {"id": g["id"], "k": g["k"], "text": p["prefix"] + c["text"], "finish": c["finish"],
                       "n_tok": p["prefix_tok"] + c["n_tok"], "protocol": g["protocol"], "forced": True}
                n += 1
            else:
                rec = dict(g, forced=False)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"merged {n} forced answers into {len(src)} -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "merge"])
    ap.add_argument("--rows")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--mode")
    ap.add_argument("--plan")
    ap.add_argument("--cont")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int)
    a = ap.parse_args()
    plan(a) if a.cmd == "plan" else merge(a)
