"""Score a generation file with the authors' rule (wsa.grade) into the authors' format: per problem {id, correct} and
a summary {acc, cut_share, mean_tokens, ...}. For thinking arms it also scores the same answers cut at 4k / 8k / 16k
tokens (a capped run produces exactly the first C tokens of the uncapped one, so no new generation is needed).

    python -m wsa.score --rows bench/math500.jsonl --gen runs/<arm>/math500.jsonl --samples 4 [--ckpt <dir> --caps]
"""
import argparse
import json
import os

from wsa.grade import grade

CAPS = (4096, 8192, 16384)
CHUNK = 1000


def offsets(tok, text):
    """token character offsets of text, and whether they are exact. The fast tokenizer panics (pyo3 PanicException,
    "Onig: Regex search error: retry-limit-in-match over") on a few degenerate answers, e.g. 10 million characters of
    spaces in two of E6's; those are tokenized in 1,000-character pieces, which can split a token at each cut,
    so their cap positions are off by at most one token per piece"""
    try:
        return tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"], True
    except BaseException as e:  # noqa: BLE001 (PanicException derives from BaseException)
        if type(e).__name__ != "PanicException":
            raise
    offs = []
    for s in range(0, len(text), CHUNK):
        offs += [(a + s, b + s) for a, b in tok(text[s:s + CHUNK], add_special_tokens=False,
                                                return_offsets_mapping=True)["offset_mapping"]]
    return offs, False


def load_gen(path):
    recs = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[(r["id"], r["k"])] = r
    return recs


def main(a):
    rows = [json.loads(l) for l in open(a.rows)]
    if a.limit:
        rows = rows[:a.limit]
    recs = load_gen(a.gen)
    tok = None
    if a.caps:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.ckpt)
    per, missing = [], 0
    tot = ok = cut = noans = ntok = approx = 0
    cap_ok = {c: 0 for c in CAPS}
    for r in rows:
        c, caps_c = [], {cc: [] for cc in CAPS}
        for k in range(a.samples):
            g = recs.get((r["id"], k))
            if g is None:
                missing += 1
                continue
            pred, good = grade(r, g["text"])
            c.append(int(good))
            cut += g["finish"] == "length"
            noans += pred is None and g["finish"] != "length"
            ntok += g["n_tok"]
            if tok is not None:
                offs = None
                for cc in CAPS:
                    if g["n_tok"] <= cc:
                        gc = good
                    else:
                        if offs is None:
                            offs, exact = offsets(tok, g["text"])
                            approx += not exact
                        end = offs[cc - 1][1] if len(offs) >= cc else len(g["text"])
                        gc = grade(r, g["text"][:end])[1]
                    caps_c[cc].append(int(gc))
                    cap_ok[cc] += gc
        per.append({"id": r["id"], "correct": c, **({f"correct_cap{cc}": caps_c[cc] for cc in CAPS} if tok else {})})
        tot += len(c)
        ok += sum(c)
    summ = {"acc": ok / max(tot, 1), "cut_share": cut / max(tot, 1), "mean_tokens": ntok / max(tot, 1),
            "n_problems": len(rows), "samples": a.samples, "total": tot, "ok": ok, "cut": cut,
            "no_answer_finished": noans, "missing": missing}
    if tok is not None:
        summ.update({f"acc_cap{cc}": cap_ok[cc] / max(tot, 1) for cc in CAPS})
        summ["caps_approx_samples"] = approx       # answers whose cap positions came from the piecewise fallback
    base = os.path.splitext(a.gen)[0]
    json.dump(per, open(base + ".per_problem.json", "w"))
    json.dump(summ, open(base + ".scores.json", "w"), indent=1)
    print(os.path.basename(a.gen), json.dumps(summ))
    return 1 if missing else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--caps", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    raise SystemExit(main(ap.parse_args()))
