"""What the authors' 64 edited neurons write along u while the base model loops, and while it does not (a follow-up of the SVD analysis, wsa.svd_edit). The edit turns their write along u, s = bᵀa (b = W_Jᵀu, a = the 64
neurons' down_proj inputs), into -5 s: if s > 0 while looping and the residual's own u component h·u is smaller than
6 s, the edit pushes the residual from +u to -u at that layer.

Windows, from the base model's own answers on practice/dir.jsonl (the authors' protocol), as in wsa.direction:
  looping      [t0, t0 + 1024) of answers cut at the token cap, t0 the first 512-token step at which the authors'
               loop rule fires
  productive   [t0 - 3072, t0 - 2048) of the same answers
  closing      the last 1024 tokens before </think> of answers that finished (the model wrapping up)
Per token and layer (15, 19): s, h·u (the layer's output residual along u) and s / |h|. u and the 64 columns are the
authors' (recovered from their edit; the sign of u agrees with our mean-difference loop direction).

    python -m wsa.loop_write --rows data/practice/dir.jsonl --gen runs/base/think35_dir.jsonl --ckpt ckpt/base \
        --dirs runs/meandiff/dirs.npz --out runs/loopwrite/loop_write [--max 64] [--gpu-share 0.40]
"""
import argparse
import json
import os

import numpy as np

from wsa.direction import LAYERS, onset
from wsa.edits import NAME, authors_delta
from wsa.grade import user_prompt

WIN = 1024


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)), "share_positive": float((x > 0).mean()),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}


def main(a):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    if a.device == "cuda" and a.gpu_share:
        torch.cuda.set_per_process_memory_fraction(a.gpu_share)       # a shared card: stay inside the given share
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    rows = {r["id"]: r for r in (json.loads(l) for l in open(a.rows))}
    gens = [json.loads(l) for l in open(a.gen)]
    end_think = tok.convert_tokens_to_ids("</think>")
    loops, closes = [], []
    for g in gens:
        enc = tok(g["text"], add_special_tokens=False, return_offsets_mapping=True)
        ids, n = enc["input_ids"], len(enc["input_ids"])
        if g["finish"] == "length" and len(loops) < a.max:
            t0 = onset(g["text"], enc["offset_mapping"], n)
            if t0 is not None and t0 >= 3072 and t0 + WIN <= n and (not a.max_len or t0 + WIN <= a.max_len):
                loops.append((g, ids[:t0 + WIN], {"productive": (t0 - 3072, t0 - 2048), "looping": (t0, t0 + WIN)}))
        elif g["finish"] != "length" and len(closes) < a.max and end_think in ids:
            e = ids.index(end_think)
            if e >= WIN and (not a.max_len or e <= a.max_len):
                closes.append((g, ids[:e], {"closing": (e - WIN, e)}))
    print(f"{len(loops)} looping answers, {len(closes)} finished answers")
    ad = authors_delta(a.ckpt)
    md = np.load(a.dirs)
    dt = torch.bfloat16 if a.device == "cuda" else torch.float32      # cpu: the smoke test only
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.ckpt, dtype=dt).to(a.device).eval()
    layers = model.model.language_model.layers
    per = {}
    for L in LAYERS:
        d = ad[NAME.format(L)]
        u = d["u"] if d["u"] @ md[f"u{L}"] >= 0 else -d["u"]
        w = layers[L].mlp.down_proj.weight.detach().double().cpu().numpy()
        per[L] = {"cols": torch.tensor(d["cols"], device=a.device),
                  "b": torch.tensor(w[:, d["cols"]].T @ u, device=a.device, dtype=torch.float32),
                  "u": torch.tensor(u, device=a.device, dtype=torch.float32)}
    got = {L: {"s": None, "hu": None, "hn": None} for L in LAYERS}
    hooks = []
    for L in LAYERS:
        hooks.append(layers[L].mlp.down_proj.register_forward_pre_hook(
            lambda m, i, L=L: got[L].__setitem__("s", (i[0][0].float()[:, per[L]["cols"]] @ per[L]["b"]).cpu().numpy())))

        def out_hook(m, i, o, L=L):
            h = (o[0] if isinstance(o, tuple) else o)[0].float()
            got[L]["hu"] = (h @ per[L]["u"]).cpu().numpy()
            got[L]["hn"] = h.norm(dim=-1).cpu().numpy()
        hooks.append(layers[L].register_forward_hook(out_hook))
    acc = {L: {} for L in LAYERS}
    with torch.no_grad():
        for g, ids, wins in loops + closes:
            prompt = tok.apply_chat_template([{"role": "user", "content": user_prompt(rows[g["id"]])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=True)
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            model(input_ids=torch.tensor([p + ids], device=a.device), use_cache=False)
            for L in LAYERS:
                for side, (s0, s1) in wins.items():
                    sl = slice(len(p) + s0, len(p) + s1)
                    for key, v in (("s", got[L]["s"][sl]), ("hu", got[L]["hu"][sl]), ("s_over_h", got[L]["s"][sl] / got[L]["hn"][sl])):
                        acc[L].setdefault(side, {}).setdefault(key, []).append(v)
    for h in hooks:
        h.remove()
    out = {"n_looping_answers": len(loops), "n_finished_answers": len(closes), "window": WIN, "layers": {}}
    arrays = {}
    for L in LAYERS:
        out["layers"][str(L)] = {"b_norm": float(per[L]["b"].norm())}
        for side, d in acc[L].items():
            out["layers"][str(L)][side] = {k: summary(np.concatenate(v)) for k, v in d.items()}
            s, hu = np.concatenate(d["s"]), np.concatenate(d["hu"])
            # after the edit the layer's residual along u is hu - 6 s (the 64 neurons' u write becomes -5 s)
            out["layers"][str(L)][side]["after_edit_hu"] = summary(hu - 6 * s)
            out["layers"][str(L)][side]["share_flipped_sign"] = float(np.mean(np.sign(hu - 6 * s) != np.sign(hu)))
            for k, v in d.items():
                arrays[f"L{L}_{side}_{k}"] = np.concatenate(v).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out + ".npz", **arrays)
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dirs", required=True, help="the mean-difference directions (for the sign of u)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=0, help="skip answers longer than this (the CPU smoke test only)")
    ap.add_argument("--gpu-share", type=float, default=0.0, help="cap this process at this share of the card")
    main(ap.parse_args())
