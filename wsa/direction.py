"""The textbook loop direction (control (c)): a difference of mean activations between looping and productive
stretches of the base model's own answers, without any agent.

From base-model answers on practice problems (practice/dir.jsonl, the authors' protocol) that ran to the token cap:
the loop is located as the first 512-token step at which wsa.grade.looping fires on the text so far (t0). For each
such answer, "looping" tokens are [t0, t0 + 1024) and "productive" tokens [t0 - 3072, t0 - 2048) of the same answer
(the gap leaves room for a loop that formed before it was detected), so the two sides share the problem and are
close in position. Per layer L in --layers (default 15, 19):

  u_L      = mean residual output of layer L over looping tokens - the same over productive tokens
  score_Lj = (mean looping - mean productive input to down_proj, neuron j) * (u_L . W0[:, j] / |u_L|):
             how much more neuron j writes along u_L while looping; the top 64 are the edited columns

    python -m wsa.direction --rows practice/dir.jsonl --gen runs/base/think35_dir.jsonl --ckpt ckpt/base \
        --out runs/meandiff/dirs.npz [--max 64]
"""
import argparse
import json
import os

import numpy as np

from wsa.grade import TAIL_CHARS, looping, user_prompt

LAYERS = (15, 19)
STEP = 512


def onset(text, offs, n):
    for t in range(STEP, n, STEP):
        end = offs[t - 1][1]
        if looping(text[max(0, end - TAIL_CHARS):end]):
            return t
    return None


def main(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration
    if a.device == "cuda" and a.gpu_share:
        torch.cuda.set_per_process_memory_fraction(a.gpu_share)       # a shared card: stay inside the given share
    LAYERS = tuple(a.layers)
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    rows = {r["id"]: r for r in (json.loads(l) for l in open(a.rows))}
    gens = [json.loads(l) for l in open(a.gen)]
    picked = []
    for g in gens:
        if g["finish"] != "length":
            continue
        enc = tok(g["text"], add_special_tokens=False, return_offsets_mapping=True)
        n = len(enc["input_ids"])
        t0 = onset(g["text"], enc["offset_mapping"], n)
        if t0 is None or t0 < 3072 or t0 + 1024 > n:
            continue
        picked.append((g, enc["input_ids"], t0))
        if len(picked) >= a.max:
            break
    print(f"{len(picked)} looping answers used (of {sum(g['finish'] == 'length' for g in gens)} cut ones)")
    dt = torch.bfloat16 if a.device == "cuda" else torch.float32      # cpu: only the smoke test
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    if "text_config" in cfg:                                             # Qwen3.5: the text model inside a VL model
        model = Qwen3_5ForConditionalGeneration.from_pretrained(a.ckpt, dtype=dt).to(a.device).eval()
        layers, tc = model.model.language_model.layers, cfg["text_config"]
    else:                                                                # Qwen2 (DeepSeek-R1-Distill-Qwen-1.5B)
        model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=dt).to(a.device).eval()
        layers, tc = model.model.layers, cfg
    H, F = tc["hidden_size"], tc["intermediate_size"]
    acc = {L: {"h": [np.zeros(H), np.zeros(H)], "a": [np.zeros(F), np.zeros(F)]} for L in LAYERS}
    counts = [0, 0]
    win = {}

    def take(x, L, key):
        x = x[0].float()
        for side, (s, e) in enumerate(win["spans"]):
            acc[L][key][side] += x[s:e].sum(0).cpu().numpy().astype(np.float64)

    hooks = []
    for L in LAYERS:
        hooks.append(layers[L].register_forward_hook(
            lambda m, i, o, L=L: take(o[0] if isinstance(o, tuple) else o, L, "h")))
        hooks.append(layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, i, L=L: take(i[0], L, "a")))
    with torch.no_grad():
        for g, ids, t0 in picked:
            prompt = tok.apply_chat_template([{"role": "user", "content": user_prompt(rows[g["id"]])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=True)
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            seq = p + ids[:t0 + 1024]
            off = len(p)
            win["spans"] = [(off + t0 - 3072, off + t0 - 2048), (off + t0, off + t0 + 1024)]   # productive, looping
            model(input_ids=torch.tensor([seq], device=a.device), use_cache=False)
            counts[0] += 1024
            counts[1] += 1024
    for h in hooks:
        h.remove()
    out = {"n_answers": len(picked)}
    for L in LAYERS:
        hp, hl = (acc[L]["h"][s] / counts[s] for s in (0, 1))
        ap_, al = (acc[L]["a"][s] / counts[s] for s in (0, 1))
        u = hl - hp
        w = layers[L].mlp.down_proj.weight.detach().float().cpu().numpy()
        proj = (u / np.linalg.norm(u)) @ w
        out[f"u{L}"] = u
        out[f"score{L}"] = (al - ap_) * proj
        print(f"layer {L}: |u| {np.linalg.norm(u):.3f}, top-64 score share "
              f"{np.sort(out[f'score{L}'])[-64:].sum() / np.abs(out[f'score{L}']).sum():.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)       # 849 died here: runs/med_md/ was new
    np.savez(a.out, **out)
    print("->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    ap.add_argument("--gpu-share", type=float, default=0, help="a shared card: this process's memory fraction")
    main(ap.parse_args())
