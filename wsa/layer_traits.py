"""Part 1b, step 1: what makes a layer good to edit? Per layer, from teacher-forced reads of the base model's own
answers on practice problems (never the test sets), the pre-registered traits:

  type, depth          full or linear attention (config layer_types), index
  separation           can the residual after layer L tell looping from productive tokens? a ridge-classifier probe
                       on standardized features, cross-validated by answer (4 folds); and d', the gap between the two
                       sides along u_L divided by the pooled spread of the projections
  concentration        share of the down_proj writing score along u_L held by its top 64 neurons (wsa.direction's score)
  copying heads        per head of a full-attention layer, the attention a query gives to the token right after the
                       previous occurrence of its own token (an induction pattern), averaged over queries that have one;
                       computed on the answer's last 2,048 tokens alone (eager attention), separately for the looping
                       half and the productive half; linear-attention layers have no attention matrix
  edit effect          the layer edited alone as the authors did (W = W0 - alpha u (mask_top64 * u^T W0), alpha 6),
                       on held-out answers: change in mean log-prob of the looping window and of the productive window
                       (the agent's own measure); the same for chosen pairs of layers

Answers and windows as wsa.direction: answers cut at the cap whose loop starts at t0 >= 3072; productive tokens
[t0 - 3072, t0 - 2048), looping tokens [t0, t0 + 1024). The first --n-dir such answers give u, the scores, the probe
and the copying heads; the next --n-eval (other problems) measure the edits.

    python -m wsa.layer_traits --rows data/practice/dir.jsonl --gen runs/base/think35_dir.jsonl --ckpt ckpt/base \
        --out runs/layers/traits_08b [--n-dir 64 --n-eval 32]
writes <out>.json (the table) and <out>.npz (u<L>, score<L> for every layer, in wsa.edits' format)
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

from wsa.direction import onset
from wsa.grade import user_prompt

PROBE_TOKENS = 24          # tokens sampled per side per answer for the probe and d'
ATTN_WINDOW = 2048


def pick(tok, gens, n):
    out = []
    for g in gens:
        if g["finish"] != "length":
            continue
        enc = tok(g["text"], add_special_tokens=False, return_offsets_mapping=True)
        t0 = onset(g["text"], enc["offset_mapping"], len(enc["input_ids"]))
        if t0 is None or t0 < 3072 or t0 + 1024 > len(enc["input_ids"]):
            continue
        out.append((g, enc["input_ids"], t0))
        if len(out) >= n:
            break
    return out


def ridge_probe(X, y, groups, folds=4, lam=1.0):
    """cross-validated accuracy of a ridge classifier on standardized features, folds split by answer"""
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    acc = []
    for f in range(folds):
        test = np.isin(groups, ug[f::folds])
        mu, sd = X[~test].mean(0), X[~test].std(0) + 1e-6
        A, B = (X[~test] - mu) / sd, (X[test] - mu) / sd
        t = np.where(y[~test] == 1, 1.0, -1.0)
        w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (t - t.mean()))
        acc.append(np.mean(((B @ w + t.mean()) > 0) == (y[test] == 1)))
    return float(np.mean(acc))


def main(a):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    types, nL = tc["layer_types"], tc["num_hidden_layers"]
    full = [i for i, x in enumerate(types) if x == "full_attention"]
    rows = {r["id"]: r for r in (json.loads(l) for l in open(a.rows))}
    picked = pick(tok, [json.loads(l) for l in open(a.gen)], a.n_dir + a.n_eval)
    n_dir = a.n_dir if len(picked) >= a.n_dir + a.n_eval else 2 * len(picked) // 3   # a model that loops less (2B)
    dir_set, eval_set = picked[:n_dir], picked[n_dir:]
    print(f"{len(dir_set)} answers for the traits, {len(eval_set)} held out for the edits", flush=True)
    dt = torch.bfloat16 if a.device == "cuda" else torch.float32
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.ckpt, dtype=dt).to(a.device).eval()
    lm = model.model.language_model
    layers = lm.layers
    d, f = tc["hidden_size"], tc["intermediate_size"]

    def seq_of(g, ids, t0):
        p = tok(tok.apply_chat_template([{"role": "user", "content": user_prompt(rows[g["id"]])}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=True), add_special_tokens=False)["input_ids"]
        return p + ids[:t0 + 1024], len(p)

    # ---------------------------------------------------------------- pass A: means, probe samples
    hs = np.zeros((nL, 2, d))
    acts = np.zeros((nL, 2, f))
    rng = np.random.default_rng(0)
    samples = np.zeros((len(dir_set), 2 * PROBE_TOKENS, nL, d), np.float16)
    cur = {}

    def keep_h(L):
        def hook(m, i, o):
            x = (o[0] if isinstance(o, tuple) else o)[0].float()
            for side, (s, e) in enumerate(cur["spans"]):
                hs[L, side] += x[s:e].sum(0).cpu().numpy()
            samples[cur["j"], :, L] = x[cur["pos"]].cpu().numpy().astype(np.float16)
        return hook

    def keep_a(L):
        def hook(m, i):
            x = i[0][0].float()
            for side, (s, e) in enumerate(cur["spans"]):
                acts[L, side] += x[s:e].sum(0).cpu().numpy()
        return hook
    hooks = [layers[L].register_forward_hook(keep_h(L)) for L in range(nL)]
    hooks += [layers[L].mlp.down_proj.register_forward_pre_hook(keep_a(L)) for L in range(nL)]
    with torch.no_grad():
        for j, (g, ids, t0) in enumerate(dir_set):
            seq, off = seq_of(g, ids, t0)
            spans = [(off + t0 - 3072, off + t0 - 2048), (off + t0, off + t0 + 1024)]
            pos = np.concatenate([rng.choice(np.arange(s, e), PROBE_TOKENS, replace=False) for s, e in spans])
            cur.update(spans=spans, pos=torch.tensor(pos, device=a.device), j=j)
            lm(input_ids=torch.tensor([seq], device=a.device), use_cache=False)
    for h in hooks:
        h.remove()
    n_tok = 1024 * len(dir_set)
    hs /= n_tok
    acts /= n_tok
    y = np.tile(np.r_[np.zeros(PROBE_TOKENS), np.ones(PROBE_TOKENS)], len(dir_set))
    groups = np.repeat(np.arange(len(dir_set)), 2 * PROBE_TOKENS)
    traits, dirs = [], {}
    for L in range(nL):
        u = hs[L, 1] - hs[L, 0]
        un = u / np.linalg.norm(u)
        w = layers[L].mlp.down_proj.weight.detach().float().cpu().numpy()
        score = (acts[L, 1] - acts[L, 0]) * (un @ w)
        X = samples[:, :, L].reshape(-1, d).astype(np.float32)
        proj = X @ un
        p0, p1 = proj[y == 0], proj[y == 1]
        dprime = (p1.mean() - p0.mean()) / np.sqrt(0.5 * (p0.var() + p1.var()) + 1e-12)
        traits.append({"layer": L, "type": "full" if L in full else "linear", "depth": round(L / (nL - 1), 3),
                       "u_norm": float(np.linalg.norm(u)), "d_prime": float(dprime), "probe_acc": ridge_probe(X, y, groups),
                       "top64_share": float(np.sort(score)[-64:].sum() / np.abs(score).sum())})
        dirs[f"u{L}"], dirs[f"score{L}"] = u, score
        print(f"layer {L:2d} {traits[-1]['type']:6s} |u| {traits[-1]['u_norm']:.2f} d' {dprime:5.2f} probe "
              f"{traits[-1]['probe_acc']:.3f} top64 {traits[-1]['top64_share']:.3f}", flush=True)

    # ---------------------------------------------------------------- pass B: copying heads (full attention only)
    model.set_attn_implementation("eager")
    ind = {L: np.zeros((2, tc["num_attention_heads"])) for L in full}
    cnt = np.zeros(2)
    with torch.no_grad():
        for g, ids, t0 in dir_set[:a.n_attn]:
            win = ids[t0 + 1024 - ATTN_WINDOW:t0 + 1024]
            o = lm(input_ids=torch.tensor([win], device=a.device), output_attentions=True, use_cache=False)
            last, qs, tg = {}, [], []
            for q, x in enumerate(win):
                if x in last and last[x] + 1 < q:
                    qs.append(q)
                    tg.append(last[x] + 1)
                last[x] = q
            qs, tg = np.array(qs), np.array(tg)
            half = (qs >= ATTN_WINDOW - 1024).astype(int)       # 1 = looping half
            for k, L in enumerate(full):
                att = o.attentions[k][0].float().cpu().numpy()      # [heads, T, T]
                vals = att[:, qs, tg]                                 # [heads, n queries]
                for side in (0, 1):
                    ind[L][side] += vals[:, half == side].sum(1)
            cnt += np.bincount(half, minlength=2)
    model.set_attn_implementation("sdpa")
    for t in traits:
        if t["layer"] in ind:
            v = ind[t["layer"]] / cnt[:, None]
            t["copy_head"] = int(np.argmax(v[1]))
            t["copy_score_loop"] = float(v[1].max())
            t["copy_score_normal"] = float(v[0, int(np.argmax(v[1]))])
            t["copy_scores_loop"] = [round(float(x), 4) for x in v[1]]

    # ---------------------------------------------------------------- pass C: edit effects on held-out answers
    def logprob_windows():
        out = []
        with torch.no_grad():
            for g, ids, t0 in eval_set:
                seq, off = seq_of(g, ids, t0)
                x = torch.tensor([seq], device=a.device)
                h = lm(input_ids=x, use_cache=False).last_hidden_state[0]
                res = []
                for s, e in [(off + t0 - 3072, off + t0 - 2048), (off + t0, off + t0 + 1024)]:
                    z = torch.log_softmax(model.lm_head(h[s - 1:e - 1]).float(), -1)
                    res.append(float(z.gather(1, x[0, s:e, None])[:, 0].mean()))
                out.append(res)
        return np.array(out)                                     # [answers, (productive, looping)]

    W0 = {L: layers[L].mlp.down_proj.weight.detach().clone() for L in range(nL)}

    def apply(Ls):
        for L in Ls:
            u = dirs[f"u{L}"] / np.linalg.norm(dirs[f"u{L}"])
            cols = np.argsort(-dirs[f"score{L}"])[:a.top]
            w = W0[L].float().cpu().numpy().astype(np.float64)
            w[:, cols] -= a.alpha * np.outer(u, u @ w[:, cols])
            layers[L].mlp.down_proj.weight.data.copy_(torch.tensor(w, dtype=W0[L].dtype))

    def reset(Ls):
        for L in Ls:
            layers[L].mlp.down_proj.weight.data.copy_(W0[L])

    base = logprob_windows()
    print(f"base log-prob: productive {base[:, 0].mean():.3f}, looping {base[:, 1].mean():.3f}", flush=True)
    configs = [(L,) for L in range(nL)]
    configs += [p for p in itertools.combinations(full, 2)]
    configs += [p for p in a.extra_pairs if p not in configs]
    if a.max_configs:
        configs = configs[:a.max_configs // 2] + configs[-(a.max_configs - a.max_configs // 2):]   # smoke tests only
    effects = []
    for Ls in configs:
        apply(Ls)
        lp = logprob_windows()
        reset(Ls)
        dlt = lp - base
        se = dlt.std(0, ddof=1) / np.sqrt(len(dlt))
        effects.append({"layers": list(Ls), "d_logprob_productive": float(dlt[:, 0].mean()), "se_productive": float(se[0]),
                        "d_logprob_looping": float(dlt[:, 1].mean()), "se_looping": float(se[1])})
        print(f"edit {str(Ls):10s} looping {dlt[:, 1].mean():+.3f} (se {se[1]:.3f}), productive {dlt[:, 0].mean():+.3f} "
              f"(se {se[0]:.3f})", flush=True)
        if len(Ls) == 1:
            traits[Ls[0]].update(edit_d_looping=effects[-1]["d_logprob_looping"], edit_d_productive=effects[-1]["d_logprob_productive"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out + ".npz", n_answers=len(dir_set), **dirs)
    json.dump({"ckpt": a.ckpt, "n_dir": len(dir_set), "n_eval": len(eval_set), "alpha": a.alpha, "top": a.top,
               "base_logprob": {"productive": float(base[:, 0].mean()), "looping": float(base[:, 1].mean())},
               "full_attention_layers": full, "layers": traits, "edits": effects, "minutes": round((time.time() - t_start) / 60, 1)},
              open(a.out + ".json", "w"), indent=1)
    print(f"-> {a.out}.json, {a.out}.npz ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="path without extension")
    ap.add_argument("--n-dir", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--n-attn", type=int, default=64, help="answers for the copying heads")
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--top", type=int, default=64)
    ap.add_argument("--extra-pairs", default="14,18 15,16 14,15 18,19 19,20 16,20",
                    type=lambda s: [tuple(int(x) for x in p.split(",")) for p in s.split()])
    ap.add_argument("--max-configs", type=int, default=0, help="testing: edit only this many configurations")
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
