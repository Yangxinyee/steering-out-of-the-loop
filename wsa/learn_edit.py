"""Part 1b, step 4: learn the authors' "64 neurons" and "strength 6" by gradient descent, the
simple automatic baseline for what the agent chose by hand. Not a new method: with the direction fixed it is a very
small fine-tune (one strength per layer and one gate per neuron).

Per edited layer L, with u_L fixed (wsa.layer_traits) and c = u_L^T W0 (what each neuron writes along u_L):
    W = W0 - alpha_L * u_L (g ⊙ c),   g = sigmoid(theta), one gate per down_proj column
applied as a hook on down_proj (the frozen model is never changed): out += -alpha_L * (a · (g ⊙ c)) u_L.
Gates start like the top-64 edit (the 64 best-scoring columns near 1, the rest near 0), alpha_L at 6.

Objective on the base model's own practice answers (practice/dir.jsonl, never a test set), the agent's own measure:
    looping tokens [t0, t0 + 1024): their log-prob, pushed down until it reaches log(--floor-p) (no gain below)
  + --kl * KL(base || edited) on productive tokens [t0 - 3072, t0 - 2048), so normal reasoning is left alone
  + --lam * (sum of gates / 64), an L1 penalty on the number of open neurons
The first --n-train looping answers train it; the next --n-val (other problems) report held-out log-probs.

    python -m wsa.learn_edit --rows data/practice/dir.jsonl --gen runs/base/think35_dir.jsonl --ckpt ckpt/base \
        --dirs runs/layers/traits_08b.npz --layers 15 19 --lam 0.1 --out runs/layers/learned_lam0.1
writes <out>.npz (u<L>, gate<L>, alpha<L>; wsa.edits learned builds the checkpoint) and <out>.json
"""
import argparse
import json
import math
import os
import time

import numpy as np

from wsa.grade import user_prompt
from wsa.layer_traits import pick


def main(a):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    t_start = time.time()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    rows = {r["id"]: r for r in (json.loads(l) for l in open(a.rows))}
    picked = pick(tok, [json.loads(l) for l in open(a.gen)], a.n_train + a.n_val)
    train, val = picked[:a.n_train], picked[a.n_train:]
    print(f"{len(train)} answers to train on, {len(val)} held out", flush=True)
    dev = a.device
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.ckpt, dtype=dt).to(dev).eval()
    model.requires_grad_(False)
    if a.checkpointing:                               # transformers checkpoints only in train mode (no dropout here)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    lm, layers = model.model.language_model, model.model.language_model.layers
    z = np.load(a.dirs)
    state = {"on": False}
    U, C, theta, alpha, top = {}, {}, {}, {}, {}
    for L in a.layers:
        u = z[f"u{L}"] / np.linalg.norm(z[f"u{L}"])
        U[L] = torch.tensor(u, dtype=torch.float32, device=dev)
        C[L] = (U[L] @ layers[L].mlp.down_proj.weight.float()).detach()          # [f]
        top[L] = np.argsort(-z[f"score{L}"])[:64]
        th = torch.full_like(C[L], a.init_off)
        th[torch.tensor(top[L], device=dev)] = a.init_on
        theta[L] = torch.nn.Parameter(th)
        alpha[L] = torch.nn.Parameter(torch.tensor(a.alpha0, device=dev))

    def hook(L):
        def f(m, i, o):
            if not state["on"]:
                return o
            s = i[0].float() @ (torch.sigmoid(theta[L]) * C[L])                 # [batch, T]
            return o - (alpha[L] * s)[..., None].to(o.dtype) * U[L].to(o.dtype)
        return f
    hooks = [layers[L].mlp.down_proj.register_forward_hook(hook(L)) for L in a.layers]
    opt = torch.optim.Adam(list(theta.values()) + list(alpha.values()), lr=a.lr)
    floor = math.log(a.floor_p)

    def seq_of(g, ids, t0):
        p = tok(tok.apply_chat_template([{"role": "user", "content": user_prompt(rows[g["id"]])}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=True), add_special_tokens=False)["input_ids"]
        seq, off = p + ids[:t0 + 1024], len(p)
        spans = [(off + t0 - 3072, off + t0 - 2048), (off + t0, off + t0 + 1024)]
        if a.max_len and len(seq) > a.max_len:                                  # smoke tests only: drop the start
            cut = len(seq) - a.max_len
            seq, spans = seq[cut:], [(s - cut, e - cut) for s, e in spans]
        return torch.tensor([seq], device=dev), spans

    def terms(x, spans, base_lp):
        h = lm(input_ids=x, use_cache=False).last_hidden_state[0]
        (ps, pe), (ls, le) = spans
        lp_loop = torch.log_softmax(model.lm_head(h[ls - 1:le - 1]).float(), -1).gather(1, x[0, ls:le, None])[:, 0]
        lp_prod = torch.log_softmax(model.lm_head(h[ps - 1:pe - 1]).float(), -1)
        kl = (base_lp.exp() * (base_lp - lp_prod)).sum(-1).mean()
        tok_prod = lp_prod.gather(1, x[0, ps:pe, None])[:, 0]
        return lp_loop, kl, tok_prod

    def base_of(x, spans):
        state["on"] = False
        with torch.no_grad():
            h = lm(input_ids=x, use_cache=False).last_hidden_state[0]
            (ps, pe), (ls, le) = spans
            base_lp = torch.log_softmax(model.lm_head(h[ps - 1:pe - 1]).float(), -1)
            base_loop = torch.log_softmax(model.lm_head(h[ls - 1:le - 1]).float(), -1).gather(1, x[0, ls:le, None])[:, 0]
        return base_lp, base_loop

    def gates_report():
        out = {}
        for L in a.layers:
            g = torch.sigmoid(theta[L]).detach().cpu().numpy()
            on = np.flatnonzero(g > 0.5)
            out[L] = {"alpha": float(alpha[L]), "n_open": int(len(on)), "gate_sum": float(g.sum()),
                      "overlap_with_top64": int(len(set(on.tolist()) & set(top[L].tolist())))}
        return out

    def evaluate():
        res = []
        for g, ids, t0 in val:
            x, spans = seq_of(g, ids, t0)
            base_lp, base_loop = base_of(x, spans)
            state["on"] = True
            with torch.no_grad():
                lp_loop, kl, tok_prod = terms(x, spans, base_lp)
            ps, pe = spans[0]
            base_tok = base_lp.gather(1, x[0, ps:pe, None])[:, 0]
            res.append([float(lp_loop.mean() - base_loop.mean()), float(tok_prod.mean() - base_tok.mean()), float(kl)])
        r = np.array(res)
        return {"d_logprob_looping": float(r[:, 0].mean()), "d_logprob_productive": float(r[:, 1].mean()),
                "kl_productive": float(r[:, 2].mean())}

    log = []
    for ep in range(a.epochs):
        order = np.random.default_rng(ep).permutation(len(train))
        sums = np.zeros(4)
        for k in order[:a.steps_per_epoch or None]:
            g, ids, t0 = train[k]
            x, spans = seq_of(g, ids, t0)
            base_lp, _ = base_of(x, spans)
            state["on"] = True
            lp_loop, kl, _ = terms(x, spans, base_lp)
            loss_loop = lp_loop.clamp(min=floor).mean()
            sparsity = sum(torch.sigmoid(theta[L]).sum() for L in a.layers) / (64 * len(a.layers))
            loss = loss_loop + a.kl * kl + a.lam * sparsity
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums += [loss.item(), loss_loop.item(), kl.item(), sparsity.item()]
        n = len(order[:a.steps_per_epoch or None])
        ev = evaluate()
        rec = {"epoch": ep + 1, "loss": sums[0] / n, "loop_term": sums[1] / n, "kl": sums[2] / n, "sparsity": sums[3] / n,
               "val": ev, "gates": gates_report(), "minutes": round((time.time() - t_start) / 60, 1)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
    for h in hooks:
        h.remove()
    from wsa.edits import NAME, authors_delta
    try:
        ad = authors_delta(a.ckpt)                    # the authors' 64 columns at layers 15 and 19 (0.8B only)
    except Exception:  # noqa: BLE001
        ad = {}
    final = gates_report()
    for L in a.layers:
        n = NAME.format(L)
        if n in ad:
            on = np.flatnonzero(torch.sigmoid(theta[L]).detach().cpu().numpy() > 0.5)
            final[L]["overlap_with_authors64"] = int(len(set(on.tolist()) & set(ad[n]["cols"].tolist())))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out + ".npz", **{f"u{L}": U[L].cpu().numpy() for L in a.layers},
             **{f"gate{L}": torch.sigmoid(theta[L]).detach().cpu().numpy() for L in a.layers},
             **{f"alpha{L}": np.array(float(alpha[L])) for L in a.layers})
    json.dump({"layers": a.layers, "lam": a.lam, "kl": a.kl, "floor_p": a.floor_p, "n_train": len(train), "n_val": len(val),
               "epochs": a.epochs, "lr": a.lr, "final": final, "log": log,
               "minutes": round((time.time() - t_start) / 60, 1)}, open(a.out + ".json", "w"), indent=1)
    print(f"-> {a.out}.npz / .json: {json.dumps(final)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dirs", required=True, help="wsa.layer_traits npz (u<L>, score<L>)")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True, help="path without extension")
    ap.add_argument("--lam", type=float, default=0.1, help="L1 weight on (sum of gates / 64)")
    ap.add_argument("--kl", type=float, default=1.0, help="weight of KL(base || edited) on productive tokens")
    ap.add_argument("--floor-p", type=float, default=0.05, help="stop pushing a looping token below this probability")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="testing: fewer steps")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--alpha0", type=float, default=6.0)
    ap.add_argument("--init-on", type=float, default=4.0, help="gate logit of the top-64 columns at the start")
    ap.add_argument("--init-off", type=float, default=-6.0, help="gate logit of the other columns at the start")
    ap.add_argument("--checkpointing", action="store_true", help="gradient checkpointing (long answers on the GPU)")
    ap.add_argument("--max-len", type=int, default=0, help="testing: keep only the last tokens of each answer")
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
