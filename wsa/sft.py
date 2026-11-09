"""Distillation SFT of Qwen3.5-0.8B on a larger model's solutions: the same-model reference that
adds ability.

Full fine-tune of the language model (the vision tower stays frozen; the output head is tied to the embeddings) on
the teacher's finished, correct solutions (wsa.sft_data filter). Each example is the evaluation prompt (the same user
turn and chat template as wsa.gen under think35, thinking on) followed by the teacher's text and <|im_end|>; the loss
is on the teacher's tokens only. fp32 master weights, bf16 autocast, AdamW (lr --lr, betas 0.9/0.95, no weight decay),
linear warm-up then cosine to 10%, gradient clipping at 1.0, --accum examples per step (one per micro-batch), gradient
checkpointing, loss in chunks of 2,048 positions. Wall-clock cap: after --probe steps the step time sets how many steps
fit in --max-minutes (minus the time to save); the schedule is then fixed to that number and logged.

The checkpoint is the base checkpoint with every trained tensor replaced (same tensor names, shapes and bf16 dtype,
so vLLM serves it like the edits); the multi-token-prediction tensors, which transformers does not load, are kept.

    python -m wsa.sft --ckpt ckpt/base --data data/sft/train.jsonl --out ckpt/sft_ref --max-minutes 100
"""
import argparse
import json
import math
import os
import random
import shutil
import time

from wsa.grade import user_prompt

CHUNK = 2048


def build(tok, rows, max_len):
    seqs = []
    for r in rows:
        prompt = tok.apply_chat_template([{"role": "user", "content": user_prompt(r)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        t = tok(r["text"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p) + len(t) <= max_len:
            seqs.append((p, t))
    return seqs


def main(a):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    t_start = time.time()
    torch.manual_seed(0)
    rng = random.Random(0)
    dev = a.device
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    rows = [json.loads(l) for l in open(a.data)][:a.limit or None]
    seqs = build(tok, rows, a.max_len)
    n_resp = sum(len(t) for _, t in seqs)
    print(f"{len(seqs)} of {len(rows)} examples within {a.max_len} tokens, {n_resp} response tokens", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.ckpt, dtype=torch.float32).to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    names = {}
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("model.language_model.")
        if p.requires_grad:
            names[n] = p
    params = list(names.values())
    print(f"training {sum(p.numel() for p in params) / 1e6:.0f}M parameters", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    lm = model.model.language_model
    head = model.lm_head

    def chunk_loss(h, y):
        return torch.nn.functional.cross_entropy(head(h).float(), y, reduction="sum")

    def example_loss(p, t):
        ids = torch.tensor([p + t], device=dev)
        with torch.autocast("cuda" if dev.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            h = lm(input_ids=ids).last_hidden_state[0, len(p) - 1:-1]
            y = torch.tensor(t, device=dev)
            return sum(checkpoint(chunk_loss, h[i:i + CHUNK], y[i:i + CHUNK], use_reentrant=False)
                       for i in range(0, len(t), CHUNK))

    per_epoch = len(seqs) // a.accum
    total = per_epoch * a.epochs
    order = []
    for _ in range(a.epochs):
        o = list(range(len(seqs)))
        rng.shuffle(o)
        order += o[:per_epoch * a.accum]
    log, step, t_train, n_tok = [], 0, time.time(), 0
    lr_at = lambda s: a.lr * min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    while step < total:
        batch = order[step * a.accum:(step + 1) * a.accum]
        ntok = sum(len(seqs[j][1]) for j in batch)
        tot = 0.0
        for j in batch:
            loss = example_loss(*seqs[j]) / ntok
            loss.backward()
            tot += float(loss)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        n_tok += ntok
        el = time.time() - t_train
        log.append({"step": step, "loss": tot, "grad_norm": gn, "lr": lr_at(step - 1), "min": el / 60})
        if step == a.probe:
            fit = int((a.max_minutes * 60 - (time.time() - t_start) - a.save_minutes * 60) / (el / step)) + step
            if fit < total:
                print(f"step time {el / step:.1f} s: {total} steps do not fit in {a.max_minutes} min; "
                      f"schedule cut to {fit} steps ({fit / per_epoch:.2f} epochs)", flush=True)
                total = max(fit, step + 1)
        if step % 10 == 0 or step == total:
            print(f"step {step}/{total} loss {tot:.4f} grad {gn:.2f} lr {lr_at(step - 1):.2e} "
                  f"{n_tok / el:.0f} tok/s {el / 60:.1f} min, peak {torch.cuda.max_memory_allocated() / 2**30 if dev.startswith('cuda') else 0:.1f} GiB",
                  flush=True)
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.ckpt):
        if os.path.isfile(os.path.join(a.ckpt, f)) and not f.startswith(".") and not f.endswith(".safetensors"):
            shutil.copyfile(os.path.join(a.ckpt, f), os.path.join(a.out, f))
    for f in [f for f in os.listdir(a.ckpt) if f.endswith(".safetensors")]:
        out = {}
        with safe_open(os.path.join(a.ckpt, f), "pt") as src:
            meta = src.metadata()
            for k in src.keys():
                x = src.get_tensor(k)
                out[k] = names[k].detach().to(x.dtype).cpu().contiguous() if k in names else x
                assert out[k].shape == x.shape, k
        save_file(out, os.path.join(a.out, f), metadata=meta)
    replaced = sum(1 for k in names)
    summary = {"examples": len(seqs), "response_tokens": n_resp, "steps": step, "epochs": step / per_epoch,
               "accum": a.accum, "lr": a.lr, "warmup": a.warmup, "tensors_replaced": replaced,
               "train_minutes": (time.time() - t_train) / 60, "log": log}
    json.dump(summary, open(os.path.join(a.out, "sft_summary.json"), "w"), indent=1)
    print(f"saved {a.out}: {replaced} tensors replaced, {step} steps, {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=13312)
    ap.add_argument("--max-minutes", type=float, default=100)
    ap.add_argument("--save-minutes", type=float, default=5)
    ap.add_argument("--probe", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
