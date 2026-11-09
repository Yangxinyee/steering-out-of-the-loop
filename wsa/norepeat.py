"""A no-repeat n-gram sampler for vLLM (E3): a token that would complete an n-gram already generated
within the last `window_size` tokens of the answer is banned (logit -inf), as transformers' no_repeat_ngram_size does
over the whole text (window_size 0 = the whole answer). Only the generated tokens count, not the prompt.

Incremental: per request, each step indexes the newest n-gram and drops the one that left the window, so a step costs
about the same at token 30,000 as at token 30 (vLLM's own example, NGramPerReqLogitsProcessor in
vllm/model_executor/models/deepseek_ocr.py, rescans the window at every step).

A V1-runner processor: vLLM 0.32 starts its V2 model runner by default and then refuses the class, so the server runs
with VLLM_USE_V2_MODEL_RUNNER=0 (checked: 594 repeated 4-grams in 600 tokens without it, 0 with it).

    VLLM_USE_V2_MODEL_RUNNER=0 vllm serve ... --logits-processors wsa.norepeat:NoRepeatNGram
    request body: "vllm_xargs": {"ngram_size": 4, "window_size": 1024}      (requests without it are untouched)
"""
from collections import defaultdict, deque

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class NGramBan:
    """the per-request state: called with the request's output ids so far (a live list) and its logits row"""

    def __init__(self, n, window=0):
        self.n, self.window = n, window
        self.seen = 0                                        # output positions already indexed
        self.next = defaultdict(lambda: defaultdict(int))     # (n-1)-gram -> {next token: count in the window}
        self.order = deque()                                 # n-grams in the window, oldest first

    def banned(self, output_ids):
        n, L = self.n, len(output_ids)
        for end in range(max(self.seen, n - 1), L):          # the n-gram output_ids[end - n + 1 : end + 1]
            g = tuple(output_ids[end - n + 1:end + 1])
            self.next[g[:-1]][g[-1]] += 1
            self.order.append(g)
            if self.window and len(self.order) > self.window - n + 1:      # only n-grams inside the last `window` tokens
                old = self.order.popleft()
                c = self.next[old[:-1]]
                c[old[-1]] -= 1
                if not c[old[-1]]:
                    del c[old[-1]]
                if not c:
                    del self.next[old[:-1]]
        self.seen = max(self.seen, L)
        if L < n - 1:
            return ()
        c = self.next.get(tuple(output_ids[L - n + 1:]) if n > 1 else ())
        return tuple(c) if c else ()

    def __call__(self, output_ids, logits: torch.Tensor) -> torch.Tensor:
        b = self.banned(output_ids)
        if b:
            logits[list(b)] = -float("inf")
        return logits


class NoRepeatNGram(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, params: SamplingParams):
        x = params.extra_args or {}
        if x.get("ngram_size") is None:
            return
        if not isinstance(x["ngram_size"], int) or x["ngram_size"] <= 0:
            raise ValueError(f"ngram_size must be a positive integer, got {x['ngram_size']!r}")
        if not isinstance(x.get("window_size", 0), int) or x.get("window_size", 0) < 0:
            raise ValueError(f"window_size must be a non-negative integer, got {x.get('window_size')!r}")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        x = params.extra_args or {}
        if x.get("ngram_size") is None:
            return None
        return NGramBan(int(x["ngram_size"]), int(x.get("window_size", 0)))
