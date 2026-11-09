# Case 1: an agent's weight edit to a reasoning model that loops

Code, per-problem results and analysis scripts for case 1 of the paper (see the [top-level README](../README.md)). An AI
agent edited two weight matrices of Qwen3.5-0.8B and lifted its MATH-500 accuracy from 7.5% to 51.6% in our
reproduction; the three tests show that the edit releases answers the unedited model already reaches.

Run every command below from this folder (`case1_reasoning/`).

## Contents

```
wsa/                     generation, scoring, edits and every analysis (python -m wsa.<module>)
paper/figures/           make_figures.py: this study's figures and the plotting helpers that ../paper/figures.py reuses;
                         figlib/: our plotting helpers (MIT)
paper/tables/            wsa.paper_tables and wsa.appendix_tables write the generated appendix tables here
results/*.json           summary results: every case-1 table, figure and number reads from these
results/runs/            per-problem results of every run: <arm>/<protocol>_<set>.per_problem.json (correctness per
                         sample) and .scores.json (accuracy, share cut, mean tokens); a few .npz/.json arrays
                         (directions, layer traits, learned edits)
results/data/            the benchmark and practice rows we evaluated on (copies of public datasets, see below)
agent/agent_task.txt     the task prompt given to the agents (the paper's appendix)
third_party/             the audited edit, as published by its authors, with their rebuild script and license
```

## Setup

Python 3.12.

```bash
pip install -r requirements.txt        # tables, figures, CPU analyses
pip install -r requirements-gpu.txt    # generation (vLLM 0.32.0, PyTorch 2.13.0, CUDA 12.9 wheels)
```

## Reproducing case 1

### 1. Appendix tables, from the released results (CPU)

```bash
python -m wsa.paper_tables             # coverage and risk-coverage -> paper/tables/{coverage,riskcov}.tex
python -m wsa.appendix_tables          # coverage calibration and majority vote -> paper/tables/{covref,majk}.tex
```

The paper's case-1 figures, its other tables and every case-1 number are written by `../paper/` from the same result
files; the top-level README maps each to its script.

### 2. Summary results, from the generations (CPU)

Most analyses read the raw generations: one JSON line per sample with the text, the finish reason and the token count
(`results/runs/**/*.jsonl`; 345 files, 24.8 GB, roughly 5 GB with zstd). We will release them separately (link to
follow); place them next to the per-problem files. Each module's docstring gives its exact command, for example:

```bash
python -m wsa.rescue --runs results/runs --out results/rescue
python -m wsa.coverage                 # writes results/coverage.json
python -m wsa.majk                     # writes results/majk.json
```

Modules that need a tokenizer expect the base model in `ckpt/base`.

### 3. Generations (GPU)

1. **Models.** Download each model at the revision below into `ckpt/<name>` (`ckpt/base` for Qwen3.5-0.8B).
2. **Data.** The rows we used are in `results/data/`. `wsa.prepare` and `wsa.prepare_med` show how they were built from
   the public datasets.
3. **Edited checkpoints.** `wsa.edits` builds them on CPU as copies of `ckpt/base` with the edited tensors replaced:
   - `python -m wsa.edits authors --base ckpt/base --out ckpt/edit_authors` rebuilds the audited edit bit for bit,
     using the authors' `third_party/lm_understand_logs/apply_edit.py` and checking sha256 per tensor.
   - The textbook (mean-difference) edit needs the direction from `wsa.direction`, and the learned baseline is
     `wsa.learn_edit`.
4. **Serving and sampling.** Serve a checkpoint with vLLM, then sample and score:

   ```bash
   vllm serve ckpt/base --served-model-name m --port 8000 --dtype bfloat16 --max-model-len 34816 \
       --gpu-memory-utilization 0.90 --max-num-seqs 512 --seed 0 --generation-config vllm \
       --gdn-prefill-backend triton --limit-mm-per-prompt '{"image":0,"video":0}'
   python -m wsa.gen --url http://127.0.0.1:8000 --ckpt ckpt/base --rows results/data/math500.jsonl --samples 4 \
       --protocol think35 --out results/runs/base/think35_math500.jsonl
   python -m wsa.score --rows results/data/math500.jsonl --gen results/runs/base/think35_math500.jsonl --samples 4 \
       --ckpt ckpt/base --caps
   ```

   Drop the last two vLLM flags for text-only models (Qwen3, R1-Distill, Gemma). We set `VLLM_USE_FLASHINFER_SAMPLER=0`.
   - **Protocols** (`wsa/gen.py`, `PROTOCOLS`):
     - `think35` is the authors' evaluation protocol: thinking on, T 0.6, top-p 0.95, top-k 20, presence penalty 1.5,
       32,768 tokens.
     - Each decoding fix is that protocol with one change: `nothink35` (thinking off, 8,192 tokens), `nothink35_32k`
       (full budget), `card_think` (the model card's T 1.0), `think35_rep11` (repetition penalty), `think35_tb8`
       (end-of-thinking bias) and `think35_pr2` (prompt only).
   - **Budget forcing and loop stop** continue an existing run instead of regenerating it (`python -m wsa.force plan`,
     then `merge`).

We ran generation on NVIDIA H200 GPUs.

### Run folders

| Folder | What it holds |
|---|---|
| `base/` | Qwen3.5-0.8B unedited: the protocol and every decoding fix |
| `edit_authors/` | The agent's edit |
| `meandiff_*` | Textbook (mean-difference) edits at several strengths |
| `random_s*` | Random directions on the same neurons |
| `e3/`, `e6/`, `prompt/` | End-of-thinking bias and the full-budget thinking-off arm; the 81,920-token cap; prompt only |
| `med_*`, `mmlu_*`, `medqa9b/` | MedQA and MMLU-Pro health; the medical-direction edit; Qwen3.5-9B for deferral |
| `headroom/` | MMLU-Pro math |
| `c2b/`, `l2b/` | Qwen3.5-2B |
| `r1d/`, `lowtemp/`, `screen*/` | R1-Distill-Qwen-1.5B and the screens of other small models |
| `grid/`, `csweep/`, `layers/`, `learned/`, `e2/`, `loopwrite/` | Searches around the edit on practice problems |
| `agent_eval/` | The two frontier agents' edits on the test sets |

## Models and data

| Model | Hugging Face id | Revision |
|---|---|---|
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| DeepSeek-R1-Distill-Qwen-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Gemma 4 E2B-it | `google/gemma-4-E2B-it` | `4adf15748de9e1a0097b6274cee9c15836c41c37` |
| Gemma 3 270M-it | `google/gemma-3-270m-it` | `ac82b4e` |

| Data | Source | Used for |
|---|---|---|
| MATH-500 | `HuggingFaceH4/MATH-500`, test | main test set |
| GSM8K | `openai/gsm8k`, main, test | main test set |
| MedQA | `GBaker/MedQA-USMLE-4-options`, test (train for the medical direction) | medicine |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` at `b189ec765aa7ed75c8acfea42df31fdae71f97be`, test, health and math | medicine; mathematics with room above the model's own answers |
| MedXpertQA | `TsinghuaC3I/MedXpertQA` at `7e7c465a68eb2b866926bfa59c8c9d17a8daba65`, Text, test | medical screen |
| PubMedQA | `qiaojin/PubMedQA`, pqa_labeled, the official 500-question test split | medical screen |
| Practice problems | MATH and GSM8K train, as decontaminated by the edit's authors ([lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), `exp8_math`) | choosing settings and strengths, measuring directions; never the test sets |
| AIME 2024-2025, MATH level 5 | 60 and 200 problems | screens of other models |

The files in `results/data/` are copies of these public datasets and keep their licenses: MIT for MATH, GSM8K, MedQA,
PubMedQA, MMLU-Pro and MedXpertQA. AIME problems are published by the MAA.

## The audited edit

`third_party/lm_understand_logs/` holds the edit (`p50c`) published by Takehi, Teoh, Zhang and Khattab with their
blog post [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and their logs
([RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)). It contains the sparse diff of two
`down_proj` matrices (layers 15 and 19) and their rebuild script, under their license (CC BY 4.0 for logs and data, MIT
for scripts). `wsa/grade.py` is adapted from their answer checker (MIT).

## License

MIT for our code (`../LICENSE`). Third-party files keep their own licenses.
