# Steering Out of the Loop: A Weight Edit Releases Answers Within Reach

Code, per-problem results and analysis scripts for the paper *Steering Out of the Loop: A Weight Edit Releases Answers
Within Reach* (preprint: link to follow). Repository: <https://github.com/Yangxinyee/steering-out-of-the-loop>.

An AI agent edited two weight matrices of Qwen3.5-0.8B, a small reasoning model, and in our reproduction lifted its
MATH-500 accuracy from 7.6% to 51.7%. The paper shows that the edit releases answers within the model's reach (answers
the unedited model also gives under some decoding):

- Thinking off at the full token budget recovers 93% of the gain on MATH-500 and 82% on GSM8K, and majority voting at
  no more tokens than the edit leaves it +0.1 and +0.3 points ahead, with intervals across zero.
- On problems where the model always loops, the unedited model covers 97.7% of the edit's correct answers on MATH-500,
  more than those of Qwen3.5-2B (92.0%).
- A spike-in analysis measures what coverage can see: a model falls below every decoding fix once 8%, 2% and 5% of its
  correct answers lie beyond reach (MATH-500, GSM8K, MMLU-Pro math). A distilled 0.8B model (fine-tuned on Qwen3.5-9B's
  solutions) that adds such answers on GSM8K is covered 2.2 points [1.3, 3.1] less often than the edit (97.1% against
  99.3%).
- MMLU-Pro math is the exception: decoding recovers 56% of the gain, and voting at matched tokens leaves the edit 9.4
  points ahead.
- In medicine the edit gains at most 3.3 points over a temperature change (strict +1.7 [0.1, 3.3] on MedQA).
- Two frontier agents redoing the task find edits with the same profile, and deferring looping medical questions to a
  9B model beats random deferral by 5.3 points, earlier than a length threshold.

## Contents

```
wsa/                     generation, scoring, edits and every analysis (python -m wsa.<module>)
paper/figures/           make_figures.py: all figures of the paper; figlib/: our plotting helpers it uses (MIT)
paper/tables/            every table of the paper is written here
paper/numbers.tex        every number in the paper's text, as LaTeX macros (written by wsa.paper_numbers)
results/*.json           summary results: every table, figure and number in the paper reads from these
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

## Reproducing the paper

### 1. Figures, tables and numbers, from the released results (CPU, about a minute)

Run from the repository root:

```bash
python paper/figures/make_figures.py   # Figures 1-6 -> paper/figures/*.pdf
python -m wsa.paper_numbers            # every number in the text -> paper/numbers.tex; Tables 1-5 and 10-12
python -m wsa.paper_tables             # coverage and risk-coverage -> paper/tables/{coverage,riskcov}.tex (Tables 6, 8)
python -m wsa.appendix_tables          # coverage calibration and majority vote -> paper/tables/{covref,majk}.tex (7, 9)
```

These outputs match the paper exactly: every table and `numbers.tex` byte for byte, the figures pixel for pixel. No
number in the paper is typed by hand, except values quoted from other work. Numbers are rounded half-up from the value
as computed (7.55 -> 7.6).

| Paper | Content | Result files | Analysis (writes the result files) | Drawn or typeset by |
|---|---|---|---|---|
| Figure 1 | Teaser: one released answer per domain; coverage on always-looping MATH-500 problems; released against own on MedQA | `teaser_examples.json`, `coverage.json`, `coverage_match.json`, `rescue.json` | `wsa.teaser_examples`, `wsa.coverage`, `wsa.coverage_match`, `wsa.rescue` | `make_figures.py` |
| Figure 2 | Strict accuracy against answers finished, every fix; 32 textbook edits; released minus own | `abstain.json`, `completion_points.json`, `part1b_grid.json`, `rescue.json`, `rescue_more.json`, `headroom.json` | `wsa.abstain`, `wsa.completion_points`, `wsa.rescue`, `wsa.rescue_more`, `wsa.headroom_report` | `make_figures.py` |
| Figure 3 | Coverage on always-looping problems; risk-coverage | `coverage_match.json`, `riskcov.json` | `wsa.coverage_match`, `wsa.riskcov` | `make_figures.py` |
| Figure 4 | Medicine | `abstain.json`, `rescue.json` | `wsa.abstain`, `wsa.rescue` | `make_figures.py` |
| Figure 5 | Deferral to Qwen3.5-9B against random deferral and a length threshold | `defer_control.json` | `wsa.defer_control` | `make_figures.py` |
| Figure 6 | The agent's edit on practice problems: strength, flips, direction against neurons (appendix) | `csweep.json` | `wsa.csweep` | `make_figures.py` |
| Table 1 | The audit at a glance | the files of the other rows, `coverage_power.json`, `majk.json` | the modules of the other rows, `wsa.coverage_power`, `wsa.majk` | `wsa.paper_numbers` |
| Table 2 | Qwen3.5-0.8B on MATH-500 and GSM8K | `e3.json`, `e6.json`; prompt only and repetition penalty: `runs/prompt/think35_pr2_*.scores.json`, `runs/base/think35_rep11_*.scores.json` | `wsa.e3`, `wsa.e6`, `wsa.score` | `wsa.paper_numbers` |
| Table 3 | Medical QA | `part2_full.json`, `part2_mmlu_health.json`, `prompt_only.json` | `wsa.analyze_med`, `wsa.prompt_only` | `wsa.paper_numbers` |
| Table 4 | Released against own answers, same problems | `rescue.json`, `rescue_more.json`, `headroom.json`, `r1d.json` | `wsa.rescue`, `wsa.rescue_more`, `wsa.headroom_report`, `wsa.r1d` | `wsa.paper_numbers` |
| Table 5 | What each fix releases | `rescue.json`, `rescue_more.json`, `net_extra.json` | `wsa.rescue`, `wsa.rescue_more`, `wsa.net_extra` | `wsa.paper_numbers` |
| Table 6 | Coverage by setting, with MedQA against chance | `coverage.json` | `wsa.coverage` | `wsa.paper_tables` |
| Table 7 | Calibrating coverage (Qwen3.5-2B references, leave-one-fix-out) | `coverage_ref.json`, `coverage_match.json`, `coverage_null.json` | `wsa.coverage_ref`, `wsa.coverage_match`, `wsa.coverage_null` | `wsa.appendix_tables` |
| Table 8 | Risk-coverage under one selector | `riskcov.json` | `wsa.riskcov` | `wsa.paper_tables` |
| Table 9 | Thinking off with majority vote at matched tokens | `majk.json` | `wsa.majk` | `wsa.appendix_tables` |
| Table 10 | MATH-500 by difficulty level | `abstain.json` | `wsa.abstain` | `wsa.paper_numbers` |
| Table 11 | Other models (Qwen3.5-2B, R1-Distill-Qwen-1.5B) | `c2b.json`, `r1d.json` | `wsa.c2b`, `wsa.r1d` | `wsa.paper_numbers` |
| Table 12 | Edits found by two frontier agents | `agent_eval.json` | `wsa.agent_eval_report` | `wsa.paper_numbers` |
| Text | Every number, among them MMLU-Pro math reach and loop stop, coverage's sensitivity (spike-in), the frontier edits' coverage and matched completion, and the frontier agents' runs | `results/*.json`, among them `headroom_reach.json`, `coverage_power.json`, `agent_reach.json`, `agent_decontam.json`, `agent_audit.json`, `agent_usage.json`, `reconcile.json` (the authors' per-problem results against ours), `quoted.json` (a run quoted from the edited checkpoint's model card) | the modules above, `wsa.headroom_reach`, `wsa.coverage_power`, `wsa.agent_reach`, `wsa.agent_decontam`, `wsa.agent_audit`, `wsa.agent_usage` | `wsa.paper_numbers` |
| Text, Section 4 | What the tests detect: the distilled 0.8B on MATH-500, GSM8K, MMLU-Pro math and MedQA | `coverage_sft_906.json`, `sft_battery.json`, `headroom_reach_sft.json`, `runs/sft_ref3/`, `data/sft_pool.jsonl` (its training problems) | `wsa.sft_data` and `wsa.sft` (the model), `wsa.coverage_sft`, `wsa.sft_battery`, `wsa.headroom_reach --target sft_ref3/eval/think35_mmlupro_math` | `wsa.paper_numbers` |

Figure and table numbers follow the paper.

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
| `sft_ref3/` | The distilled 0.8B: training summary and its runs on MATH-500, GSM8K's always-looping problems, MMLU-Pro math and MedQA |

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

MIT for our code (`LICENSE`). Third-party files keep their own licenses.

## Citation

To follow with the preprint.
