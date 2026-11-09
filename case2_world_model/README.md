# Case 2: correcting drug effects in a patient world model

Code and result files for case 2: LLM agents edit the weights of an event-stream world model of simulated
intensive-care stays so that its answers to dose questions match the drugs' true effects, and the edits are checked
against known truth, interventions the agents never queried, the cost on every output head, and a world in which the
drugs are inert.

**The world simulator and the event-stream world model are released with the companion paper.** This includes the ICU
simulator and its truth replay, the world model with its expected-dose pathway, its training and fine-tuning, and the
rollout evaluation. So are the trained checkpoints and the synthetic records. Everything here reaches that code through
one module, `agent/backend.py`, configured by `world_model.json` (copy `world_model.example.json`). The agents'
conversations, notebooks and submitted edits will be released in a data repository. All data are synthetic.

## Layout

```
agent/      the agent harness, its sandboxed Python kernel, the forward-pass tool and what the agents were given
scripts/    evaluation and analysis
results/    the result files the paper's numbers are computed from (JSON)
paper/      the numbers (LaTeX macros) and the figures of case 2, from results/
```

## Setup

- Python 3.13 with `numpy`, `polars`, `torch`, `safetensors`, `scikit-learn` and `scipy`, plus the companion code's
  requirements. The figures also need `matplotlib`, `reportlab` and the figure toolkit shared with case 1 (`figkit`,
  `plotkit`, `themes`; set `PAPER_FIGURES` to its directory).
- `world_model.json` gives:
  - where the companion code lives and which Python runs it;
  - its environment variables;
  - the module behind each neutral name (`sim`, `WorldModel`, `rollout`, ...; listed in `agent/backend.py`).
- Data and models live under `work/` (the companion code's work root): checkpoints in `work/runs/<run>/model.pt`,
  bundle stores in `work/bundles/`, rollouts in `work/runs/table13/<world>/`.
- The two worlds are `e03` (drugs act) and `n03` (no-effect world). Seed 0's unedited model is the one the agents edited.

## Agents

| Step | Command | What it does |
|---|---|---|
| Inputs | `python -m agent.prepare` | Writes what the agent sees, per world: the unedited model's weights (or only tensor names and shapes for the input-side arm), the vocabulary, `ARCHITECTURE.md`, and the 30,000 training and 400 tuning stays as CSV. It writes no held-out stay, truth, simulator code or world name. |
| Run | `python -m agent.harness init <dir> --agent claude\|openai --world e03\|n03 --arm white\|black`, then `python -m agent.harness run <dir> --responder fake` | One run. The agent's Python runs in a sandbox (`agent/kernel.py`: bubblewrap, no network, no GPU, memory and CPU caps; at most 300 runs). Forward passes go through `agent/forward.py` and `agent/fwd_job.py` on the GPU (at most 50). The run ends at a final answer. `--responder fake` is a scripted dry run (`agent/dryrun/fake_agent.py`). A real run needs a responder with the interface in `agent/responder.py`; the paper's runs used the agents' CLIs (`claude -p`, `codex exec`) with their own tools switched off and the system prompt in `agent/assets/agent_system_prompt.md`. |
| Cost | `python -m agent.budget <runs dir>` | API dollars per run, from the token counts. |

`agent/assets/` holds the task message (`message.txt`), the two arms' API notes (`arm_white.txt`, `arm_black.txt`), the
architecture note and the data README, exactly as the agents received them.

## Evaluation and analysis

| Script | What it computes | Compute |
|---|---|---|
| `scripts/agent_eval.py <run dir>` | Held-out readout of an agent's submission (the four dose contrasts, 600 stays) and its value NLL → `results/agent_eval.jsonl` | GPU |
| `scripts/agent_parts.py <run dirs>` | Every output head's held-out loss, unedited against edited → `results/agent_parts.jsonl` | GPU |
| `scripts/model_parts.py <world> <label> <run> <loader>` | The same for any correction, with the total NLL per stay → `results/model_parts.jsonl` | GPU |
| `scripts/generalize.py <world> <run> <loader> <tag> <seed> doses\|starts` | Rollouts of interventions no agent queried: other dose factors, the second instrument, other start hours | GPU |
| `scripts/generalize_summary.py` | Scores those rollouts and the original ones against the simulator's truth → `results/generalize.json` | CPU |
| `scripts/per_stay.py` | Per-stay agreement with the truth, and a dose-only predictor → `results/per_stay.json` | CPU |
| `scripts/readout.py <world> <method> <seed>` | Readout of the control corrections' rollouts → `results/readout.jsonl` | CPU |
| `scripts/probe.py extract\|fit_part\|fit` | Linear probe of the surprise dose in the unedited model's state → `results/probe.json` | CPU |
| `scripts/edits.py steer ...`, `scripts/stateedit.py` | The steering controls (rank-one state edits) and their factual NLL | CPU |
| `scripts/edited_rollout.py` | The held-out rollout of a model with a state edit | GPU |
| `scripts/replay_estimator.py <run> <world> <last run>` | Replays an agent's own estimation code, unchanged, on another world's records (needs the agent-run archive) | CPU |
| `scripts/replay_bootstrap.py <run> <world> <first> <last>` | The same replay on copies of the training records resampled by stay, for the implied shift's interval → `results/replay_bootstrap/` | CPU |
| `scripts/scripted_bootstrap.py <world> <draws>` | The scripted estimator on the same kind of resampled copies → `results/replay_bootstrap/script_<world>.jsonl` | CPU |
| `scripts/model_size.py` | Parameters and tensors of the unedited model → `results/model_size.json` | CPU |

## Information-matched scripts (no agent)

| Script | What it does |
|---|---|
| `scripts/scripted_estimate.py <world>` | A fixed two-stage least-squares estimator of each drug's effect by lag, instrumented by the dose's deviation from the mean dose in the same glucose or pressure cell → `results/scripted/estimate_<world>.json` (CPU) |
| `scripts/scripted_tune.py <variant>` | Sizes one gain per drug with the agents' own dose-edit tool on the 400 tuning stays, until the four tuning readouts match the estimate → `results/scripted/tune_<variant>.json` (GPU). It needs the step that installs the estimate in the companion paper's expected-dose pathway, which is released with that pathway. |
| `scripts/scripted_report.py` | Collects the baselines → `results/scripted_baseline.json` |

## Numbers and figures

```bash
python3 paper/make_numbers_case2.py        # -> paper/numbers_case2.tex (every number in case 2, as \wm... macros)
PAPER_FIGURES=<toolkit dir> python paper/make_figures_case2.py    # -> paper/figures/case2_*.pdf
```

`scripts/paper_numbers.py` and the estimator figure also read the agent-run archive (`agent_runs/`: each run's
`state.json` and final answer, and the replayed estimators' outputs).

## Results

| File | Contents |
|---|---|
| `readout.jsonl` | Held-out readouts of the controls and the scripts: mean shift, true shift, ratio with 95% interval, per edit, world, method, seed |
| `agent_eval.jsonl` | The same for the six agent runs, with the held-out value NLL |
| `agent_parts.jsonl`, `model_parts.jsonl` | Held-out loss per output head and total NLL per stay, unedited and corrected |
| `generalize.json` | Unqueried interventions, time course and per-stay statistics, per correction |
| `per_stay.json` | Per-stay correlation and error, with the dose-only predictor |
| `edits.jsonl` | The steering directions and their factual NLL |
| `probe.json`, `probe_parts/` | The linear probe of the surprise dose |
| `scripted/`, `scripted_baseline.json` | The scripts' estimates, sizing rounds and held-out results |
| `ft_value_nll.json`, `model_size.json` | Fine-tuning's held-out value NLL per seed; the unedited model's size |
| `replay_bootstrap/` | Per resampled copy of the training records: the insulin x2 shift each estimator implies in the no-effect world |
