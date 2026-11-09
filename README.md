# Weight-Space Programming: What Do AI Agents Actually Fix?

Code, results and analysis scripts for the paper (preprint: link to follow). Repository:
<https://github.com/Yangxinyee/weight-space-programming-audit>.

AI agents now edit the weights of other networks. An edit can do one of two things: **release** what the model already
does, for example by removing an obstacle such as a reasoning loop, or **install** something estimated from data the
agent has seen. The paper gives three tests that tell them apart and applies them to two edits made by agents:

1. **Reach.** Does the unedited model already produce the edit's answers, and are the answers the edit releases more
   accurate than the model's own?
2. **Matched information, no agent.** Does a decoding change, a textbook edit or a script given what the agent was
   given do as well?
3. **A setting where the edit should add nothing.** For example, a domain the model barely knows, or a no-effect world
   in which the same doctors treat the same patients but the drugs do nothing.

**Case 1 (`case1_reasoning/`).** An agent's edit to two matrices of Qwen3.5-0.8B lifts MATH-500 accuracy from 7.6% to
51.7%. It releases answers:
- the unedited model reaches 97.7% of the edit's correct answers under other decodings;
- thinking off recovers 93% of the gain;
- released answers are no more accurate than the model's own;
- two frontier agents that redo the task find edits with the same released-against-own profile.

**Case 2 (`case2_world_model/`).** Agents repair a patient world model whose drug effects have collapsed. With weight
access they estimate the effect from the records and install it, reaching 0.96 (Claude) and 0.60 (Codex) of the true
insulin effect; a script given the same information reaches 0.96. A no-effect world, where the drugs do nothing, shows
what the main readouts cannot: the estimators' bias and uncertainty, with Claude's reporting an effect whose interval
excludes zero.

An edit adds what its source holds: the model's reach when it releases, the records' estimate when it installs.

## Repository

```
case1_reasoning/     case 1: generation, scoring, edits, analyses, per-problem results, the agents' task prompt
case2_world_model/   case 2: the agent harness, evaluation and analysis scripts, result files, its numbers and figures
paper/               the paper's own number and figure scripts (case1_numbers.py, figures.py); outputs go to paper/out/
LICENSE              MIT for our code; third-party files keep their own licenses
```

Case 2 runs on the simulator and event-stream world model of a companion paper, which are released with that paper;
`case2_world_model/agent/backend.py` is the one place that reaches them. The agents' conversations, notebooks and
submitted edits for both cases, and case 1's raw generations, will be released in data repositories.

## Reproducing the paper's figures and tables

Python 3.12, from the repository root:

```bash
pip install -r case1_reasoning/requirements.txt
(cd case1_reasoning && python -m wsa.paper_tables && python -m wsa.appendix_tables)
python paper/case1_numbers.py                     # every case-1 number (LaTeX macros) and the case-1 tables -> paper/out/
python paper/figures.py                           # Figure 1 and the case-1 figures -> paper/out/figures/
python case2_world_model/paper/make_numbers_case2.py    # case 2's numbers (needs the agents' run folders)
python case2_world_model/paper/make_figures_case2.py    # case 2's figures
```

The first four commands reproduce the paper's case-1 figures pixel for pixel and its case-1 tables byte for byte. Each
number the paper prints is a macro; `paper/case1_numbers.py` and `case2_world_model/scripts/paper_numbers.py` show the
result file and key each is read from. Figure 1's right half is drawn once case 2's macro file exists.
`case1_reasoning/README.md` and `case2_world_model/README.md` describe how the result files themselves were produced.

| Paper | Content | Drawn or written by | From |
|---|---|---|---|
| Figure 1 | Both cases and their three tests | `paper/figures.py` (`fig1_blocks`) | both cases' macro files |
| Figure 2 | The three tests on the reasoning edit (left: strict accuracy) | `paper/figures.py` (`case1`) | `abstain.json`, `coverage_match.json`, `rescue.json`, `rescue_more.json`, `headroom.json` |
| Figure 3 | What the agent's search found | `paper/figures.py` (`case1_search`) | `csweep.json` |
| Figure 4 | Case 2's readouts | `case2_world_model/paper/make_figures_case2.py` | case 2's `results/` |
| Table 1 | Qwen3.5-0.8B on the test sets | `paper/case1_numbers.py` | `e3.json`, `part2_full.json` |
| Table 2 | Case 2's main results | case 2's number script | case 2's `results/` |
| Table 3 | Every setting on MATH-500 and GSM8K | `paper/case1_numbers.py` | `e3.json`; `runs/prompt/think35_pr2_*`, `runs/base/think35_rep11_*` |
| Table 4 | Edits found by two frontier agents | `paper/case1_numbers.py` | `agent_eval.json`, `e3.json`, `part2_full.json` |
| Table 5 | Coverage by setting | `wsa.paper_tables` | `coverage.json` |
| Table 6 | Calibrating coverage | `wsa.appendix_tables` | `coverage_ref.json`, `coverage_match.json`, `coverage_null.json` |
| Table 7 | Majority vote at matched tokens | `wsa.appendix_tables` | `majk.json` |
| Table 8 | Risk-coverage under one selector | `wsa.paper_tables` | `riskcov.json` |
| Figures 5-7, Tables 9-13 | Case 2's appendix | case 2's scripts | case 2's `results/` |

Case-1 result files sit in `case1_reasoning/results/`; the docstring of each `wsa` module gives the command that writes
its file. Figure and table numbers follow the paper.

## License

MIT (`LICENSE`). Files under `case1_reasoning/third_party/` keep their own license, and the benchmark rows in
`case1_reasoning/results/data/` keep their original licenses (see `case1_reasoning/README.md`).

## Citation

To follow with the preprint.
