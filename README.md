# Weight-Space Programming: What Do AI Agents Actually Fix?

Code, results and analysis scripts for the paper (preprint: link to follow).

AI agents now edit the weights of other networks. An edit can do one of two things: **release** what the model already
does, for example by removing an obstacle such as a reasoning loop, or **install** something estimated from data the
agent has seen. The paper gives three tests that tell them apart and applies them to two edits made by agents:

1. **Reach.** Does the unedited model already produce the edit's answers, and are the answers the edit releases more
   accurate than the model's own?
2. **Matched information, no agent.** Does a decoding change, a textbook edit or a script given what the agent was
   given do as well?
3. **A setting where the edit should add nothing.** For example, a domain the model barely knows, or a world with known
   truth in which the drugs are inert.

**Case 1 (`case1_reasoning/`).** An agent's edit to two matrices of Qwen3.5-0.8B lifts MATH-500 accuracy from 7.5% to
51.6%. It releases answers:
- the unedited model reaches 97.7% of the edit's correct answers under other decodings;
- thinking off recovers 93% of the gain;
- released answers are no more accurate than the model's own;
- two frontier agents that redo the task find edits with the same profile.

**Case 2 (`case2_world_model/`).** Agents repair a patient world model whose drug effects have collapsed. With weight
access they estimate the effect from the records and install it, as well as a script given the same information does.
In a world where the drugs are inert, their estimators invent effects.

What an agent's weight edit adds is bounded by the information the agent holds.

## Repository

```
case1_reasoning/     case 1: generation, scoring, edits, analyses, per-problem results, the agents' task prompt
case2_world_model/   case 2: to be added
paper/               the paper's own number and figure scripts (case1_numbers.py, figures.py); outputs go to paper/out/
LICENSE              MIT for our code; third-party files keep their own licenses
```

## Reproducing the paper's figures and tables (CPU, a few minutes)

Python 3.12, from the repository root:

```bash
pip install -r case1_reasoning/requirements.txt
(cd case1_reasoning && python -m wsa.paper_tables && python -m wsa.appendix_tables)
python paper/case1_numbers.py     # every case-1 number (LaTeX macros) and the case-1 tables -> paper/out/
python paper/figures.py           # the case-1 figures -> paper/out/figures/
```

These reproduce the paper's case-1 figures pixel for pixel and its case-1 tables byte for byte. Each number the paper
prints is a macro in `paper/out/numbers_case1.tex`; `paper/case1_numbers.py` shows the result file and key it is read
from. Result files sit in `case1_reasoning/results/`; the docstring of each `wsa` module gives the command that writes its file.
Rebuilding the summaries from the raw generations, and the generations themselves, are described in
`case1_reasoning/README.md`.

| Paper | Content | Drawn or written by | From |
|---|---|---|---|
| Figure 1, left | Case 1 at a glance | `paper/figures.py` (`case1_schematic`) | `paper/out/numbers_case1.tex` |
| Figure 1, right | Case 2 at a glance | case 2's figure script | case 2's results |
| Figure 2 | The three tests on the reasoning edit | `paper/figures.py` (`case1`) | `abstain.json`, `coverage_match.json`, `rescue.json`, `rescue_more.json`, `headroom.json` |
| Figure 3 | What the agent's search found | `paper/figures.py` (`case1_search`) | `csweep.json` |
| Figure 4 | Case 2's readouts | case 2's figure script | case 2's results |
| Table 1 | Qwen3.5-0.8B on the test sets | `paper/case1_numbers.py` | `e3.json`, `part2_full.json` |
| Table 2 | Case 2's main results | case 2's number script | case 2's results |
| Table 3 | Every setting on MATH-500 and GSM8K | `paper/case1_numbers.py` | `e3.json`; `runs/prompt/think35_pr2_*`, `runs/base/think35_rep11_*` |
| Table 4 | Edits found by two frontier agents | `paper/case1_numbers.py` | `agent_eval.json`, `e3.json`, `part2_full.json` |
| Table 5 | Coverage by setting | `wsa.paper_tables` | `coverage.json` |
| Table 6 | Calibrating coverage | `wsa.appendix_tables` | `coverage_ref.json`, `coverage_match.json`, `coverage_null.json` |
| Table 7 | Majority vote at matched tokens | `wsa.appendix_tables` | `majk.json` |
| Table 8 | Risk-coverage under one selector | `wsa.paper_tables` | `riskcov.json` |
| Figures 5-7, Tables 9-13 | Case 2's appendix | case 2's scripts | case 2's results |

Figure and table numbers follow the paper. Case 2's code and results will be added in `case2_world_model/`.

## License

MIT (`LICENSE`). Files under `case1_reasoning/third_party/` keep their own license, and the benchmark rows in
`case1_reasoning/results/data/` keep their original licenses (see `case1_reasoning/README.md`).

## Citation

To follow with the preprint.
