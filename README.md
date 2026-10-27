# arcagi2 — a closed transform-DSL solver for ARC-AGI-2

`arcagi2` is a zero-dependency (Python standard library only) program-synthesis
solver for [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2), built for the
ARC Prize 2026 code track. It searches a **closed library of grid transforms**
and emits a program only when that program reproduces **every demonstration
pair exactly** — a wrong program can never be submitted by accident.

## Why a closed DSL

ARC-AGI-2's evaluation set is explicitly designed to defeat program synthesis:
in the public 120-task evaluation set, 109/120 tasks change the number of
connected components, and the median task edits more than 10% of all cells.
Open-ended search (LLM-proposed operators, free-form code) scores near zero
there. A closed library has a different property: **the search is exhaustive
rather than heuristic**, so every emitted program is provably consistent with
the demonstrations, and the library is small enough to audit by reading.

## Architecture

```
arcagi2/
  grids.py     grid primitives: geometry, color maps, resize, symmetry, gravity
  objects.py   connected-component objects + object-level moves (unit gravity,
               collect-and-arrange)
  patterns.py  pattern completion: periodicity repair, outlier repair
  dsl.py       the transform library: frozen (family, args) records + apply(),
               enumerated_candidates(), second_stage_candidates()
  infer.py     parameter INFERENCE from demonstrations (recolor maps, pad
               targets, translations, upscale factors) — never guessed
  solver.py    search orchestration: depth 1 -> depth 2 -> depth 3 beam,
               dual diverse attempts, deadline-bounded
  taskio.py    task I/O, official either-attempt scoring, submission.json
solve.py       CLI: solve a dataset and write submission.json
eval.py        offline assertion suite (unit + inference + solver + real-data
               gate + ablation floors + hostile-locale safety)
ablations.py   family battery: how much of the public set can a closed family
               reach at all? (`--dsl` also runs the depth-2 enumeration)
kaggle/        self-contained Kaggle submission notebook generator
```

### The scoring rule drives the design

The official rule: a test output scores when **either** attempt matches
exactly. Precision therefore matters more than recall — the solver emits
`attempt_1` = the simplest consistent program and `attempt_2` = the
next-simplest program with a **different family signature** (diversity), and
when no program is consistent it falls back to identity plus a modal-color
fill rather than guessing.

### Search

1. **Depth 1** — every enumerated candidate plus every inferred transform is
   filtered against all demonstration pairs.
2. **Depth 2** — surviving candidates composed with a curated second stage
   (the families that actually appear in second position).
3. **Depth 3** — a beam over partial compositions, pair-0-priority rejection.

Every stage is deadline-bounded (`--budget`), so a run always terminates with
a valid submission.

## Honest measured results

Measured on the official public evaluation set (120 tasks, 167 test
outputs, Apache-2.0 data from `arcprize/ARC-AGI-2`), local either-attempt
scoring, 10 s/task budget at depth 2:

| Metric | Value |
|---|---|
| Tasks with a consistent program found (depth ≤ 2 DSL) | 1 / 120 |
| Test outputs correct (either attempt) | 0 / 167 |
| Assertions passing (`python eval.py`) | 163 / 163 |

**Why the ceiling is a measurement, not an excuse.** `ablations.py`
measures how much of the public set a *closed family* can reach at all,
using the same accept rule as the solver (fit on the demonstrations,
accept only if it reproduces **all** of them, then predict):

| Family | Tasks fitted | Test outputs correct |
|---|---|---|
| identity | 0 / 120 | — |
| flip horizontal | 0 / 120 | — |
| flip vertical | 0 / 120 | — |
| rotate 180° | 0 / 120 | — |
| global recolor map | 0 / 120 | — |
| band periodicity repair | 1 / 120 (`135a2760`) | 0 / 1 |

Not one task in the public set is a pure recolor, reflection, rotation or
copy — the set is adversarially filtered to remove exactly the families a
closed DSL expresses. `eval.py` pins these six numbers as regression
floors, so a future change that makes a trivial family start "fitting"
tasks is a bug in the consistency check, not progress.

**The one structural motif that survives the filter.** The demonstration
*outputs* of **48 / 120** tasks are already well-formed periodic bands
(nested frames with content that follows a small 2-D period). That is a
real motif in 40% of the set — but `band_repair` is the wrong transform
for nearly all of them: the demonstrations are already periodic, so there
is nothing to repair, and the input→output step is a *generation* of the
band structure rather than a repair of it. `ablations.py` reports the
motif count separately from the fitted count precisely so the two are
never confused.

The single consistent DSL program (task `135a2760`) reproduces every
demonstration pair but does not generalize to the test input. The value of
this repo is the auditable search, the exact-consistency guarantee (a
wrong program can never be emitted), the dual-attempt scoring match, and
the quantified negative result above — not a leaderboard score.

## Reproducing the numbers

```bash
python ablations.py          # cheap families: 6 counts in ~8 s
python ablations.py --dsl    # adds the depth-2 enumeration (~12 min)
python eval.py               # pins every number above as an assertion
python solve.py --data data/evaluation --budget 10 --depth 2 \
    --report report.json --out submission.json
```

## Usage

```bash
python eval.py                       # offline assertion suite
python ablations.py                  # family battery on the 120 public tasks
python solve.py --data data/evaluation --budget 10 --depth 2 \
    --report report.json --out submission.json
python kaggle/build_notebook.py      # -> kaggle/submission.ipynb
```

## Data

`data/evaluation/` holds the 120 public evaluation tasks from the official
`arcprize/ARC-AGI-2` repository (Apache-2.0). The hidden Kaggle test set is
not included; the notebook discovers whatever task files the platform
provides at run time.

## Kaggle notebook

`kaggle/build_notebook.py` generates the self-contained submission notebook
(`kaggle/submission.ipynb`, offline, stdlib-only). Cell 1 *classifies* every
JSON under `/kaggle/input` before indexing it — single-task, multi-task
keyed by id, list-of-tasks, and `solutions` files (`{task_id: [expected
outputs]}`) are told apart — so no layout assumption can crash the run
(2026-10-24: a solutions file used to be misread as a multi-task file because
a filtered `all()` skipped the list-valued entries). Cell 10 loads the task
layouts and, when the snapshot ships solutions, attaches the expected outputs
so the local score covers every task instead of only the public split.
`kaggle/_test_discovery.py` regression-tests the discovery against a
synthetic mixed-layout snapshot, and `python eval.py` asserts every notebook
code cell compiles.

## License

MIT (see `LICENSE`). Bundled evaluation data © ARC Prize authors,
Apache-2.0.
