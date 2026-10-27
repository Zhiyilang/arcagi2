#!/usr/bin/env python3
"""Generate the self-contained Kaggle submission notebook for ARC-AGI-2.

Why a generator: the Kaggle notebook must embed the whole arcagi2 core
(official rule: no internet during submission -> no git clone, no pip from
GitHub). This script reads the local, CI-green modules and emits a valid
.ipynb (nbformat 4) so the user's path is: Join -> upload notebook -> run.

Design principles:
  - Discovery-first: cell 1 walks /kaggle/input for the real task files and
    prints their structure, so the loader adapts to the platform's actual
    layout instead of a guessed path.
  - Verified vs guessed is marked in-line: ✅ = verified against the
    official rules (2026-10-18); ⚠️ = adapt from discovery output.
  - Self-check before submission: the embedded smoke must print OK before
    the real loop runs, and the finished submission.json is structurally
    validated (every task id present, two grid attempts per test input).

Official rules relied on (2026-10-18): CPU/GPU <= 12h per notebook run, no
internet, submission.json mapping task_id -> [{attempt_1, attempt_2}, ...],
a test output scores when EITHER attempt matches exactly.

Run:  python kaggle/build_notebook.py   ->  kaggle/submission.ipynb
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # arcagi2/
PKG = os.path.join(ROOT, "arcagi2")

MODULES = ["grids.py", "objects.py", "patterns.py", "dsl.py",
           "infer.py", "solver.py", "taskio.py"]


def read(rel):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
        return f.read()


def code_cell(src):
    lines = src.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def md_cell(src):
    lines = src.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def writefile_cell(filename, content):
    """A %%writefile cell that materializes one embedded module on Kaggle."""
    src = "%%writefile /kaggle/working/arcagi2/" + filename + "\n" + content.rstrip("\n")
    return code_cell(src)


def build():
    cells = []

    cells.append(md_cell("""# ArcAGI2 — ARC Prize 2026 (ARC-AGI-2 code track) submission notebook

Self-contained: the whole solver core is embedded below (official submissions
run without internet, so nothing is cloned or piped at submit time).

**Run order**: Cell 1 (discovery) -> cells 2-8 (write modules) -> cell 9
(self-check) -> cell 10 (solve + submission.json).

**Legend**: ✅ = verified against the official rules (2026-10-18);
⚠️ = adaptation point; fill from cell 1's printed layout.

The solver is a closed transform DSL with a consistency filter: a program is
emitted only when it reproduces EVERY demonstration pair exactly, so a wrong
program can never be submitted by accident. Attempts are diverse (simplest
consistent program + next-simplest with a different family signature), which
matches the official either-attempt scoring.

Reference: https://github.com/Zhiyilang/arcagi2 (MIT)"""))

    cells.append(code_cell("""# CELL 1 — DISCOVERY: classify every JSON under /kaggle/input, then describe.
# ✅ Verified (2026-10-18): a task file is JSON with keys "train" and "test";
#    train entries have input+output, test entries have input (+output in the
#    public evaluation set, absent in the hidden test set).
# ⚠️ The dataset root varies per competition snapshot — discovered below.
# ⚠️ 2026-10-24 Kaggle live-run fix: a *solutions* file
#    ({task_id: [expected test outputs]}) used to be misread as a multi-task
#    file because the old `all(... if isinstance(v, dict))` filter skipped the
#    list-valued entries and an empty generator makes all() True. Classify
#    first, index second — never assume a layout.
import os, json

def classify(d):
    # single / multi / solutions / list / other — see cell 10 for the loader
    if isinstance(d, dict):
        if "train" in d and "test" in d:
            return "single"
        if d and all(isinstance(v, dict) and "train" in v and "test" in v
                     for v in d.values()):
            return "multi"
        if d and all(isinstance(v, list) for v in d.values()):
            return "solutions"
        return "other"
    if isinstance(d, list):
        if d and all(isinstance(x, dict) and "train" in x and "test" in x
                     for x in d):
            return "list"
        return "other"
    return "other"

kinds = {}
if os.path.isdir("/kaggle/input"):
    for dirpath, _dirs, files in os.walk("/kaggle/input"):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(p) > 40 * 1024 * 1024:
                    continue
                with open(p, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
            except Exception:
                continue
            kinds.setdefault(classify(d), []).append(p)

for kind in sorted(kinds):
    print("== %-9s %d file(s)" % (kind, len(kinds[kind])))
    for p in kinds[kind][:10]:
        size = os.path.getsize(p)
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            print("   %s (%d B, unreadable)" % (p, size))
            continue
        if kind == "single":
            print("   %s (%d B) train=%d test=%d test[0] keys=%s"
                  % (p, size, len(d["train"]), len(d["test"]),
                     sorted(d["test"][0].keys())))
        elif kind == "multi":
            k = sorted(d)[0]
            print("   %s (%d B) tasks=%d first=%s train=%d test=%d"
                  % (p, size, len(d), k,
                     len(d[k]["train"]), len(d[k]["test"])))
        elif kind == "solutions":
            print("   %s (%d B) expected outputs for %d tasks (NOT a task "
                  "file — cell 10 uses it only to score locally)"
                  % (p, size, len(d)))
        elif kind == "list":
            print("   %s (%d B) %d tasks in a list (ids assigned by position)"
                  % (p, size, len(d)))
        else:
            print("   %s (%d B)" % (p, size))
print("DISCOVERY DONE — cell 10 loads single/multi/list files only")

# %%writefile does not create intermediate directories, and Kaggle's
# /kaggle/working starts empty — create the package dir once, here.
# (2026-10-24: the first writefile cell died with FileNotFoundError without
# this; locally the directory already existed, so CI never caught it.)
os.makedirs("/kaggle/working/arcagi2", exist_ok=True)"""))

    cells.append(md_cell("""## Cells 2-8 — materialize the embedded core (offline CI-green modules)

Each cell writes one module of the `arcagi2` package into the notebook working
directory. `python eval.py` on the same code passes locally (also under
`LC_ALL=C` without `PYTHONUTF8`), and the real-data gate keeps the honest
score on the official public evaluation set in the loop."""))

    for fn in MODULES:
        cells.append(writefile_cell(fn, read(os.path.join("arcagi2", fn))))

    cells.append(code_cell("""# __init__.py makes the written directory an importable package
%%writefile /kaggle/working/arcagi2/__init__.py
__version__ = "0.1.0"
"""))

    cells.append(code_cell("""# CELL 9 — SELF-CHECK: the embedded core must construct and solve before
# the real data is touched. If this raises, STOP: the embedding broke.
import sys
sys.path.insert(0, "/kaggle/working")

from arcagi2 import grids, dsl, solver

# 1. every family applies without raising, on a small grid
g = grids.from_rows([[0, 1, 1], [0, 1, 0], [0, 0, 0]])
for t in dsl.enumerated_candidates():
    t.apply(g)

# 2. a synthetic task with a known one-transform answer is solved exactly
task = {"train": [{"input": [[0, 1, 1], [0, 1, 0], [0, 0, 0]],
                   "output": [[0, 0, 0], [0, 1, 1], [0, 1, 0]]},
                  {"input": [[0, 2, 2], [0, 2, 0], [0, 0, 0]],
                   "output": [[0, 0, 0], [0, 2, 2], [0, 2, 0]]}],
        "test": [{"input": [[0, 3, 3], [0, 3, 0], [0, 0, 0]]}]}
res = solver.solve_task(task, budget_s=10.0, max_depth=2)
assert res["solved"], res
assert res["predictions"][0]["attempt_1"] == grids.from_rows(
    [[0, 0, 0], [0, 3, 3], [0, 3, 0]]), res["predictions"][0]
assert isinstance(res["predictions"][0]["attempt_2"], tuple)
print("mini smoke OK — %d candidates, solved: %s" % (
    len(dsl.enumerated_candidates()), res["program"]))"""))

    cells.append(code_cell("""# CELL 10 — LOAD, SOLVE, WRITE submission.json
# ✅ Verified (2026-10-18): submission.json = {task_id: [{attempt_1, attempt_2},
#    ...]} with plain lists; a test output scores when EITHER attempt matches
#    exactly. ✅ <= 12h CPU/GPU, no internet — the adaptive budget below keeps
#    the whole run inside 9.5h even on a large hidden set.
import sys, os, json, time, glob
sys.path.insert(0, "/kaggle/working")

from arcagi2 import taskio

TOTAL_BUDGET_S = 9.5 * 3600          # ✅ well inside the 12h cap
PER_TASK_CAP_S = 120.0               # depth-1 programs are found in <1s; the
                                     # cap only bounds the unsolved majority
MAX_DEPTH = 2                        # depth 1 + curated depth-2 compositions

def discover_tasks():
    '''Every task JSON under /kaggle/input.

    Layouts handled (see cell 1 output):
      single  {"train": [...], "test": [...]}           -> id from filename
      multi   {task_id: {"train": [...], "test": ...}}  -> ids from keys
      list    [ {"train": ..., "test": ...}, ... ]      -> ids from position
    'solutions' files ({task_id: [expected outputs]}) are NOT task files --
    their values are lists, not task dicts, so no branch ever matches them.
    '''
    tasks = {}
    for p in glob.glob("/kaggle/input/**/*.json", recursive=True):
        try:
            if os.path.getsize(p) > 40 * 1024 * 1024:
                continue
            with open(p, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        base = os.path.splitext(os.path.basename(p))[0]
        if isinstance(d, dict) and "train" in d and "test" in d:
            tasks[base] = d
        elif isinstance(d, dict) and d and all(
                isinstance(v, dict) and "train" in v and "test" in v
                for v in d.values()):
            for tid, t in d.items():
                tasks[tid] = t
        elif isinstance(d, list) and d and all(
                isinstance(x, dict) and "train" in x and "test" in x
                for x in d):
            for i, t in enumerate(d):
                tasks["%s#%d" % (base, i)] = t
    return tasks

dataset = discover_tasks()
# Bonus (2026-10-24): if the snapshot ships a solutions file
# ({task_id: [expected test outputs]}), attach the expected outputs to the
# matching test items so the local score below covers every task instead of
# only the public split. Mismatched ids or lengths are skipped, never guessed.
attached = 0
for p in glob.glob("/kaggle/input/**/*.json", recursive=True):
    try:
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        continue
    if not (isinstance(d, dict) and d
            and all(isinstance(v, list) for v in d.values())):
        continue
    for tid, outs in d.items():
        t = dataset.get(tid)
        if not t or not isinstance(outs, list) or len(outs) != len(t["test"]):
            continue
        for e, o in zip(t["test"], outs):
            if "output" not in e:
                e["output"] = o
                attached += 1
print("attached expected outputs:", attached)
print("tasks discovered:", len(dataset))
assert dataset, "no tasks found — check cell 1 discovery output"

_tasks = max(1, len(dataset))
# Hard guarantee: per_task * _tasks <= TOTAL_BUDGET_S, so the whole run stays
# inside 9.5h and therefore inside Kaggle's 12h cap even on a huge hidden set.
# The previous `max(5.0, ...)` floor could push the wall clock past 12h once
# the hidden set exceeded ~6800 tasks, which would score the run as a DNF.
per_task = min(PER_TASK_CAP_S, TOTAL_BUDGET_S / _tasks)
assert per_task * _tasks <= TOTAL_BUDGET_S + 1e-6, "budget exceeds the 9.5h cap"
print("per-task budget: %.1fs, depth: %d, tasks: %d, worst-case total: %.1fh"
      % (per_task, MAX_DEPTH, _tasks, per_task * _tasks / 3600.0))

t0 = time.time()
state = {"n": 0, "solved": 0}

def progress(i, total, tid, res):
    state["n"] = i
    state["solved"] += 1 if res.get("solved") else 0
    if i % 10 == 0 or i == total:
        print("  %d/%d done, %d solved, %.0fs elapsed"
              % (i, total, state["solved"], time.time() - t0))

results = taskio.solve_dataset(dataset, budget_s=per_task,
                               max_depth=MAX_DEPTH, progress=progress)

# score the slice we CAN score locally (public set ships test outputs)
scored = {tid: t for tid, t in dataset.items()
          if all("output" in e for e in t["test"])}
if scored:
    sc = taskio.score_dataset(scored, results)
    print("local score (public test outputs): %d/%d either-attempt correct, "
          "%d programs found" % (sc["either_correct"], sc["test_outputs"],
                                 sc["program_found_tasks"]))

sub = taskio.build_submission(dataset, results)
problems = taskio.validate_submission(sub, dataset)
assert not problems, problems[:5]
out_path = "/kaggle/working/submission.json"
taskio.write_submission(sub, out_path)
print("wrote", out_path, "tasks:", len(sub),
      "elapsed: %.0fs" % (time.time() - t0))"""))

    cells.append(md_cell("""## Submission checklist

1. Cell 1 discovery read; the loader found every task (assert passed).
2. Cell 9 self-check printed `mini smoke OK`.
3. Cell 10 wrote `submission.json` with zero validation problems.
4. Notebook set to **public** (official: prize eligibility requires open
   source) and committed identical to
   https://github.com/Zhiyilang/arcagi2 (claim-audit rule).
5. Kaggle "Submit" -> record the **submission ID**; the Paper Track writeup
   must reference it (see `ARC_PRIZE_ACTION_CARD.md`).

Deadlines (2026-10-18 official, all 11:59 PM UTC): entry 10-26 · final
submission 11-02 · Paper Track writeup 11-09 · winners 12-04."""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out = os.path.join(HERE, "submission.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote", out, "cells:", len(cells))


if __name__ == "__main__":
    build()
