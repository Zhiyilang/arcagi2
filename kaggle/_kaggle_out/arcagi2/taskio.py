"""Task I/O, scoring, and submission serialization for ARC-AGI-2.

Scoring follows the official rule verbatim: a task output scores 1 when
EITHER attempt matches the ground truth exactly; the final score is the
average over all task test outputs.
"""

import glob
import json
import os

from . import grids
from . import solver as solvermod


def load_task(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_dataset(directory):
    """task_id -> task dict, for every *.json in the directory."""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        tid = os.path.splitext(os.path.basename(path))[0]
        out[tid] = load_task(path)
    return out


def solve_dataset(dataset, budget_s=solvermod.DEFAULT_BUDGET_S,
                  max_depth=solvermod.DEFAULT_MAX_DEPTH, progress=None):
    """Solve every task. Returns {task_id: solve_task result}."""
    results = {}
    for i, (tid, task) in enumerate(sorted(dataset.items())):
        results[tid] = solvermod.solve_task(task, budget_s=budget_s,
                                            max_depth=max_depth)
        if progress:
            progress(i + 1, len(dataset), tid, results[tid])
    return results


def score_task(task, result):
    """(n_correct, n_total, attempt1_correct, either_correct) for one task."""
    total = 0
    either = 0
    only1 = 0
    for i, pred in enumerate(result["predictions"]):
        gt = grids.from_rows(task["test"][i]["output"])
        total += 1
        a1 = pred["attempt_1"] == gt
        a2 = pred["attempt_2"] == gt
        if a1 or a2:
            either += 1
        if a1:
            only1 += 1
    return total, either, only1, either


def score_dataset(dataset, results):
    """Aggregate the official either-attempt accuracy over a dataset."""
    total = 0
    either = 0
    only1 = 0
    solved_tasks = 0
    for tid, task in dataset.items():
        res = results[tid]
        t, e, o, _ = score_task(task, res)
        total += t
        either += e
        only1 += o
        if res.get("solved"):
            solved_tasks += 1
    return {
        "tasks": len(dataset),
        "test_outputs": total,
        "either_correct": either,
        "attempt1_correct": only1,
        "accuracy": (either / total) if total else 0.0,
        "attempt1_accuracy": (only1 / total) if total else 0.0,
        "program_found_tasks": solved_tasks,
    }


def build_submission(dataset, results):
    """The Kaggle submission.json structure."""
    sub = {}
    for tid, task in sorted(dataset.items()):
        entries = []
        for pred in results[tid]["predictions"]:
            entries.append({
                "attempt_1": grids.to_rows(pred["attempt_1"]),
                "attempt_2": grids.to_rows(pred["attempt_2"]),
            })
        sub[tid] = entries
    return sub


def write_submission(sub, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sub, fh)


def validate_submission(sub, dataset):
    """Structural gate: every task id present, every attempt a grid."""
    problems = []
    for tid, task in dataset.items():
        if tid not in sub:
            problems.append("missing task %s" % tid)
            continue
        entries = sub[tid]
        if len(entries) != len(task["test"]):
            problems.append("task %s: %d entries for %d test inputs"
                            % (tid, len(entries), len(task["test"])))
            continue
        for i, entry in enumerate(entries):
            for key in ("attempt_1", "attempt_2"):
                if key not in entry:
                    problems.append("task %s[%d]: missing %s" % (tid, i, key))
                    continue
                g = entry[key]
                if not g or not all(isinstance(r, list) for r in g):
                    problems.append("task %s[%d].%s: not a grid" % (tid, i, key))
    return problems
