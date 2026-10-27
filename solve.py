#!/usr/bin/env python3
"""Solve an ARC-AGI-2 dataset directory and write submission.json.

Usage:
    python solve.py --data data/evaluation --out submission.json
    python solve.py --data data/evaluation --limit 12 --budget 4   # smoke

Zero dependencies. Progress and the final score report go to stderr; the
submission JSON is the only thing written to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from arcagi2 import taskio


def main(argv=None):
    ap = argparse.ArgumentParser(description="ARC-AGI-2 program-synthesis solver")
    ap.add_argument("--data", required=True,
                    help="directory of task JSON files")
    ap.add_argument("--out", default="submission.json",
                    help="output submission.json path")
    ap.add_argument("--budget", type=float, default=8.0,
                    help="per-task search budget in seconds")
    ap.add_argument("--depth", type=int, default=3,
                    help="maximum program length to search")
    ap.add_argument("--limit", type=int, default=0,
                    help="only solve the first N tasks (0 = all)")
    ap.add_argument("--report", default="",
                    help="optional path for the JSON score report")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    dataset = taskio.load_dataset(args.data)
    if args.limit:
        keys = sorted(dataset)[:args.limit]
        dataset = {k: dataset[k] for k in keys}
    if not dataset:
        print("no tasks found in %s" % args.data, file=sys.stderr)
        return 2

    started = time.monotonic()
    last = [0]

    def progress(done, total, tid, result):
        if args.quiet:
            return
        if done == total or done - last[0] >= max(1, total // 40):
            last[0] = done
            el = time.monotonic() - started
            print("  [%d/%d] %s %s (%.0fs)"
                  % (done, total, tid,
                     "solved" if result["solved"] else "-", el),
                  file=sys.stderr)

    if not args.quiet:
        print("solving %d tasks (budget %.1fs/task, depth<=%d)..."
              % (len(dataset), args.budget, args.depth), file=sys.stderr)
    results = taskio.solve_dataset(dataset, budget_s=args.budget,
                                   max_depth=args.depth, progress=progress)
    elapsed = time.monotonic() - started

    report = taskio.score_dataset(dataset, results)
    report["elapsed_s"] = round(elapsed, 2)
    report["budget_s"] = args.budget
    report["max_depth"] = args.depth

    sub = taskio.build_submission(dataset, results)
    problems = taskio.validate_submission(sub, dataset)
    if problems:
        print("SUBMISSION INVALID:", file=sys.stderr)
        for p in problems[:10]:
            print("  " + p, file=sys.stderr)
        return 1

    taskio.write_submission(sub, args.out)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    print("=" * 60, file=sys.stderr)
    print("tasks              : %d" % report["tasks"], file=sys.stderr)
    print("test outputs       : %d" % report["test_outputs"], file=sys.stderr)
    print("either-attempt acc : %d/%d = %.2f%%"
          % (report["either_correct"], report["test_outputs"],
             100.0 * report["accuracy"]), file=sys.stderr)
    print("attempt-1-only acc : %d/%d = %.2f%%"
          % (report["attempt1_correct"], report["test_outputs"],
             100.0 * report["attempt1_accuracy"]), file=sys.stderr)
    print("program found      : %d/%d tasks"
          % (report["program_found_tasks"], report["tasks"]), file=sys.stderr)
    print("elapsed            : %.1fs (%.2fs/task)"
          % (elapsed, elapsed / max(1, len(dataset))), file=sys.stderr)
    print("submission         : %s" % args.out, file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
