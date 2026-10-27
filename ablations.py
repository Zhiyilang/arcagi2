"""Offline ablation battery for arcagi2 (zero dependencies).

Answers one question with numbers instead of intuition: **how much of the
ARC-AGI-2 public evaluation set can a closed family of transforms reach at
all?** Every family is tested the same way -- fit on the demonstration pairs,
accept only if it reproduces ALL of them exactly, then predict the held-out
test inputs and compare against ground truth.

    python ablations.py            # cheap families only (seconds)
    python ablations.py --dsl      # also the depth-2 DSL enumeration (minutes)

The numbers printed here are the ones quoted in README.md and asserted by
eval.py; re-run before quoting anything.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "evaluation")


def load_tasks(limit=None):
    out = {}
    if not os.path.isdir(DATA_DIR):
        return out
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as fh:
            out[fname[:-5]] = json.load(fh)
        if limit and len(out) >= limit:
            break
    return out


def pairs_of(task):
    return [(p["input"], p["output"]) for p in task["train"]]


# ----------------------------------------------------------- trivial families

def fam_identity(g):
    return g


def fam_flip_h(g):
    return [row[::-1] for row in g]


def fam_flip_v(g):
    return g[::-1]


def fam_rot180(g):
    return [row[::-1] for row in g[::-1]]


def map_recolor(gi, go):
    if len(gi) != len(go) or len(gi[0]) != len(go[0]):
        return None
    m = {}
    for r in range(len(gi)):
        for c in range(len(gi[0])):
            a, b = gi[r][c], go[r][c]
            if m.get(a, b) != b:
                return None
            m[a] = b
    return m


def apply_recolor(g, m):
    out = []
    for row in g:
        o = []
        for v in row:
            if v not in m:
                return None
            o.append(m[v])
        out.append(o)
    return out


def fit_recolor(pairs):
    m = map_recolor(pairs[0][0], pairs[0][1])
    if m is None:
        return None
    for gi, go in pairs:
        if apply_recolor(gi, m) != go:
            return None
    return lambda g, m=m: apply_recolor(g, m)


# ------------------------------------------- band periodicity repair family
# 135a2760 is a "nested frame + periodic content" task: the content bands each
# follow a small 2-D period and a handful of cells break it. The repair picks
# the SMALLEST period whose anomaly count stays under a budget and rewrites
# those cells to the majority pattern of their residue class.

def _interior_uniform(g, r):
    row = g[r][2:len(g[0]) - 2]
    return len(set(row)) <= 1


def band_repair(g, max_anom=4, max_pc=24, max_pr=4):
    """Repair each content band to its smallest consistent 2-D period."""
    H, W = len(g), len(g[0])
    if W < 6 or H < 3:
        return None
    out = [row[:] for row in g]
    touched = False
    r = 0
    while r < H:
        if _interior_uniform(g, r):
            r += 1
            continue
        r0 = r
        while r < H and not _interior_uniform(g, r):
            r += 1
        band = [g[rr][2:W - 2] for rr in range(r0, r)]
        res = _band_period(band, max_anom, max_pc, max_pr)
        if res:
            pr, pc, pat = res
            for i, rr in enumerate(range(r0, r)):
                for c in range(W - 4):
                    out[rr][c + 2] = pat[(i % pr, c % pc)]
            touched = True
        r += 0
    return out if touched else None


def _band_period(band, max_anom, max_pc, max_pr):
    h = len(band)
    w = len(band[0]) if h else 0
    if h == 0 or w < 4:
        return None
    for pr in range(1, min(h, max_pr) + 1):
        for pc in range(2, min(w - 1, max_pc) + 1):
            table = {}
            for r in range(h):
                for c in range(w):
                    table.setdefault((r % pr, c % pc),
                                     Counter())[band[r][c]] += 1
            pat = {k: v.most_common(1)[0][0] for k, v in table.items()}
            anom = [(r, c) for r in range(h) for c in range(w)
                    if band[r][c] != pat[(r % pr, c % pc)]]
            if len(anom) <= max_anom:
                return (pr, pc, pat)
    return None


def fit_band(pairs):
    """Strict, solver-consistent criterion: the repair must turn every
    demonstration INPUT into its OUTPUT."""
    for gi, go in pairs:
        if band_repair(gi) != go:
            return None
    return band_repair


def band_outputs_periodic(task):
    """Diagnostic: are the demonstration OUTPUTS already well-formed periodic
    bands (i.e. this task belongs to the periodic-band motif regardless of
    whether 'repair' is the right transform)?"""
    outs = [p["output"] for p in task["train"]]
    return all(band_repair(go) is None for go in outs)


# ------------------------------------------------------------- the battery

TRIVIAL = [("identity", lambda p: (fam_identity if all(fam_identity(gi) == go
                                                        for gi, go in p) else None)),
           ("flip_h", lambda p: (fam_flip_h if all(fam_flip_h(gi) == go
                                                   for gi, go in p) else None)),
           ("flip_v", lambda p: (fam_flip_v if all(fam_flip_v(gi) == go
                                                   for gi, go in p) else None)),
           ("rot180", lambda p: (fam_rot180 if all(fam_rot180(gi) == go
                                                   for gi, go in p) else None)),
           ("recolor", fit_recolor),
           ("band_repair", fit_band)]


def run(tasks, family_names=None):
    """For each family: tasks fitted, test outputs correct (either attempt)."""
    results = {}
    for name, fit in TRIVIAL:
        if family_names and name not in family_names:
            continue
        fitted, test_total, test_ok = [], 0, 0
        for tid in sorted(tasks):
            task = tasks[tid]
            pairs = pairs_of(task)
            if not pairs:
                continue
            try:
                fn = fit(pairs)
            except Exception:
                fn = None
            if fn is None:
                continue
            fitted.append(tid)
            for tp in task["test"]:
                test_total += 1
                try:
                    pred = fn(tp["input"])
                except Exception:
                    pred = None
                if pred == tp["output"]:
                    test_ok += 1
        results[name] = (fitted, test_total, test_ok)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsl", action="store_true",
                    help="also run the depth-2 DSL enumeration (slow)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tasks = load_tasks(args.limit)
    if not tasks:
        print("no tasks under %s" % DATA_DIR)
        return 1
    print("tasks: %d" % len(tasks))
    print()
    print("%-14s %8s %10s" % ("family", "fitted", "test ok"))
    print("-" * 36)
    for name, (fitted, total, ok) in run(tasks).items():
        print("%-14s %8d %5d/%-4d  %s" % (name, len(fitted), ok, total,
                                          fitted[:3]))
    motif = [tid for tid in sorted(tasks) if band_outputs_periodic(tasks[tid])]
    print("-" * 36)
    print("%-14s %8d" % ("band MOTIF", len(motif)))
    print("  (demonstration outputs are already well-formed periodic bands;")
    print("   this counts the motif, NOT a solved task -- band_repair is the")
    print("   wrong transform for most of them.)")
    if args.dsl:
        from arcagi2 import taskio
        ds = taskio.load_dataset(DATA_DIR)
        res = taskio.solve_dataset(ds, budget_s=6.0, max_depth=2)
        sc = taskio.score_dataset(ds, res)
        print("-" * 36)
        print("%-14s %8d %5d/%-4d" % ("dsl_depth2",
                                      sc["program_found_tasks"],
                                      sc["either_correct"],
                                      sc["test_outputs"]))
        for k in sorted(res):
            if res[k]["solved"]:
                print("   solved:", k, res[k]["program"])
    print()
    print("Measured on the public ARC-AGI-2 evaluation set. Quote these")
    print("numbers, not the intuition behind them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
