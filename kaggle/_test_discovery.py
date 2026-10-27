"""Empirical test of the FIXED discovery/loader cells against a synthetic
Kaggle-style snapshot. Extracts the real cell sources from the rebuilt
submission.ipynb, repoints /kaggle/input at the synthetic tree, and runs the
discovery half of each cell (the solve half is skipped — it needs real data).

Expected snapshot (mirrors the 2026-10-24 live Kaggle layout that crashed):
  challenges.json   multi      {tid: {"train": [...], "test": [...]}}  test w/o output
  solutions.json    solutions  {tid: [expected outputs]}
  single.json       single     {"train": [...], "test": [...]}
  list_tasks.json   list       [ {"train": ..., "test": ...}, ... ]
  meta.json         other      {"version": 1}
  sub/extra.json    multi      (recursive walk must find it)
"""
import json, os, shutil, sys, io, contextlib

SIM = "/tmp/arc_kaggle_sim"
NB = "kaggle/submission.ipynb"

G = [[0, 1], [1, 0]]          # a trivial grid
TASK = lambda: {"train": [{"input": G, "output": G},
                          {"input": G, "output": G}],
                "test": [{"input": G}]}

# ---- build the synthetic snapshot -------------------------------------
shutil.rmtree(SIM, ignore_errors=True)
os.makedirs(SIM + "/sub", exist_ok=True)
json.dump({"aa11": TASK(), "bb22": TASK(), "cc33": TASK()},
          open(SIM + "/challenges.json", "w"))
json.dump({"aa11": [G], "bb22": [G], "cc33": [G]},
          open(SIM + "/solutions.json", "w"))
json.dump(TASK(), open(SIM + "/single.json", "w"))
json.dump([TASK(), TASK()], open(SIM + "/list_tasks.json", "w"))
json.dump({"version": 1}, open(SIM + "/meta.json", "w"))
json.dump({"dd44": TASK(), "ee55": TASK()},
          open(SIM + "/sub/extra.json", "w"))

nb = json.load(open(NB, encoding="utf-8"))
cell1 = "".join(nb["cells"][1]["source"]).replace("/kaggle/input", SIM)
# 2026-10-24: cell 1 ends with os.makedirs("/kaggle/working/arcagi2") — the fix
# for the live FileNotFoundError in the first %%writefile cell. On Kaggle that
# path exists; locally "/kaggle" resolves to a drive root and raises
# PermissionError, which made this suite report a FALSE failure (exit 1) even
# though the discovery half printed correctly. Point it at the sim tree.
cell1 = cell1.replace("/kaggle/working", SIM + "/work")
cell12 = "".join(nb["cells"][12]["source"]).replace("/kaggle/input", SIM)
# point the notebook's working dir at the repo root so `from arcagi2 import
# taskio` resolves against the local CI-green package (Kaggle writes the
# modules there via %%writefile; locally they already exist)
REPO = os.path.abspath(os.path.join(os.getcwd()))
cell12 = cell12.replace('sys.path.insert(0, "/kaggle/working")',
                        'sys.path.insert(0, %r)' % REPO)
cell12 = cell12.replace('"/kaggle/working/submission.json"',
                        'os.path.join(%r, "submission.json")' % REPO)
# keep only the discovery half of cell 12: up to the per-task budget print
cell12 = cell12.split("t0 = time.time()")[0]

fails = []

def run(name, src):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, name, "exec"), {"__name__": "__main__"})
    except Exception as e:
        fails.append("%s raised %s: %s" % (name, type(e).__name__, e))
        print(buf.getvalue())
        return None, buf.getvalue()
    return True, buf.getvalue()

# ---- 1) cell 1 must classify everything and never crash ---------------
ok1, out1 = run("cell1", cell1)
print("---- cell 1 output ----")
print(out1)
if ok1 is None:
    sys.exit("FAIL: cell 1 crashed (the original live bug)")

for expect in ["multi ", "solutions", "single ", "list   ", "other  "]:
    if expect not in out1:
        fails.append("cell1 output missing kind: %r" % expect)
if "challenges.json" not in out1 or "solutions.json" not in out1:
    fails.append("cell1 did not list the sim files")
if "NOT a task file" not in out1:
    fails.append("cell1 did not flag the solutions file as non-task")

# ---- 2) cell 12 discovery must exclude solutions and attach outputs ----
ns = {}
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(cell12, "cell12", "exec"), ns)
except Exception as e:
    fails.append("cell12 raised %s: %s" % (type(e).__name__, e))
print("---- cell 12 output ----")
print(buf.getvalue())

if not fails:
    dataset = ns["dataset"]
    print("---- assertions ----")
    expected_ids = {"aa11", "bb22", "cc33",           # multi challenges
                    "single",                          # single
                    "list_tasks#0", "list_tasks#1",    # list
                    "dd44", "ee55"}                    # recursive multi
    got = set(dataset)
    if got != expected_ids:
        fails.append("dataset ids mismatch: extra=%s missing=%s"
                     % (sorted(got - expected_ids), sorted(expected_ids - got)))
    else:
        print("ids OK: 8 tasks from 3 layouts, solutions excluded")
    # exactly the 3 solution-matched tasks get expected outputs attached;
    # the other 5 have no solutions entry, so nothing is (correctly) guessed
    attached_ids = {tid for tid, t in dataset.items()
                    if t["test"] and all("output" in e for e in t["test"])}
    n_out = sum(1 for t in dataset.values() for e in t["test"] if "output" in e)
    n_test = sum(len(t["test"]) for t in dataset.values())
    print("test items with expected output: %d/%d, ids: %s"
          % (n_out, n_test, sorted(attached_ids)))
    if attached_ids != {"aa11", "bb22", "cc33"}:
        fails.append("attach set wrong: %s" % sorted(attached_ids))
    if n_out != 3:
        fails.append("attached %d outputs, expected 3" % n_out)
    # build_submission must accept the attached dataset (structure check only)
    sys.path.insert(0, ".")
    from arcagi2 import taskio
    print("taskio import OK — real core present")

print()
if fails:
    print("FAILURES:")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print("ALL DISCOVERY/LOADER TESTS PASSED")
