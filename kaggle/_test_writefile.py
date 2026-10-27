"""Regression proof for the 2026-10-24 writefile FileNotFoundError.

Simulates the Kaggle run order in a FRESH EMPTY directory:
  cell 1 (discovery + makedirs) -> cell 3 (first %%writefile)
IPython's %%writefile is exactly `io.open(path, "w").write(body)` with no
directory creation, so if the makedirs line is missing this reproduces the
live FileNotFoundError; with it, the file lands and compiles.
"""
import io, json, os, shutil, sys, tempfile

NB = "kaggle/submission.ipynb"
WORK = os.path.join(tempfile.gettempdir(), "kaggle_work_sim")
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)          # start empty, like Kaggle's /kaggle/working

nb = json.load(open(NB, encoding="utf-8"))
cell1 = "".join(nb["cells"][1]["source"]).replace("/kaggle/working", WORK)
cell3 = "".join(nb["cells"][3]["source"])   # first %%writefile cell (grids.py)

# --- run cell 1 (it must create the package dir) -------------------------
g1 = {"__name__": "__main__"}
exec(compile(cell1, "cell1", "exec"), g1)
pkg = os.path.join(WORK, "arcagi2")
assert os.path.isdir(pkg), "cell 1 did not create the package dir"
print("cell 1 OK: created", pkg)

# --- run cell 3 with real %%writefile semantics --------------------------
line, _, body = cell3.partition("\n")
assert line.startswith("%%writefile "), line[:40]
target = line.split()[1].replace("/kaggle/working", WORK)
try:
    with io.open(target, "w", encoding="utf-8") as fh:
        fh.write(body)
except FileNotFoundError as e:
    print("FAIL: %%writefile still dies:", e)
    sys.exit(1)
compile(io.open(target, encoding="utf-8").read(), "grids.py", "exec")
print("cell 3 OK: wrote + compiled", target,
      "(%d bytes)" % os.path.getsize(target))

# --- negative control: WITHOUT makedirs the original bug must reappear ----
WORK2 = os.path.join(tempfile.gettempdir(), "kaggle_work_sim2")
shutil.rmtree(WORK2, ignore_errors=True)
os.makedirs(WORK2)
cell1_nomk = "\n".join(ln for ln in cell1.splitlines()
                       if "makedirs" not in ln)
exec(compile(cell1_nomk, "cell1_nomk", "exec"),
     {"__name__": "__main__"})
target2 = line.split()[1].replace("/kaggle/working", WORK2)
try:
    with io.open(target2, "w", encoding="utf-8") as fh:
        fh.write(body)
    print("UNEXPECTED: negative control passed (bug not reproduced)")
    sys.exit(1)
except FileNotFoundError:
    print("negative control OK: without makedirs the FileNotFoundError "
          "reproduces exactly as on Kaggle")

shutil.rmtree(WORK, ignore_errors=True)
shutil.rmtree(WORK2, ignore_errors=True)
print("WRITEFILE REGRESSION TEST PASSED")
