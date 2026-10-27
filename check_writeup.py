"""ARC Prize 2026 — ARC-AGI-2 Solution Writeup gate.

Official constraints (arcprize.org/competitions/2026/arc-agi-2 and the Kaggle
rules page, verified 2026-10-19):

  * The Grand Prize ($275,000) / Innovation Prize goes to the highest scoring
    Solution Writeup, judged equally on six criteria: Accuracy, Universality,
    Progress, Theory, Completeness, Novelty (each 0-5, averaged).
  * "All artifacts should be open sourced and attached to an official
    competition Solution Writeup within seven days of the competition's
    submission deadline."
  * "What to Include" (arcprize.org): Abstract, Intro, Prior work, Approach,
    Results, Conclusion.
  * No word limit is published for the track writeup. The Paper Track limit is
    1,500 words, so this gate enforces the same budget as a conservative
    upper bound — a tight writeup cannot be penalised for length.

What this gate enforces, so a violation cannot ship by accident:

  1. body word count <= 1500 (blockquote ">" lines are local editor notes and
     are NOT part of the submitted text)
  2. every official "What to Include" section present
  3. all six judging criteria are explicitly addressed (a judge scoring six
     dimensions should not have to infer which paragraph maps to which)
  4. every measured number claimed in the writeup is present verbatim, so an
     edit that quietly changes a result fails here
  5. no unfilled placeholders at submit time

Measured numbers are additionally pinned as assertions by `eval.py` (163/163),
which runs `ablations.py`'s rule set; this gate pins the *prose* against them.

Usage:
    python check_writeup.py                 # gate (fails on any violation)
    python check_writeup.py --allow-draft   # placeholders tolerated
"""

from __future__ import annotations

import argparse
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "SOLUTION_WRITEUP.md")
WORD_LIMIT = 1500

# Official "What to Include" headings -> regexes that accept our section names.
REQUIRED_SECTIONS = [
    ("Abstract", r"(?i)^\s*#+\s*.*\babstract\b"),
    ("Intro", r"(?i)^\s*#+\s*.*(intro|introduction)"),
    ("Prior work", r"(?i)^\s*#+\s*.*(prior work|related work|background)"),
    ("Approach", r"(?i)^\s*#+\s*.*(approach|method)"),
    ("Results", r"(?i)^\s*#+\s*.*\bresults?\b"),
    ("Conclusion", r"(?i)^\s*#+\s*.*(conclusion|summary)"),
]

# The six judging criteria must be named explicitly in the text.
REQUIRED_CRITERIA = [
    "Accuracy",
    "Universality",
    "Progress",
    "Theory",
    "Completeness",
    "Novelty",
]

# Measured values that the prose is not allowed to drift away from.
# Each entry: label -> regex accepting the spaced form used in the writeup.
REQUIRED_NUMBERS = [
    ("tasks=120 / test outputs=167", r"120\s+tasks.{0,40}167\s+test\s+outputs"),
    ("DSL consistent program 1/120", r"1\s*/\s*120"),
    ("test outputs correct 0/167", r"0\s*/\s*167"),
    ("assertions 163/163", r"163\s*/\s*163"),
    ("trivial families 0/120", r"0\s*/\s*120"),
    ("band motif 48/120", r"48\s*/\s*120"),
    ("band_repair test 0/1", r"0\s*/\s*1\b"),
    ("component-count statistic 109/120", r"109\s+of\s+120"),
]

PLACEHOLDER = re.compile(r"\[[A-Z_0-9]{3,}\]")


def body_text(md: str) -> str:
    """Submitted text = everything except local editor notes (blockquote lines)."""
    return "\n".join(l for l in md.splitlines() if not l.lstrip().startswith(">"))


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-draft", action="store_true",
                    help="tolerate unfilled [RESULT]-style placeholders")
    ap.add_argument("--file", default=TARGET)
    args = ap.parse_args()

    md = open(args.file, encoding="utf-8").read()
    body = body_text(md)
    n = words(body)
    full = words(md)

    problems = []
    if n > WORD_LIMIT:
        problems.append(
            f"body word count {n} > limit {WORD_LIMIT} (over by {n - WORD_LIMIT})")

    for name, rx in REQUIRED_SECTIONS:
        if not re.search(rx, body, re.M):
            problems.append(f"missing required section: {name}")

    missing_criteria = [c for c in REQUIRED_CRITERIA
                        if not re.search(r"\b" + c + r"\b", body)]
    if missing_criteria:
        problems.append("judging criteria not explicitly addressed: "
                        + ", ".join(missing_criteria))

    missing_numbers = [lab for lab, rx in REQUIRED_NUMBERS
                       if not re.search(rx, body)]
    if missing_numbers:
        problems.append("measured numbers missing/drifting: "
                        + "; ".join(missing_numbers))

    ph = sorted(set(PLACEHOLDER.findall(body)))
    if ph and not args.allow_draft:
        problems.append("unfilled placeholders present: " + ", ".join(ph))

    print(f"file              : {os.path.relpath(args.file, HERE)}")
    print(f"full word count   : {full}  (includes local notes)")
    print(f"body word count   : {n} / {WORD_LIMIT}  "
          f"({WORD_LIMIT - n} words of headroom)")
    print(f"required sections : "
          f"{sum(1 for _, rx in REQUIRED_SECTIONS if re.search(rx, body, re.M))}"
          f"/{len(REQUIRED_SECTIONS)}")
    print(f"judging criteria  : "
          f"{len(REQUIRED_CRITERIA) - len(missing_criteria)}"
          f"/{len(REQUIRED_CRITERIA)}")
    print(f"measured numbers  : "
          f"{len(REQUIRED_NUMBERS) - len(missing_numbers)}"
          f"/{len(REQUIRED_NUMBERS)}")
    print(f"placeholders      : {ph if ph else 'none'}")

    if problems:
        print("\n--- FAIL ---")
        for p in problems:
            print("  " + p)
        return 1
    print("\nRESULT: OK — writeup is submittable "
          "(modulo the Kaggle writeup attachment itself)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
