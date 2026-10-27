"""Pattern-completion transforms: repair cells that break local regularity.



ARC-AGI-2 contains tasks whose output is the input with a few anomalous

cells restored to the local pattern. These transforms derive the expected

value from local evidence only (periodicity along a row/column, or the

neighborhood majority) and change a cell only when that evidence is

unambiguous. Whether the repair is correct is decided by exact consistency

with the demonstrations, never by the transform itself.



Period detection is the subtle part. A periodic interior framed by constant

borders (a very common ARC layout) defeats naive detectors: violation

counts pick the frame's period, and smallest-agreeing-period picks up

coincidental distance-1 agreement inside alternating runs. The detector

used here scores each candidate period by how many cells agree with their

residue-class majority, measured on the line with its first and last cell

dropped (frames live at the edges), and breaks ties toward the smaller

period. A cell is then repaired only when the evidence is unambiguous —

both same-period neighbours agree, or the cell is the lone dissenter in a

strictly-majority class with same-period neighbours on both sides.

"""



from collections import Counter



from . import grids

from .grids import Grid, shape, cell





def _class_majority_total(line, p):

    """Cells agreeing with their residue-class majority for period p."""

    total = 0

    for r in range(p):

        vals = [line[c] for c in range(r, len(line), p)]

        if not vals:

            continue

        counts = Counter(vals)

        total += max(counts.values())

    return total





def _dominant_period(line, max_period=4):

    """The line's period: most class-majority agreement, smallest on ties.



    The first and last cells are dropped before scoring because frames and

    borders live at line edges and would otherwise drag the estimate toward

    the frame's period.

    """

    core = line[1:-1] if len(line) > 2 else line

    n = len(core)

    best_p, best_s = 1, -1

    for p in range(1, min(max_period, n - 1) + 1):

        s = _class_majority_total(core, p)

        if s > best_s:

            best_p, best_s = p, s

    return best_p





def _repair_line(line, max_period=4):

    """Repair cells that contradict the line's dominant period.



    Period p is detected on the line's core (first/last cell dropped) by

    residue-class majority agreement. Two unambiguous-evidence rules then

    run on the full line, both gated so a frame cell can never be touched:



    1. both-neighbours: the cells at distance p on either side exist and

       agree with each other but not with the cell, and that agreed value

       is not contradicted by a strict class majority -> restore it.

    2. lone-dissenter: the cell sits at least two cells in from each line

       edge (frame territory is closer), is the only occurrence of its

       value inside its residue class, and the class has a strict majority

       -> take the majority.



    Anything ambiguous (conflicting evidence, non-strict classes, cells

    hugging the line edges) is left untouched. Over-repairing a frame cell

    would make the whole transform inconsistent with the demonstrations

    and get it rejected; under-repairing only costs that one task.

    """

    p = _dominant_period(line, max_period)

    n = len(line)

    core = line[1:-1] if n > 2 else line

    # class statistics come from the core (frames live at the edges), but a

    # core of 1-2 cells carries no class signal at all -> fall back to the

    # full line for short lines

    src, off = (core, 1) if len(core) >= 3 else (line, 0)

    out = list(line)

    for c in range(n):

        cur = line[c]

        ci = c - off

        cls = [src[i] for i in range(ci % p, len(src), p)] if src else []

        counts = Counter(cls)

        majority, strict = None, False

        if counts:

            top_n = max(counts.values())

            if top_n * 2 > len(cls):

                majority = max(counts.items(),

                               key=lambda kv: (kv[1], -kv[0]))[0]

                strict = True

        a = line[c - p] if c - p >= 0 else None

        b = line[c + p] if c + p < n else None

        if (a is not None and b is not None and a == b and a != cur

                and (not strict or a == majority)):

            out[c] = a

            continue

        if (strict and majority != cur and counts.get(cur, 0) == 1

                and 2 <= c <= n - 3):

            out[c] = majority

    return out





def fix_period(g, axis="both", max_period=4):

    """Restore cells that contradict the local periodic pattern."""

    h, w = shape(g)

    rows = [list(r) for r in g]

    if axis in ("row", "both"):

        for r in range(h):

            rows[r] = _repair_line(rows[r], max_period)

    if axis in ("col", "both"):

        cols = [[rows[r][c] for r in range(h)] for c in range(w)]

        cols = [_repair_line(col, max_period) for col in cols]

        for c in range(w):

            for r in range(h):

                rows[r][c] = cols[c][r]

    return tuple(tuple(r) for r in rows)





def fix_majority(g):

    """Repaint cells that disagree with ALL eight neighbours.



    The neighborhood majority is the expectation; a cell is only repaired

    when it matches none of its neighbours (a genuine outlier), so

    structured regions and boundaries survive.

    """

    h, w = shape(g)

    out = [list(row) for row in g]

    for r in range(h):

        for c in range(w):

            v = g[r][c]

            neighbors = []

            for dr in (-1, 0, 1):

                for dc in (-1, 0, 1):

                    if dr == 0 and dc == 0:

                        continue

                    nv = cell(g, r + dr, c + dc)

                    if nv is not None:

                        neighbors.append(nv)

            if not neighbors or v in neighbors:

                continue

            counts = Counter(neighbors)

            best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

            out[r][c] = best

    return tuple(tuple(row) for row in out)





PATTERN_MODES = ("row", "col", "both", "majority")





def apply_pattern(g, mode):

    if mode == "majority":

        return fix_majority(g)

    return fix_period(g, axis=mode)