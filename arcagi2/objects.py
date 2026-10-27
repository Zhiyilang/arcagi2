"""Connected-component objects for the ARC-AGI-2 solver.

Objects are extracted from a grid as maximal same-color connected regions
(4- or 8-connectivity). The object layer is what lets the solver express
"move the red square onto the blue line" without hand-coded coordinates.
"""

from collections import deque

from . import grids
from .grids import Grid, shape, cell, blank, height, width


class Obj:
    """A connected component. Immutable snapshot of cells + derived facts."""

    __slots__ = ("color", "cells", "bbox", "bg")

    def __init__(self, color, cells, bg):
        self.color = color
        self.cells = frozenset(cells)
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        self.bbox = (min(rs), min(cs), max(rs), max(cs))
        self.bg = bg

    @property
    def size(self):
        return len(self.cells)

    @property
    def h(self):
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def w(self):
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def center(self):
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return (sum(rs) / len(rs), sum(cs) / len(cs))

    def subgrid(self, pad=0):
        """The object rendered alone on a background canvas."""
        r0, c0, r1, c1 = self.bbox
        h = r1 - r0 + 1 + 2 * pad
        w = c1 - c0 + 1 + 2 * pad
        out = [[self.bg] * w for _ in range(h)]
        for r, c in self.cells:
            out[r - r0 + pad][c - c0 + pad] = self.color
        return tuple(tuple(row) for row in out)

    def normalized_cells(self):
        """Cells shifted so the bbox corner is (0, 0)."""
        r0, c0 = self.bbox[0], self.bbox[1]
        return frozenset((r - r0, c - c0) for r, c in self.cells)

    def same_shape_as(self, other):
        return self.normalized_cells() == other.normalized_cells()

    def __repr__(self):
        return "Obj(color=%s, size=%d, bbox=%s)" % (
            self.color, self.size, self.bbox)


def components(g, connectivity=4, bg=None, min_size=1):
    """All maximal connected same-color regions (the bg color is skipped
    unless it is the only color present)."""
    if bg is None:
        bg = grids.background_color(g)
    h, w = shape(g)
    seen = [[False] * w for _ in range(h)]
    out = []
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (1, -1), (-1, 1), (-1, -1))
    for r in range(h):
        for c in range(w):
            if seen[r][c] or g[r][c] == bg:
                continue
            color = g[r][c]
            cells = []
            dq = deque([(r, c)])
            seen[r][c] = True
            while dq:
                cr, cc = dq.popleft()
                cells.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] \
                            and g[nr][nc] == color:
                        seen[nr][nc] = True
                        dq.append((nr, nc))
            if len(cells) >= min_size:
                out.append(Obj(color, cells, bg))
    out.sort(key=lambda o: (o.bbox[0], o.bbox[1], -o.size))
    return out


def components_by_color(g, connectivity=4, bg=None):
    """components(), grouped per non-bg color, preserving discovery order."""
    out = {}
    for o in components(g, connectivity, bg):
        out.setdefault(o.color, []).append(o)
    return out


def paint(g, objs, bg=0):
    """Paint objects onto a copy of grid g (in place semantics via rebuild)."""
    h, w = shape(g)
    out = [list(row) for row in g]
    for o in objs:
        for r, c in o.cells:
            if 0 <= r < h and 0 <= c < w:
                out[r][c] = o.color
    return tuple(tuple(row) for row in out)


def move_object(obj, dr, dc):
    """A copy of the object shifted by (dr, dc)."""
    return Obj(obj.color, [(r + dr, c + dc) for r, c in obj.cells], obj.bg)


def recolored(obj, color):
    return Obj(color, obj.cells, obj.bg)


def largest(objs):
    if not objs:
        return None
    return max(objs, key=lambda o: o.size)


def smallest(objs):
    if not objs:
        return None
    return min(objs, key=lambda o: o.size)


def sort_by(objs, key):
    return sorted(objs, key=key)


def objects_overlap(a, b):
    return bool(a.cells & b.cells)


def grid_of_objects(objs, h, w, bg=0):
    return paint(blank(h, w, bg), objs, bg)


# ------------------------------------------------------- object-level movement

def _shift_ok(cells, dr, dc, h, w, others):
    for r, c in cells:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return False
        if (nr, nc) in others:
            return False
    return True


def object_gravity(g, direction, bg=None, connectivity=4):
    """Slide every connected component as a UNIT until blocked.

    Unlike cell-level gravity, an object keeps its shape while falling.
    Objects are processed nearest-to-the-target-edge first so that piles
    form deterministically; passes repeat until nothing moves.
    """
    comps = components(g, connectivity, bg)
    if not comps:
        return g
    h, w = shape(g)
    if direction == "down":
        dr, dc = 1, 0
        comps = sorted(comps, key=lambda o: -o.bbox[2])
    elif direction == "up":
        dr, dc = -1, 0
        comps = sorted(comps, key=lambda o: o.bbox[0])
    elif direction == "right":
        dr, dc = 0, 1
        comps = sorted(comps, key=lambda o: -o.bbox[3])
    else:
        dr, dc = 0, -1
        comps = sorted(comps, key=lambda o: o.bbox[1])
    cells = [set(o.cells) for o in comps]
    moved = True
    while moved:
        moved = False
        for i in range(len(cells)):
            others = set()
            for j in range(len(cells)):
                if j != i:
                    others |= cells[j]
            while _shift_ok(cells[i], dr, dc, h, w, others):
                cells[i] = {(r + dr, c + dc) for r, c in cells[i]}
                others = {(r + dr, c + dc) for r, c in others}
                moved = True
    bgc = bg if bg is not None else 0
    out = [[bgc] * w for _ in range(h)]
    for o, cs in zip(comps, cells):
        for r, c in cs:
            out[r][c] = o.color
    return tuple(tuple(row) for row in out)


def extract_objects(g, orientation="h", order="reading", gap=0,
                    bg=None, connectivity=4):
    """Collect every component onto a fresh canvas, arranged in a row or
    column (top/left aligned), separated by `gap` background cells."""
    comps = components(g, connectivity, bg)
    if not comps:
        return g
    if bg is None:
        bg = comps[0].bg
    if order == "size_desc":
        comps = sorted(comps, key=lambda o: (-o.size, o.bbox))
    elif order == "size_asc":
        comps = sorted(comps, key=lambda o: (o.size, o.bbox))
    elif order == "color":
        comps = sorted(comps, key=lambda o: (o.color, o.bbox))
    subs = [o.subgrid() for o in comps]
    heights = [height(s) for s in subs]
    widths = [width(s) for s in subs]
    if orientation == "h":
        oh = max(heights)
        ow = sum(widths) + gap * (len(subs) - 1)
        out = [[bg] * ow for _ in range(oh)]
        c = 0
        for s, hh, ww in zip(subs, heights, widths):
            for r in range(hh):
                for cc in range(ww):
                    out[r][c + cc] = s[r][cc]
            c += ww + gap
    else:
        oh = sum(heights) + gap * (len(subs) - 1)
        ow = max(widths)
        out = [[bg] * ow for _ in range(oh)]
        r = 0
        for s, hh, ww in zip(subs, heights, widths):
            for rr in range(hh):
                for cc in range(ww):
                    out[r + rr][cc] = s[rr][cc]
            r += hh + gap
    return tuple(tuple(row) for row in out)
