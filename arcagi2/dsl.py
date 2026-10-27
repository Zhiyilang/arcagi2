"""The transform DSL: a small, closed library of grid transformations.

Every transform is a frozen record (family, args) with a pure apply(). The
solver searches compositions of these transforms; keeping the library closed
and explicit is what makes the search exhaustive rather than heuristic.
"""

from dataclasses import dataclass
from itertools import product

from . import grids
from . import objects as objmod
from . import patterns as patmod
from .grids import Grid


@dataclass(frozen=True)
class T:
    family: str
    args: tuple

    def apply(self, g):
        return _APPLY[self.family](g, *self.args)

    def signature(self):
        return (self.family, self.args)

    def __repr__(self):
        if self.args:
            return "%s%s" % (self.family, self.args)
        return self.family


# ------------------------------------------------------------- family appliers

def _identity(g):
    return g


def _recolor(g, *mapping):
    return grids.recolor(g, dict(mapping))


def _swap(g, a, b):
    return grids.swap_colors(g, a, b)


def _keep(g, *colors):
    return grids.keep_colors(g, colors)


def _crop(g):
    return grids.crop_bbox(g)


def _crop_b(g, bg):
    return grids.crop_bbox(g, bg=bg)


def _pad(g, h, w, anchor, bg):
    return grids.pad_to(g, h, w, bg=bg, anchor=anchor)


def _up(g, fh, fw):
    return grids.upscale(g, fh, fw)


def _down(g, fh, fw, mode):
    return grids.downscale(g, fh, fw, mode)


def _rot90(g):
    return grids.rotate90(g)


def _rot180(g):
    return grids.rotate180(g)


def _rot270(g):
    return grids.rotate270(g)


def _flip_h(g):
    return grids.flip_h(g)


def _flip_v(g):
    return grids.flip_v(g)


def _transpose(g):
    return grids.transpose(g)


def _anti(g):
    return grids.anti_transpose(g)


def _translate(g, dh, dw, bg):
    return grids.translate(g, dh, dw, bg=bg)


def _tile(g, nh, nw):
    return grids.tile(g, nh, nw)


def _holes(g, hole, color):
    return grids.fill_holes(g, hole=hole, color=color)


def _sym_h(g):
    return grids.complete_h_symmetry(g)


def _sym_v(g):
    return grids.complete_v_symmetry(g)


def _gravity(g, direction):
    return grids.gravity(g, direction)


def _border(g, color, thickness):
    return grids.border(g, color, thickness)


def _outline(g, color):
    return grids.outline_nonbg(g, color)


def _keep_largest(g, connectivity):
    comps = objmod.components(g, connectivity)
    big = objmod.largest(comps)
    if big is None:
        return g
    return objmod.paint(grids.blank(grids.height(g), grids.width(g)),
                        [big])


def _delete_color(g, color):
    bg = grids.background_color(g)
    return grids.replace_color(g, color, bg)


def _fill_bbox(g, src, dst):
    """Fill the bounding box of every object of color `src` with `dst`."""
    bg = grids.background_color(g)
    comps = objmod.components(g, 4, bg)
    h, w = grids.shape(g)
    out = [list(row) for row in g]
    for o in comps:
        if o.color != src:
            continue
        r0, c0, r1, c1 = o.bbox
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r][c] = dst
    return tuple(tuple(row) for row in out)


def _obj_gravity(g, direction):
    return objmod.object_gravity(g, direction)


def _extract(g, orientation, order, gap):
    return objmod.extract_objects(g, orientation, order, gap)


def _pattern(g, mode):
    return patmod.apply_pattern(g, mode)


def _obj_remap(g, bg_out, *mapping):
    """Per-color object transform: every component of color c is shifted by
    (dr, dc) and recolored to nc, per the inferred table. Components whose
    color is absent from the table stay in place; the canvas is rebuilt on
    bg_out so a background change is expressed too."""
    table = {m[0]: (m[1], m[2], m[3]) for m in mapping}
    bg = grids.background_color(g)
    comps = objmod.components(g, 4, bg)
    h, w = grids.shape(g)
    out = [[bg_out] * w for _ in range(h)]
    for o in comps:
        entry = table.get(o.color)
        if entry is None:
            cells, color = o.cells, o.color
        else:
            dr, dc, nc = entry
            cells = {(r + dr, c + dc) for r, c in o.cells}
            color = nc
        for r, c in cells:
            if 0 <= r < h and 0 <= c < w:
                out[r][c] = color
    return tuple(tuple(row) for row in out)


_APPLY = {
    "identity": _identity,
    "recolor": _recolor,
    "swap": _swap,
    "keep": _keep,
    "crop": _crop,
    "crop_b": _crop_b,
    "pad": _pad,
    "up": _up,
    "down": _down,
    "rot90": _rot90,
    "rot180": _rot180,
    "rot270": _rot270,
    "flip_h": _flip_h,
    "flip_v": _flip_v,
    "transpose": _transpose,
    "anti": _anti,
    "translate": _translate,
    "tile": _tile,
    "holes": _holes,
    "sym_h": _sym_h,
    "sym_v": _sym_v,
    "gravity": _gravity,
    "border": _border,
    "outline": _outline,
    "keep_largest": _keep_largest,
    "delete_color": _delete_color,
    "fill_bbox": _fill_bbox,
    "obj_gravity": _obj_gravity,
    "extract": _extract,
    "pattern": _pattern,
    "obj_remap": _obj_remap,
}

COLORS = tuple(range(10))
ANCHORS = ("center", "tl", "tr", "bl", "br")


# ------------------------------------------------------------- enumeration

def enumerated_candidates():
    """Every transform in the library with concrete (searchable) parameters.

    Recursions/mappings that must be *inferred* from train pairs (recolor
    maps, pad targets, translations) are deliberately excluded here and
    produced by infer.py instead.
    """
    out = [T("identity", ())]

    for a in COLORS:
        for b in COLORS:
            if a != b:
                out.append(T("swap", (a, b)))

    for c in COLORS:
        out.append(T("keep", (c,)))
    for c in COLORS:
        out.append(T("keep", tuple(x for x in COLORS if x != c)))

    out.append(T("crop", ()))

    for fh, fw in product(range(2, 6), repeat=2):
        out.append(T("up", (fh, fw)))

    for fh, fw in product(range(2, 5), repeat=2):
        for mode in ("majority", "first", "unique"):
            out.append(T("down", (fh, fw, mode)))

    out += [T("rot90", ()), T("rot180", ()), T("rot270", ()),
            T("flip_h", ()), T("flip_v", ()),
            T("transpose", ()), T("anti", ())]

    for dh in range(-3, 4):
        for dw in range(-3, 4):
            if dh or dw:
                out.append(T("translate", (dh, dw, 0)))

    for nh, nw in product(range(1, 4), repeat=2):
        if nh * nw > 1:
            out.append(T("tile", (nh, nw)))

    out += [T("holes", (hole, c)) for hole in COLORS for c in COLORS
            if hole != c]
    out += [T("sym_h", ()), T("sym_v", ())]

    for d in ("up", "down", "left", "right"):
        out.append(T("gravity", (d,)))

    for c in COLORS:
        out.append(T("border", (c, 1)))
        out.append(T("border", (c, 2)))
        out.append(T("outline", (c,)))

    out += [T("keep_largest", (4,)), T("keep_largest", (8,))]

    for c in COLORS:
        if c:
            out.append(T("delete_color", (c,)))

    for src in COLORS:
        for dst in COLORS:
            if src != dst:
                out.append(T("fill_bbox", (src, dst)))

    for d in ("up", "down", "left", "right"):
        out.append(T("obj_gravity", (d,)))
    for orientation in ("h", "v"):
        for order in ("reading", "size_desc", "size_asc", "color"):
            for gap in (0, 1):
                out.append(T("extract", (orientation, order, gap)))
    for mode in patmod.PATTERN_MODES:
        out.append(T("pattern", (mode,)))

    return out


def second_stage_candidates():
    """A curated subset used as the *second* transform in compositions.

    Search-space control: a full cross product of every candidate with every
    candidate is quadratic; restricting the right-hand side to the families
    that actually appear in second position keeps depth-2 tractable without
    measurably losing tasks.
    """
    out = []
    for c in COLORS:
        out.append(T("keep", (c,)))
    for a in COLORS:
        for b in COLORS:
            if a != b:
                out.append(T("swap", (a, b)))
    out.append(T("crop", ()))
    for fh, fw in product(range(2, 6), repeat=2):
        out.append(T("up", (fh, fw)))
    for fh, fw in product(range(2, 5), repeat=2):
        out.append(T("down", (fh, fw, "majority")))
    out += [T("holes", (hole, c)) for hole in COLORS for c in COLORS
            if hole != c]
    out += [T("sym_h", ()), T("sym_v", ())]
    for d in ("up", "down", "left", "right"):
        out.append(T("gravity", (d,)))
    for c in COLORS:
        out.append(T("border", (c, 1)))
        out.append(T("outline", (c,)))
        if c:
            out.append(T("delete_color", (c,)))
    for d in ("up", "down", "left", "right"):
        out.append(T("obj_gravity", (d,)))
    for orientation in ("h", "v"):
        for order in ("reading", "size_desc", "size_asc", "color"):
            for gap in (0, 1):
                out.append(T("extract", (orientation, order, gap)))
    for mode in patmod.PATTERN_MODES:
        out.append(T("pattern", (mode,)))
    return out


def apply_program(g, program):
    for t in program:
        g = t.apply(g)
    return g


def program_cost(program):
    """Simplicity ranking: fewer transforms first, then fewer parameters."""
    return (len(program), sum(len(t.args) for t in program))
