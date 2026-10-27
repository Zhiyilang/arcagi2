"""Parameter inference: derive transform parameters from the demonstrations.

The ARC-AGI-2 scoring gives credit when *either* of two attempts matches, so
precision matters more than recall: every transform inferred here must
reproduce ALL train pairs exactly, or it is discarded. A wrong-but-plausible
guess would waste one of the two attempts.
"""

import itertools

from . import grids
from . import objects as objmod
from .dsl import T, ANCHORS, COLORS


def _consistent(t, pairs):
    """True when t reproduces every (input, output) train pair exactly."""
    for gi, go in pairs:
        if t.apply(gi) != go:
            return False
    return True


def infer_recolor_map(pairs):
    """If every pair is shape-preserving, derive the global color map.

    Only emitted when the cell-wise map is a consistent *function* (each
    input color maps to exactly one output color) and is not the identity.
    """
    mapping = {}
    for gi, go in pairs:
        if grids.shape(gi) != grids.shape(go):
            return []
        for r in range(grids.height(gi)):
            for c in range(grids.width(gi)):
                a, b = gi[r][c], go[r][c]
                if a in mapping and mapping[a] != b:
                    return []
                mapping[a] = b
    if not mapping or all(k == v for k, v in mapping.items()):
        return []
    return [T("recolor", tuple(sorted(mapping.items())))]


def infer_pad(pairs):
    """If every output shares one shape, find the (anchor, bg) padding."""
    shapes = {grids.shape(go) for _, go in pairs}
    if len(shapes) != 1:
        return []
    h, w = next(iter(shapes))
    out = []
    for anchor in ANCHORS:
        for bg in COLORS:
            t = T("pad", (h, w, anchor, bg))
            if _consistent(t, pairs):
                out.append(t)
    return out


def infer_upscale(pairs):
    """Exact integer upscale by block repetition (beyond the enumerated 2-5)."""
    out = []
    for fh in range(2, 9):
        for fw in range(2, 9):
            t = T("up", (fh, fw))
            if _consistent(t, pairs):
                out.append(t)
    return out


def infer_translate(pairs):
    """Largest-range translation search, with every candidate background."""
    out = []
    max_h = max(grids.height(gi) for gi, _ in pairs)
    max_w = max(grids.width(gi) for gi, _ in pairs)
    seen = set()
    for dh in range(-max_h, max_h + 1):
        for dw in range(-max_w, max_w + 1):
            if not dh and not dw:
                continue
            for bg in COLORS:
                t = T("translate", (dh, dw, bg))
                if t.signature() in seen:
                    continue
                if _consistent(t, pairs):
                    out.append(t)
                    seen.add(t.signature())
    return out


def infer_crop_bg(pairs):
    """Crop with an explicit background color (auto-bg may pick wrong)."""
    out = []
    for bg in COLORS:
        t = T("crop_b", (bg,))
        if _consistent(t, pairs):
            out.append(t)
    return out


def infer_object_remap(pairs):
    """Per-color object transform (AGI-2 "every color moves/recolors by its
    own rule" family).

    Components are matched across each pair by normalized shape; within a
    shape group the input/output objects are paired by the bijection with
    minimal total bbox movement (so an in-place recolor pairs with zero
    movement, not by color order). Each input color must then map to one
    (dr, dc, new_color), identical in every pair. Anything else (shape
    gained/lost, ambiguous per-color rule) yields nothing — a wrong guess
    would burn one of the two attempts. Connectivity 4 is tried first, then
    8 (diagonal merges can align the segmentation).
    """
    if not pairs:
        return []
    for conn in (4, 8):
        found = _object_remap_with_connectivity(pairs, conn)
        if found:
            return found
    return []


def _shape_groups(objs):
    groups = {}
    for o in objs:
        groups.setdefault(o.normalized_cells(), []).append(o)
    return groups


def _min_movement_bijections(ins, outs, cap=256):
    """All bijections input->output achieving minimal total movement.

    Exact for groups up to 7 objects; larger groups fall back to a
    deterministic greedy nearest assignment (one bijection).
    """
    k = len(ins)
    if k == 1:
        return [(0,)]

    def cost(i, j):
        return (abs(outs[j].bbox[0] - ins[i].bbox[0])
                + abs(outs[j].bbox[1] - ins[i].bbox[1]))

    if k > 7:
        remaining = list(range(k))
        perm = [None] * k
        for i in range(k):
            j = min(remaining, key=lambda j: (cost(i, j), j))
            perm[i] = j
            remaining.remove(j)
        return [tuple(perm)]

    best = None
    bijs = []
    for perm in itertools.permutations(range(k)):
        c = 0
        for i in range(k):
            c += cost(i, perm[i])
        if best is None or c < best:
            best = c
            bijs = [perm]
        elif c == best:
            bijs.append(perm)
        if len(bijs) >= cap:
            break
    return bijs


def _pair_partial_mappings(gi, go, bg_i, bg_o, conn):
    """Per-color move/recolor maps suggested by ONE train pair, or None when
    the pair is outside this family (object count or shape multiset changed).
    """
    in_objs = objmod.components(gi, conn, bg_i)
    out_objs = objmod.components(go, conn, bg_o)
    if len(in_objs) != len(out_objs):
        return None
    in_groups = _shape_groups(in_objs)
    out_groups = _shape_groups(out_objs)
    if set(in_groups) != set(out_groups):
        return None
    group_bijs = []
    for key in sorted(in_groups, key=repr):
        ins = in_groups[key]
        outs = out_groups[key]
        if len(ins) != len(outs):
            return None
        group_bijs.append((ins, outs, _min_movement_bijections(ins, outs)))

    partials = []

    def rec(i, acc):
        if len(partials) >= 64:
            return
        if i == len(group_bijs):
            partials.append(dict(acc))
            return
        ins, outs, bijs = group_bijs[i]
        for perm in bijs:
            ok = True
            added = []
            for a, j in zip(ins, perm):
                b = outs[j]
                entry = (b.bbox[0] - a.bbox[0], b.bbox[1] - a.bbox[1],
                         b.color)
                if a.color in acc and acc[a.color] != entry:
                    ok = False
                    break
                if a.color not in acc:
                    added.append(a.color)
                acc[a.color] = entry
            if ok:
                rec(i + 1, acc)
            for color in added:
                acc.pop(color, None)

    rec(0, {})
    return partials


def _object_remap_with_connectivity(pairs, conn):
    bg_out = None
    pair_partials = []
    for gi, go in pairs:
        if grids.shape(gi) != grids.shape(go):
            return []
        bg_i = grids.background_color(gi)
        bg_o = grids.background_color(go)
        if bg_out is None:
            bg_out = bg_o
        elif bg_out != bg_o:
            return []
        partials = _pair_partial_mappings(gi, go, bg_i, bg_o, conn)
        if not partials:
            return []
        pair_partials.append(partials)

    candidates = list(pair_partials[0])
    for partials in pair_partials[1:]:
        nxt = {}
        for cand in candidates:
            for p in partials:
                merged = dict(cand)
                ok = True
                for color, entry in p.items():
                    if color in merged and merged[color] != entry:
                        ok = False
                        break
                    merged[color] = entry
                if ok:
                    nxt[tuple(sorted(merged.items()))] = merged
        candidates = [nxt[k] for k in sorted(nxt, key=repr)]
        if not candidates:
            return []

    out = []
    for m in candidates:
        if all(dr == 0 and dc == 0 and nc == c
               for c, (dr, dc, nc) in m.items()):
            continue
        args = (bg_out,) + tuple(
            (c, dr, dc, nc) for c, (dr, dc, nc) in sorted(m.items()))
        out.append(T("obj_remap", args))
    return out


def inferred_transforms(pairs):
    """All inferred transforms, simplest first, de-duplicated."""
    found = []
    for fn in (infer_recolor_map, infer_crop_bg, infer_pad,
               infer_upscale, infer_translate, infer_object_remap):
        found.extend(fn(pairs))
    uniq = {}
    for t in found:
        uniq[t.signature()] = t
    return sorted(uniq.values(), key=lambda t: (len(t.args), t.family, t.args))
