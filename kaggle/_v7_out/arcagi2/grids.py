"""Grid primitives for the ARC-AGI-2 solver.



A Grid is an immutable tuple of tuples of ints (colors 0-9). Every function in

this module is pure: nothing mutates its input, and nothing depends on global

state. This keeps the program search reproducible and the test suite honest.

"""



from collections import Counter



Grid = tuple  # tuple[tuple[int, ...], ...]



BG_FALLBACK = 0





# ---------------------------------------------------------------- constructors



def from_rows(rows):

    """Build a grid from a list of lists (the JSON task format)."""

    return tuple(tuple(int(c) for c in row) for row in rows)





def to_rows(g):

    """Inverse of from_rows (for JSON serialization)."""

    return [list(row) for row in g]





def blank(h, w, color=0):

    return tuple(tuple(color for _ in range(w)) for _ in range(h))





# ------------------------------------------------------------------- accessors



def shape(g):

    return (len(g), len(g[0]) if g else 0)





def height(g):

    return len(g)





def width(g):

    return len(g[0]) if g else 0





def colors_used(g):

    out = set()

    for row in g:

        out.update(row)

    return out





def color_counts(g):

    c = Counter()

    for row in g:

        c.update(row)

    return c





def most_common_color(g, exclude=()):

    c = color_counts(g)

    for col in exclude:

        c.pop(col, None)

    if not c:

        return BG_FALLBACK

    best_n = max(c.values())

    tied = sorted(col for col, n in c.items() if n == best_n)

    # ARC convention: 0 is the background, so it wins count ties.

    return 0 if 0 in tied else tied[0]





def background_color(g):

    """Heuristic background: the most common color in the grid."""

    return most_common_color(g)





def cell(g, r, c):

    if 0 <= r < height(g) and 0 <= c < width(g):

        return g[r][c]

    return None





# ------------------------------------------------------------------ comparison



def grids_equal(a, b):

    return a == b





def diff_cells(a, b):

    """Cells (r, c, color_a, color_b) where two same-shaped grids differ."""

    if shape(a) != shape(b):

        return None

    out = []

    for r in range(height(a)):

        for c in range(width(a)):

            if a[r][c] != b[r][c]:

                out.append((r, c, a[r][c], b[r][c]))

    return out





# ------------------------------------------------------------------ geometry



def rotate90(g):

    return tuple(zip(*g[::-1]))





def rotate180(g):

    return tuple(tuple(row) for row in rotate90(rotate90(g)))





def rotate270(g):

    return tuple(zip(*g))[::-1]





def flip_h(g):

    """Mirror left-right."""

    return tuple(tuple(row[::-1]) for row in g)





def flip_v(g):

    """Mirror top-bottom."""

    return g[::-1]





def transpose(g):

    return tuple(zip(*g))





def anti_transpose(g):

    return rotate180(transpose(g))





def translate(g, dh, dw, bg=BG_FALLBACK):

    """Shift content by (dh, dw); vacated cells become bg."""

    h, w = shape(g)

    out = [[bg] * w for _ in range(h)]

    for r in range(h):

        for c in range(w):

            nr, nc = r + dh, c + dw

            if 0 <= nr < h and 0 <= nc < w:

                out[nr][nc] = g[r][c]

    return tuple(tuple(row) for row in out)





# -------------------------------------------------------------------- crop/pad



def bbox(g, bg=None):

    """Bounding box (r0, c0, r1, c1) of non-bg cells; None if empty."""

    if bg is None:

        bg = background_color(g)

    r0, c0, r1, c1 = height(g), width(g), -1, -1

    for r in range(height(g)):

        for c in range(width(g)):

            if g[r][c] != bg:

                r0 = min(r0, r)

                c0 = min(c0, c)

                r1 = max(r1, r)

                c1 = max(c1, c)

    if r1 < r0:

        return None

    return (r0, c0, r1, c1)





def crop_bbox(g, bg=None):

    box = bbox(g, bg)

    if box is None:

        return g

    r0, c0, r1, c1 = box

    return tuple(tuple(g[r][c0:c1 + 1]) for r in range(r0, r1 + 1))





def pad_to(g, h, w, bg=BG_FALLBACK, anchor="center"):

    """Pad (or crop) to exactly (h, w). anchor: center|tl|tr|bl|br."""

    cur_h, cur_w = shape(g)

    if h < cur_h or w < cur_w:

        return crop_to(g, h, w, anchor)

    top, left = _anchor_offsets(anchor, cur_h, cur_w, h, w)

    out = [[bg] * w for _ in range(h)]

    for r in range(cur_h):

        for c in range(cur_w):

            out[top + r][left + c] = g[r][c]

    return tuple(tuple(row) for row in out)





def crop_to(g, h, w, anchor="center"):

    ch, cw = shape(g)

    top, left = _anchor_offsets(anchor, ch, cw, h, w)

    out = []

    for r in range(top, top + h):

        row = []

        for c in range(left, left + w):

            row.append(g[r][c] if 0 <= r < ch and 0 <= c < cw else BG_FALLBACK)

        out.append(tuple(row))

    return tuple(out)





def _anchor_offsets(anchor, cur_h, cur_w, new_h, new_w):

    if anchor == "tl":

        return 0, 0

    if anchor == "tr":

        return 0, max(0, new_w - cur_w)

    if anchor == "bl":

        return max(0, new_h - cur_h), 0

    if anchor == "br":

        return max(0, new_h - cur_h), max(0, new_w - cur_w)

    return max(0, (new_h - cur_h) // 2), max(0, (new_w - cur_w) // 2)  # center





# ---------------------------------------------------------------------- resize



def upscale(g, fh, fw):

    if fh <= 0 or fw <= 0:

        return g

    out = []

    for row in g:

        big_row = []

        for cell_v in row:

            big_row.extend([cell_v] * fw)

        for _ in range(fh):

            out.append(tuple(big_row))

    return tuple(out)





def downscale(g, fh, fw, mode="majority"):

    """Reduce each fh x fw block to one cell.



    mode:

      majority - most common color in the block (ties -> smaller color)

      first    - top-left cell

      unique   - the color if the block is uniform, else majority

    """

    h, w = shape(g)

    if fh <= 0 or fw <= 0:

        return g

    oh, ow = h // fh, w // fw

    if oh == 0 or ow == 0:

        return g

    out = []

    for br in range(oh):

        row = []

        for bc in range(ow):

            block = [

                g[br * fh + dr][bc * fw + dc]

                for dr in range(fh)

                for dc in range(fw)

            ]

            if mode == "first":

                row.append(block[0])

            elif mode == "unique":

                row.append(block[0] if len(set(block)) == 1 else

                           Counter(block).most_common(1)[0][0])

            else:  # majority

                cnt = Counter(block)

                best = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]

                row.append(best)

        out.append(tuple(row))

    return tuple(out)





# -------------------------------------------------------------------- color ops



def recolor(g, mapping):

    return tuple(tuple(mapping.get(v, v) for v in row) for row in g)





def keep_colors(g, colors, bg=BG_FALLBACK):

    keep = set(colors)

    return tuple(tuple(v if v in keep else bg for v in row) for row in g)





def replace_color(g, old, new):

    return tuple(tuple(new if v == old else v for v in row) for row in g)





def swap_colors(g, a, b):

    return tuple(

        tuple(b if v == a else (a if v == b else v) for v in row) for row in g

    )





def invert_colors(g, palette):

    """Map each color to its index-mirror inside the given palette."""

    pal = list(palette)

    mapping = {c: pal[len(pal) - 1 - i] for i, c in enumerate(pal)}

    return recolor(g, mapping)





# ------------------------------------------------------------------ structural



def tile(g, nh, nw):

    out = []

    for _ in range(nh):

        for row in g:

            out.append(tuple(row) * nw)

    return tuple(out)





def fill_holes(g, hole=0, color=None):

    """Fill regions of `hole` color that are fully enclosed by other colors.



    color defaults to the most common non-hole color (the usual ARC intent:

    "fill the enclosed region with the surrounding color").

    """

    h, w = shape(g)

    outside = [[False] * w for _ in range(h)]

    stack = []

    for r in range(h):

        for c in range(w):

            if (r in (0, h - 1) or c in (0, w - 1)) and g[r][c] == hole and not outside[r][c]:

                outside[r][c] = True

                stack.append((r, c))

    while stack:

        r, c = stack.pop()

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):

            nr, nc = r + dr, c + dc

            if 0 <= nr < h and 0 <= nc < w and not outside[nr][nc] and g[nr][nc] == hole:

                outside[nr][nc] = True

                stack.append((nr, nc))

    if color is None:

        counts = {}

        for row in g:

            for v in row:

                if v != hole:

                    counts[v] = counts.get(v, 0) + 1

        if not counts:

            return g

        color = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    out = []

    for r in range(h):

        row = []

        for c in range(w):

            row.append(color if (g[r][c] == hole and not outside[r][c])

                       else g[r][c])

        out.append(tuple(row))

    return tuple(out)





def complete_h_symmetry(g, bg=None):

    """Mirror-complete left-right: fill bg cells from their mirror if set."""

    if bg is None:

        bg = background_color(g)

    w = width(g)

    out = []

    for row in g:

        new = list(row)

        for c in range(w):

            m = w - 1 - c

            if new[c] == bg and new[m] != bg:

                new[c] = new[m]

            elif new[m] == bg and new[c] != bg:

                new[m] = new[c]

        out.append(tuple(new))

    return tuple(out)





def complete_v_symmetry(g, bg=None):

    return transpose(complete_h_symmetry(transpose(g), bg))





def gravity(g, direction, bg=None):

    """Slide every non-bg cell in `direction` until blocked."""

    if bg is None:

        bg = background_color(g)

    h, w = shape(g)

    if direction == "down":

        for c in range(w):

            write = h - 1

            for r in range(h - 1, -1, -1):

                if g[r][c] != bg:

                    g = _set(g, write, c, g[r][c]) if write != r else g

                    write -= 1

            for r in range(write, -1, -1):

                g = _set(g, r, c, bg)

    elif direction == "up":

        for c in range(w):

            write = 0

            for r in range(h):

                if g[r][c] != bg:

                    g = _set(g, write, c, g[r][c]) if write != r else g

                    write += 1

            for r in range(write, h):

                g = _set(g, r, c, bg)

    elif direction == "right":

        for r in range(h):

            write = w - 1

            for c in range(w - 1, -1, -1):

                if g[r][c] != bg:

                    g = _set(g, r, write, g[r][c]) if write != c else g

                    write -= 1

            for c in range(write, -1, -1):

                g = _set(g, r, c, bg)

    elif direction == "left":

        for r in range(h):

            write = 0

            for c in range(w):

                if g[r][c] != bg:

                    g = _set(g, r, write, g[r][c]) if write != c else g

                    write += 1

            for c in range(write, w):

                g = _set(g, r, c, bg)

    return g





def _set(g, r, c, v):

    return g[:r] + (g[r][:c] + (v,) + g[r][c + 1:],) + g[r + 1:]





def border(g, color, thickness=1):

    h, w = shape(g)

    t = min(thickness, h - 1, w - 1)

    if t < 1:

        return g

    out = [list(row) for row in g]

    for k in range(t):

        for c in range(w):

            out[k][c] = color

            out[h - 1 - k][c] = color

        for r in range(h):

            out[r][k] = color

            out[r][w - 1 - k] = color

    return tuple(tuple(row) for row in out)





def outline_nonbg(g, color, bg=None):

    """Set to `color` every bg cell orthogonally adjacent to a non-bg cell."""

    if bg is None:

        bg = background_color(g)

    h, w = shape(g)

    out = [list(row) for row in g]

    for r in range(h):

        for c in range(w):

            if g[r][c] == bg and _has_nonbg_neighbor(g, r, c, bg):

                out[r][c] = color

    return tuple(tuple(row) for row in out)





def _has_nonbg_neighbor(g, r, c, bg):

    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):

        v = cell(g, r + dr, c + dc)

        if v is not None and v != bg:

            return True

    return False