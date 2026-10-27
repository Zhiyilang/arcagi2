"""The solver: search transform programs that reproduce every train pair.



Strategy

--------

1. Infer high-precision transforms from the demonstrations (recolor maps,

   padding targets, translations) — these are derived, not guessed.

2. Enumerate the closed transform library and keep everything that

   reproduces ALL train pairs exactly (depth 1).

3. If nothing is consistent, search depth-2 compositions (and a beam-limited

   depth 3) with pair-0-first rejection ordering.

4. Produce TWO attempts per test input, because Kaggle scores a task output

   as correct when either attempt matches. Attempt 1 is the simplest

   consistent program; attempt 2 is the simplest program with a different

   family signature (genuine diversity, not noise).



Every consistency check is exact equality on the demonstration pairs. The

solver never "votes" or averages — a program either reproduces the

demonstrations or it is discarded.

"""



import time



from . import grids

from . import dsl

from . import infer

from .dsl import T





DEFAULT_BUDGET_S = 8.0

DEFAULT_MAX_DEPTH = 3

BEAM_WIDTH = 4000





def _consistent(t, pairs):

    for gi, go in pairs:

        if t.apply(gi) != go:

            return False

    return True





def _program_consistent(program, pairs):

    for gi, go in pairs:

        if dsl.apply_program(gi, program) != go:

            return False

    return True





def _ranked(solutions):

    """De-duplicate by signature, simplest first."""

    uniq = {}

    for p in solutions:

        uniq[tuple(t.signature() for t in p)] = p

    return sorted(uniq.values(), key=dsl.program_cost)





def search_programs(pairs, budget_s=DEFAULT_BUDGET_S,

                    max_depth=DEFAULT_MAX_DEPTH, deadline=None):

    """Return consistent programs, simplest first (may be empty)."""

    if deadline is None:

        deadline = time.monotonic() + budget_s



    cands = dsl.enumerated_candidates()

    cands.extend(infer.inferred_transforms(pairs))

    seen = set()

    uniq = []

    for t in cands:

        if t.signature() not in seen:

            seen.add(t.signature())

            uniq.append(t)

    cands = uniq

    stage2 = dsl.second_stage_candidates()



    solutions = []



    # ---- depth 1 -------------------------------------------------------

    for t in cands:

        if _consistent(t, pairs):

            solutions.append((t,))

    if solutions:

        return _ranked(solutions)



    if max_depth < 2 or time.monotonic() > deadline:

        return []



    # ---- depth 2 -------------------------------------------------------

    partials = []

    for t1 in cands:

        if time.monotonic() > deadline:

            break

        inters = [t1.apply(gi) for gi, _ in pairs]

        for t2 in stage2:

            if t2.apply(inters[0]) != pairs[0][1]:

                continue

            if all(t2.apply(inters[i]) == pairs[i][1]

                   for i in range(1, len(pairs))):

                solutions.append((t1, t2))

            else:

                partials.append((t1, t2, [t2.apply(x) for x in inters]))

    if solutions:

        return _ranked(solutions)



    if max_depth < 3 or time.monotonic() > deadline:

        return []



    # ---- depth 3 (beam over partials that already satisfy pair 0) -------

    partials.sort(key=lambda p: dsl.program_cost(p[:2]))

    for t1, t2, inters in partials[:BEAM_WIDTH]:

        for t3 in stage2:

            if t3.apply(inters[0]) != pairs[0][1]:

                continue

            if all(t3.apply(inters[i]) == pairs[i][1]

                   for i in range(1, len(pairs))):

                solutions.append((t1, t2, t3))

        if time.monotonic() > deadline:

            break

    return _ranked(solutions)





def _fallback_attempts(pairs, test_input):

    """No consistent program: two cheap, diverse guesses."""

    a1 = test_input  # identity: some tasks are (near) identity

    shapes = [grids.shape(go) for _, go in pairs]

    colors = []

    for _, go in pairs:

        colors.extend(grids.colors_used(go))

    from collections import Counter

    modal_shape = Counter(shapes).most_common(1)[0][0]

    modal_color = Counter(colors).most_common(1)[0][0] if colors else 0

    a2 = grids.blank(modal_shape[0], modal_shape[1], modal_color)

    return a1, a2





def solve_task(task, budget_s=DEFAULT_BUDGET_S,

               max_depth=DEFAULT_MAX_DEPTH):

    """Solve one task dict (as loaded from the ARC-AGI-2 JSON format).



    Returns {"predictions": [ {"attempt_1": grid, "attempt_2": grid}, ... ],

             "program": [transform signatures] or None,

             "solved": bool}

    """

    pairs = [(grids.from_rows(p["input"]), grids.from_rows(p["output"]))

             for p in task["train"]]

    test_inputs = [grids.from_rows(p["input"]) for p in task["test"]]



    programs = search_programs(pairs, budget_s=budget_s,

                               max_depth=max_depth)



    predictions = []

    used = None

    if programs:

        used = programs[0]

        alt = None

        sig0 = tuple(t.family for t in used)

        for p in programs[1:]:

            if tuple(t.family for t in p) != sig0:

                alt = p

                break

        for gi in test_inputs:

            a1 = dsl.apply_program(gi, used)

            a2 = dsl.apply_program(gi, alt) if alt else _fallback_attempts(

                pairs, gi)[1]

            predictions.append({"attempt_1": a1, "attempt_2": a2})

        return {"predictions": predictions,

                "program": [t.signature() for t in used],

                "solved": True}



    for gi in test_inputs:

        a1, a2 = _fallback_attempts(pairs, gi)

        predictions.append({"attempt_1": a1, "attempt_2": a2})

    return {"predictions": predictions, "program": None, "solved": False}