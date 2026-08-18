"""Learning transitivity from data — incrementally, per new edge.

Transitivity is not declared by hand, it is learned: if with the same
predicate A→B, B→C AND A→C are all WITNESSED (taught or read, never inferred),
that predicate closes triangles and is transitive. "tür" learns it from is-a
examples; "sever" never does, because the chain of love does not close in the
graph. ONE triangle is not enough — a non-transitive relation can close one by
coincidence, and a single accident used to make a predicate transitive for
life and produce a wrong #inference from every new edge. Two independently
witnessed triangles are required.

WHY THIS FILE EXISTS. The rule above was implemented twice (v3 and lmm), and
both implementations answered the question by SCANNING THE WHOLE PREDICATE
from scratch, on every single written cell. Measured on table ingestion:

    2000 cells   0.91 s
   10000 cells  25.6 s          5x the data, 28x the time

That is the shape of a quadratic: each cell re-derives what all the previous
cells already established. The question "did this predicate close a triangle"
is CUMULATIVE — the answer only ever grows — so it is asked once per EDGE and
never re-asked:

    the first edge of a predicate  builds the adjacency and counts what stands
    every edge after it            counts only the triangles THAT EDGE closes,
                                   in the degree of its two endpoints
    a predicate already transitive returns immediately
    a predicate that closed nothing REMEMBERS that it closed nothing

The last line is the one that matters most in practice: a negative answer is
kept, not recomputed. Nothing here belongs to language — it is adjacency sets
and a counter.
"""

# Independently witnessed triangles needed before a predicate is transitive.
WITNESSED = 2

# Records with this source are DERIVED, and a derivation may not be evidence
# for the rule that produced it — that circle is how a single accident used to
# bootstrap itself into a law.
DERIVED = "#inference"


class Transitivity:
    """Per-memory tracker: adjacency, triangle count, and what it has seen."""

    def __init__(self, memory):
        self.memory = memory
        self._out = {}          # predicate -> {subject: {value}}
        self._in = {}           # predicate -> {value: {subject}}
        self._count = {}        # predicate -> triangles counted so far
        self._seen = {}         # predicate -> {(subject, value)} already counted

    # --- the one question -------------------------------------------------

    def observe(self, predicate, subject=None, value=None):
        """A new edge was witnessed (or: re-examine this predicate).

        `subject`/`value` may be omitted — then only the initial pass runs,
        which is what a caller asking "is this predicate transitive by now"
        (rather than reporting an edge) wants.

        Returns: True if the predicate is transitive.
        """
        if predicate is None:
            return False
        if predicate in self.memory.transitive:
            return True
        first = predicate not in self._out
        if first:
            self._build(predicate)
        if not first and subject is not None and value is not None:
            self._add(predicate, subject, value)
        if self._count.get(predicate, 0) >= WITNESSED:
            self.memory.transitive.add(predicate)
            return True
        return False

    # --- the initial pass -------------------------------------------------

    def _build(self, predicate):
        """Reads what already stands for this predicate — ONCE.

        The cost is the predicate's degree, paid on its first edge (and on a
        memory loaded from file, whose triangles no `observe` ever saw).
        """
        out, inn, seen = {}, {}, set()
        for key in self.memory.by_predicate.get(predicate, ()):
            record = self.memory.records.get(key)
            if record is None or record.source == DERIVED:
                continue
            out.setdefault(record.subject, set()).add(record.value)
            inn.setdefault(record.value, set()).add(record.subject)
            seen.add((record.subject, record.value))
        self._out[predicate] = out
        self._in[predicate] = inn
        self._seen[predicate] = seen
        triangles = 0
        for a, bs in out.items():
            for b in bs:
                for c in out.get(b, ()):
                    if c in bs and c != a and b != a:
                        triangles += 1
                        if triangles >= WITNESSED:
                            break
                if triangles >= WITNESSED:
                    break
            if triangles >= WITNESSED:
                break
        self._count[predicate] = triangles

    # --- the incremental step ---------------------------------------------

    def _add(self, predicate, subject, value):
        """Counts the triangles THIS edge closes, then keeps it.

        A new edge (u, v) can complete a triangle in exactly three ways —
        as its first leg, its second, or its closing edge:

            (A,B) = (u,v)    some c with  v→c  and  u→c
            (B,C) = (u,v)    some x with  x→u  and  x→v
            (A,C) = (u,v)    some m with  u→m  and  m→v

        Each is a lookup in the endpoints' adjacency, so the work is bounded
        by their degree instead of the predicate's whole extent. A triangle is
        counted by whichever edge closes it, once: the two earlier edges could
        not see it, because the third was not there yet.
        """
        out, inn, seen = (self._out[predicate], self._in[predicate],
                          self._seen[predicate])
        if subject == value or (subject, value) in seen:
            return              # a repetition is not a new triangle
        u_out = out.get(subject, set())
        triangles = sum(1 for c in out.get(value, ()) if c in u_out and c != subject)
        triangles += sum(1 for x in inn.get(subject, ()) if value in out.get(x, ()) and x != value)
        triangles += sum(1 for m in u_out if value in out.get(m, ()) and m != value)
        seen.add((subject, value))
        out.setdefault(subject, set()).add(value)
        inn.setdefault(value, set()).add(subject)
        self._count[predicate] = self._count.get(predicate, 0) + triangles
