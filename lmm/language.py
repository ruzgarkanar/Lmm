"""The language organ contract.

LMM keeps language and knowledge apart on purpose. Memory, reasoning, the
epistemic gate and curiosity never see a sentence — they work in concepts and
relations. Everything that touches Turkish sits behind this one interface.

That makes the language layer replaceable without touching the parts that make
LMM what it is. Our own `Intuition` implements it today. Anyone who wants a
fluent front end can implement it with a language model instead — and that model
would still have no authority over what is true, because truth lives in memory
and answers are built by the gate.
"""


class LanguageOrgan:
    """Turns a sentence into an Intent. That is the whole contract.

    Implementations must return an `lmm.intuition.Intent` with:
      kind        — TEACH, ASK, ASK_WHO, ASK_ABILITIES, ASK_WHY or UNKNOWN
      concept     — the thing being spoken about (None for ASK_WHO)
      relation    — IS_A, NOT_A, CAN or CANNOT, where the kind needs one
      target      — the type, action or property the relation points at
      confidence  — 0..1; below the session threshold the system says it did not
                    understand rather than guessing at a reading
    """

    def understand(self, sentence):
        raise NotImplementedError
