"""The LMM layer: Qwen (language) + graph/gate (truth & growth).

Backbone rule — the same as lmm/core/gate.py's "geometry cannot write records":
QWEN CANNOT WRITE RECORDS. Qwen only produces CANDIDATES (text, extraction,
answers); the GATE decides what enters memory and what gets spoken. This one
sentence is the spine of the whole module.

Reused core (unchanged): lmm/core/memory.py · lmm/core/gate.py · lmm/core/geometry.py ·
lmm/core/dynamics.py.
"""
