"""The LMM layer: Qwen (language) + graph/gate (truth & growth).

Backbone rule — the same as v3/gate.py's "geometry cannot write records":
QWEN CANNOT WRITE RECORDS. Qwen only produces CANDIDATES (text, extraction,
answers); the GATE decides what enters memory and what gets spoken. This one
sentence is the spine of the whole module.

Reused core (unchanged): v3/memory.py · v3/gate.py · v3/geometry.py ·
v3/dynamics.py.
"""
