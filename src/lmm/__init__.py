"""LMM — Living Memory Model.

A memory-and-reasoning layer that wraps a language model. Knowledge lives in a
verifiable graph plus an evidence index — writable at any moment, persistent,
source-stamped — instead of being frozen into weights.

    from lmm import Memory

    m = Memory("mind.lmm")
    m.learn("manual.pdf")
    print(m.ask("what is the screen's diagonal?"))
    m.save()

Backbone rule — the same as `lmm.core.gate`'s "geometry cannot write records":
THE ENGINE CANNOT WRITE RECORDS. It only produces CANDIDATES (text, extraction,
answers); the GATE decides what enters memory and what gets spoken. That one
sentence is the spine of the whole package.

Layout: `lmm.core` is the discrete substrate (memory · gate · geometry ·
dynamics · transitivity — pure python, no model, no network). The modules
beside it are the layer that gives it language.

Names are exported LAZILY. `import lmm` must stay cheap and must not drag in an
engine, so the heavy modules are imported on first attribute access.
"""
__version__ = "0.4.0"

__all__ = ["Memory", "Answer", "Learned", "Session", "__version__"]

_LAZY = {
    "Memory": ("lmm.api", "Memory"),
    "Answer": ("lmm.api", "Answer"),
    "Learned": ("lmm.api", "Learned"),
    "Session": ("lmm.session", "Session"),
}


def __getattr__(name):
    """PEP 562 lazy export — see the module docstring."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'lmm' has no attribute {name!r}")
    import importlib                                        # noqa: PLC0415
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value                     # bind, so this runs once
    return value


def __dir__():
    return sorted(__all__)
