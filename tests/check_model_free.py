"""Does the invariant suite really run with no model? — run it and see.

`test_core.py` promises it is "pure python, NO model" and model-free by
construction. That promise is easy to break without noticing, because on a
machine where a local model happens to be installed a test that reaches an
engine still passes: the model answers, and the answer happens to satisfy
the assertion. Nothing fails where the author works.

This runs the whole suite as a machine with NO model installed would —
`transformers`, `torch` and `llama_cpp` cannot be imported and no backend
is chosen — and lists every invariant that fails there.

    python3 tests/check_model_free.py

Measured when it was written (2026-09-27): 263 of 266 pass. The three that
do not are older tests that stub some engine readings and not others —
W147 (`phrasings`, `things_of` in the counting organ), W161 (the plan
rescue and the refusal's language reading) and W151, whose purpose is to
count real engine calls during deep ingestion. They are listed rather than
patched, because a stub chosen in a hurry can change what a test asserts.
"""
import builtins
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
os.chdir(ROOT)
os.environ.pop("LMM_BACKEND", None)

_real_import = builtins.__import__


def _no_models(name, *args, **kwargs):
    if name.split(".")[0] in ("transformers", "torch", "llama_cpp"):
        raise ImportError("no model on this machine (simulated): %s" % name)
    return _real_import(name, *args, **kwargs)


builtins.__import__ = _no_models

import test_core                                            # noqa: E402

failed = []
for name, function in test_core.PASSED:
    try:
        function()
    except Exception as broke:                              # noqa: BLE001
        failed.append((name, repr(broke)[:120]))

print("\nwith no model installed: %d / %d pass"
      % (len(test_core.PASSED) - len(failed), len(test_core.PASSED)))
for name, why in failed:
    print("  NEEDS A MODEL  %s\n                 %s" % (name, why))
sys.exit(1 if failed else 0)
