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

Measured when it was written (2026-09-27): 263 of 266 passed. The three
that did not were older tests that stubbed some engine readings and not
others — W147, W161 and W151 — and they were also why the public CI had
been red on every push since 24 September. Each was then fixed by stubbing
only what the test was not about, and each fix was checked three ways: it
passes with a model, it passes without one, and it still FAILS when the
behaviour it guards is broken on purpose. Now: 266 of 266.
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
