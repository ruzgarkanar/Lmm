"""The wiring audit: where each file sits in the architecture, who calls whom.

Throughout this project the most expensive mistakes happened not because they
were unmeasured but because they were UNSEEN: organs built but never called,
two paths unaware of each other, a gate that had died silently. The question
to ask after a file is written is this — is this file really wired in, or is
it merely standing there.

Four things are audited and all four return as numbers:

    WIRING     which module imports which · if nobody calls it, it is dead
    USAGE      is every public function called from another file
    CLEANLINESS Turkish identifier · word list · letter rule — all must be 0

THE ONE THING THAT CANNOT REACH 0 is a data file's own FIELD NAMES: dataset.py
reads a Turkish JSONL whose keys are Turkish words, and renaming them here
would simply stop the file from loading. The audit counts them because it
cannot tell a key from a sentence; two is the count, and it is the whole
remainder. Everything else — printed output, examples, identifiers — is in the
codebase's own language, so that nothing the SYSTEM says or does depends on
one user language.
    LAYER      its place in the architecture: memory · continuous · gate · dynamics · network

Usage:
    python3 -m v3.check
"""
import ast
import os
import re

TURKISH = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
VOWEL_RULE = re.compile(r"[aeıioöuü]{5,}")

# The layers in the architecture. The order says which way dependency must
# flow: the lower may call the upper, the reverse is an architectural break.
LAYERS = ["memory", "geometry", "gate", "dynamics", "reader", "speaker",
          "session"]


def modules(folder="v3"):
    found = {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".py") or name.startswith("__"):
            continue
        path = os.path.join(folder, name)
        found[name[:-3]] = (path, ast.parse(open(path, encoding="utf-8").read()))
    return found


def imports_of(tree, package="v3"):
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(package):
                piece = node.module.split(".")[-1]
                if piece != package:
                    found.add(piece)
                for alias in node.names:
                    found.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(package + "."):
                    found.add(alias.name.split(".")[-1])
    return found


def public_names(tree):
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) \
                and not node.name.startswith("_"):
            found.add(node.name)
    return found


def called_names(tree):
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Name):
            found.add(node.id)
    return found


def turkish_in(tree):
    """Turkish text outside docstrings — commentary is free, data is not."""
    docs = {ast.get_docstring(node, clean=False) for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef,
                                 ast.AsyncFunctionDef))}
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value not in docs and TURKISH.search(node.value)]


def turkish_identifiers(tree):
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            found.add(node.id)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
    return [name for name in found if TURKISH.search(name)]


def word_lists(tree):
    """A collection carrying three or more strings — suspected word list."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = [one.value for one in node.elts
                     if isinstance(one, ast.Constant)
                     and isinstance(one.value, str)]
            if len(items) >= 3 and any(TURKISH.search(one) for one in items):
                found.append((node.lineno, items[:5]))
    return found


def main():
    held = modules()
    print(f"\n=== {len(held)} modules ===\n")

    # Who calls whom
    graph = {name: imports_of(tree) & set(held)
             for name, (_, tree) in held.items()}
    used_by = {name: set() for name in held}
    for name, targets in graph.items():
        for target in targets:
            if target in used_by:
                used_by[target].add(name)

    print("  module       layer    calls                     called by")
    for name in sorted(held, key=lambda one: (LAYERS.index(one)
                                              if one in LAYERS else 99)):
        layer = LAYERS.index(name) if name in LAYERS else -1
        calls = ", ".join(sorted(graph[name])) or "—"
        callers = ", ".join(sorted(used_by[name])) or "—"
        print(f"  {name:<12} {layer:>4}     {calls:<24}  {callers}")

    # Architectural direction: the lower layer must call the upper, the
    # reverse is a break
    print("\n  DIRECTION AUDIT")
    broken = 0
    for name, targets in graph.items():
        if name not in LAYERS:
            continue
        for target in targets:
            if target in LAYERS and LAYERS.index(target) > LAYERS.index(name):
                print(f"    REVERSED DEPENDENCY: {name} -> {target}")
                broken += 1
    print(f"    reversed dependencies: {broken}")

    # Dead modules and unused functions
    print("\n  WIRING")
    every_call = set()
    for _, (_, tree) in held.items():
        every_call |= called_names(tree)
    dead = [name for name in held
            if not used_by[name] and name not in ("check", "session")]
    print(f"    never called: {dead or '—'}")
    for name, (_, tree) in sorted(held.items()):
        if name == "check":
            continue
        unused = sorted(public_names(tree) - every_call)
        if unused:
            print(f"    {name}: never called from outside {unused}")

    # Cleanliness
    print("\n  CLEANLINESS (every count must be 0)")
    totals = {"turkish string": 0, "turkish name": 0, "word list": 0,
              "letter rule": 0}
    for name, (path, tree) in sorted(held.items()):
        if name == "check":
            continue
        strings = turkish_in(tree)
        names = turkish_identifiers(tree)
        lists = word_lists(tree)
        rules = len([one for one in VOWEL_RULE.findall(open(path).read())])
        totals["turkish string"] += len(strings)
        totals["turkish name"] += len(names)
        totals["word list"] += len(lists)
        totals["letter rule"] += rules
        if strings or names or lists:
            print(f"    {name}: strings {len(strings)} · names {len(names)} "
                  f"· lists {len(lists)}")
            for line, items in lists:
                print(f"       line {line}: {items}")
    for key, count in totals.items():
        mark = "OK" if not count else "LOOK"
        print(f"    {key:<16} {count:>3}   {mark}")
    print()
    return 0 if not any(totals.values()) and not broken else 1


if __name__ == "__main__":
    raise SystemExit(main())
