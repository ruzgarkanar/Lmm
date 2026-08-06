"""Bağlantı denetimi: her dosya mimarinin neresine oturuyor, kim kimi çağırıyor.

Bu proje boyunca en pahalı hatalar ölçülmediği için değil, GÖRÜLMEDİĞİ için
oluştu: kurulmuş ama hiç çağrılmayan organlar, birbirinden habersiz iki yol,
sessizce ölmüş bir kapı. Bir dosya yazıldıktan sonra sorulacak soru şudur —
bu dosya gerçekten bağlı mı, yoksa yalnız duruyor mu.

Dört şey denetlenir ve dördü de sayıyla döner:

    BAĞLANTI   hangi modül hangisini içe aktarıyor · kimse çağırmıyorsa ölü
    KULLANIM   her genel işlev başka bir dosyadan çağrılıyor mu
    TEMİZLİK   Türkçe tanımlayıcı · kelime listesi · harf kuralı — hepsi 0 olmalı
    KATMAN     mimarideki yeri: bellek · sürekli · kapı · dinamik · ağ

Kullanım:
    python3 -m v3.check
"""
import ast
import os
import re

TURKISH = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
VOWEL_RULE = re.compile(r"[aeıioöuü]{5,}")

# Mimarideki katmanlar. Sıra, bağımlılığın hangi yöne akması gerektiğini
# söyler: aşağıdaki yukarıdakini çağırabilir, tersi mimari kırılmasıdır.
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
    """Belge dizgileri dışında Türkçe metin — açıklama serbest, veri değil."""
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
    """Üç ve daha çok dizgi taşıyan koleksiyon — kelime listesi şüphesi."""
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
    print(f"\n=== {len(held)} modül ===\n")

    # Kim kimi çağırıyor
    graph = {name: imports_of(tree) & set(held)
             for name, (_, tree) in held.items()}
    used_by = {name: set() for name in held}
    for name, targets in graph.items():
        for target in targets:
            if target in used_by:
                used_by[target].add(name)

    print("  modül        katman   çağırdıkları              çağıranlar")
    for name in sorted(held, key=lambda one: (LAYERS.index(one)
                                              if one in LAYERS else 99)):
        layer = LAYERS.index(name) if name in LAYERS else -1
        calls = ", ".join(sorted(graph[name])) or "—"
        callers = ", ".join(sorted(used_by[name])) or "—"
        print(f"  {name:<12} {layer:>4}     {calls:<24}  {callers}")

    # Mimari yönü: aşağı katman yukarıyı çağırmalı, tersi kırılma
    print("\n  YÖN DENETİMİ")
    broken = 0
    for name, targets in graph.items():
        if name not in LAYERS:
            continue
        for target in targets:
            if target in LAYERS and LAYERS.index(target) > LAYERS.index(name):
                print(f"    TERS BAĞIMLILIK: {name} -> {target}")
                broken += 1
    print(f"    ters bağımlılık: {broken}")

    # Ölü modül ve kullanılmayan işlev
    print("\n  BAĞLANTI")
    every_call = set()
    for _, (_, tree) in held.items():
        every_call |= called_names(tree)
    dead = [name for name in held
            if not used_by[name] and name not in ("check", "session")]
    print(f"    hiç çağrılmayan modül: {dead or '—'}")
    for name, (_, tree) in sorted(held.items()):
        if name == "check":
            continue
        unused = sorted(public_names(tree) - every_call)
        if unused:
            print(f"    {name}: dışarıdan çağrılmayan {unused}")

    # Temizlik
    print("\n  TEMİZLİK (hepsi 0 olmalı)")
    totals = {"türkçe dizgi": 0, "türkçe ad": 0, "kelime listesi": 0,
              "harf kuralı": 0}
    for name, (path, tree) in sorted(held.items()):
        if name == "check":
            continue
        strings = turkish_in(tree)
        names = turkish_identifiers(tree)
        lists = word_lists(tree)
        rules = len([one for one in VOWEL_RULE.findall(open(path).read())])
        totals["türkçe dizgi"] += len(strings)
        totals["türkçe ad"] += len(names)
        totals["kelime listesi"] += len(lists)
        totals["harf kuralı"] += rules
        if strings or names or lists:
            print(f"    {name}: dizgi {len(strings)} · ad {len(names)} "
                  f"· liste {len(lists)}")
            for line, items in lists:
                print(f"       satır {line}: {items}")
    for key, count in totals.items():
        mark = "TAMAM" if not count else "BAK"
        print(f"    {key:<16} {count:>3}   {mark}")
    print()
    return 0 if not any(totals.values()) and not broken else 1


if __name__ == "__main__":
    raise SystemExit(main())
