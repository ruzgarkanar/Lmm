"""Grafı onarır: tekil/çoğul ikizlerini birleştirir.

Ölçüldü: 16 bin kavramlı grafta **101 kavramın hem tekili hem çoğulu ayrı
düğüm** olarak duruyor — `araba` (8 olgu) ve `arabalar` (1 olgu), `balina` (6)
ve `balinalar` (1). Graf sessizce ikiye bölünüyor: aynı şey hakkında bilinenler
iki yere dağılıyor ve hiçbir soru ikisini birden göremiyor.

Kaynağı belli: ansiklopedi başlığı çoğul olabiliyor ("Alkaloidler", "Belgeler")
ve başlık doğrudan kavram yapılıyor. Okuyucu cümle içindeki çoğulu zaten
soyuyor; soyulmayan yer başlık.

Birleştirme yönü **tekile doğru**, çünkü kalıtım tekil üzerinden kuruluyor:
"penguen bir kuştur" yazılırken "kuş" aranıyor, "kuşlar" değil.

Karar kurala değil sayıma bırakılıyor: tekil derlemde çoğulu kadar sık
geçiyorsa ek gerçektir — `case_of`'un yıllardır kullandığı testin aynısı.
"popüler" kelimesinin "popü"ye indirilmemesini sağlayan da bu.

Kullanım:
    python3 scripts/graf_onar.py models/graph/birlesik.lmm [--yaz]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm import frequency                                   # noqa: E402
from lmm.memory import Edge, Memory                         # noqa: E402

PLURAL = ("lar", "ler")


def twins(memory, counts):
    """{çoğul: tekil} — ikisi de grafta duran ve birleşmesi gereken çiftler."""
    known = set(memory.concepts())
    found = {}
    for name in known:
        for ending in PLURAL:
            if not name.endswith(ending) or len(name) <= len(ending) + 2:
                continue
            single = name[: -len(ending)]
            if single not in known:
                continue
            # Derlem ikinci tanık: tekil, çoğulu kadar sık geçmeli. "bunlar"
            # gibi bir kelimede "bun" seyrektir ve ikisi ikiz sayılmaz.
            if counts and counts.get(single, 0) < counts.get(name, 0):
                continue
            found[name] = single
    return found


def merged(memory, pairs):
    """Çoğul düğümdeki kayıtları tekile taşıyan yeni bir bellek."""
    fresh = Memory(lexicon=memory.lexicon)
    fresh.patterns = list(memory.patterns)
    moved = dropped = 0
    for edge in memory.edges:
        concept = pairs.get(edge.concept, edge.concept)
        target = pairs.get(edge.target, edge.target) if edge.target else edge.target
        if concept != edge.concept or target != edge.target:
            moved += 1
        if concept == target:
            dropped += 1        # "araba bir arabadır" — birleşmenin yan ürünü
            continue
        try:
            fresh.write(Edge(concept, edge.relation, target,
                             object=edge.object, role=edge.role,
                             source=edge.source, confidence=edge.confidence,
                             is_exception=edge.is_exception,
                             quantifier=edge.quantifier))
        except Exception:                                   # noqa: BLE001
            dropped += 1
    return fresh, moved, dropped


def main(argv):
    given = [a for a in argv[1:] if not a.startswith("--")]
    if not given:
        print(__doc__.strip().splitlines()[-1])
        return 1
    path = given[0]
    if not os.path.exists(path):
        print(f"yok: {path}")
        return 1

    memory = Memory.load(path)
    counts = frequency.counts()
    pairs = twins(memory, counts)
    print(f"{os.path.basename(path)} — {len(memory.edges):,} olgu, "
          f"{len(memory.concepts()):,} kavram")
    print(f"{len(pairs)} ikiz bulundu")
    for plural, single in sorted(pairs.items())[:8]:
        print(f"    {plural:<20} -> {single:<16} "
              f"({len(memory.query(plural))} + {len(memory.query(single))} olgu)")

    fresh, moved, dropped = merged(memory, pairs)
    print(f"\n  {moved} kayıt taşındı, {dropped} eridi")
    print(f"  {len(fresh.edges):,} olgu, {len(fresh.concepts()):,} kavram")
    if "--yaz" not in argv:
        print("\n  (yazmak için --yaz ekle)")
        return 0
    fresh.save(path)
    print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
