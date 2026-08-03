"""Grafta anlamı olmayan kayıtları siler. Elle dosya açmadan.

Bir olgu yanlış olabilir ve o bir bilgi sorunudur; ama bir olgu ANLAMSIZ da
olabilir ve o bir ayrıştırma artığıdır. İkincisi silinir, çünkü düzeltilecek
bir iddia taşımıyor.

İki sınıf, ikisi de ölçülerek bulundu:

    OLUMSUZLUK HEDEFİ   `meslekler kolay --property--> değil`
                        "meslekler kolay değildir" cümlesi eski bir sürümde
                        böyle okunmuştu: niteleme özneye yapışmış, olumsuzluk
                        nitelik sanılmış. Bugünün ayrıştırıcısı doğru okuyor
                        (`meslek --not_property--> kolay`), ama eski kayıtlar
                        duruyor ve merak organı onlar hakkında soru soruyor:
                        "meslekler kolay nedir?" — ve bir dil modeli o olmayan
                        şeye tanım uyduruyor. Kötü kayıt, kötü sorunun
                        kaynağı; kötü soru, kötü cevabın.

    KENDİNE İŞARET      `X --type--> X`. Bir şey kendi türü olamaz.

Silme SAYILIR ve gösterilir; `--yaz` verilmeden hiçbir şey değişmez.

Kullanım:
    python3 scripts/graf_temizle.py [--graf ...] [--yaz]
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.memory import Memory                               # noqa: E402
from lmm.relations import IS_A                              # noqa: E402
from lmm.turkish import TurkishMorphology                   # noqa: E402


def junk(memory):
    """Silinecek kenarlar, sebebiyle birlikte."""
    denials = set(getattr(TurkishMorphology, "denials", ()))
    found = []
    for edge in memory.edges:
        target = str(edge.target or "")
        if target in denials:
            found.append((edge, "olumsuzluk hedefi"))
        elif edge.relation == IS_A and target == edge.concept:
            found.append((edge, "kendine işaret"))
    return found


def orphaned(memory, removed):
    """Silme sonrası hiç olgusu kalmayan kavramlar — merak onlara sormasın."""
    gone = {id(edge) for edge, _ in removed}
    left = collections.Counter()
    for edge in memory.edges:
        if id(edge) not in gone:
            left[edge.concept] += 1
    return [edge.concept for edge, _ in removed if not left[edge.concept]]


def main(argv):
    graph = (argv[argv.index("--graf") + 1] if "--graf" in argv
             else "models/graph/birlesik.lmm")
    if not os.path.exists(graph):
        print(f"yok: {graph}")
        return 1
    memory = Memory.load(graph)
    found = junk(memory)
    print(f"{os.path.basename(graph)} — {len(memory.edges):,} olgu")
    if not found:
        print("  temiz.")
        return 0
    counted = collections.Counter(why for _, why in found)
    for why, count in counted.most_common():
        print(f"  {why:<20} {count:>5}")
    for edge, why in found[:8]:
        print(f"    {edge.concept} --{edge.relation}--> {edge.target}"
              f"  ({edge.source})")
    left = orphaned(memory, found)
    if left:
        print(f"\n  silince olgusuz kalacak kavram: {len(left)}  {left[:6]}")

    if "--yaz" not in argv:
        print("\n  (silmek için --yaz ekle)")
        return 0
    gone = {id(edge) for edge, _ in found}
    kept = [edge for edge in memory.edges if id(edge) not in gone]
    # Yeniden kurmak, indeksleri elle onarmaktan güvenli: bellek kendi
    # indekslerini yazarken kuruyor ve elle dokunmak ikinci bir doğru yaratır.
    fresh = Memory(graph)
    fresh.patterns = memory.patterns
    fresh.vocabulary = memory.vocabulary
    fresh.kinds = memory.kinds
    for edge in kept:
        try:
            fresh.write(edge)
        except Exception:                                   # noqa: BLE001
            continue
    fresh.save(graph)
    print(f"\n  {len(found)} kayıt silindi -> {graph}"
          f"  ({len(fresh.edges):,} olgu kaldı)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
