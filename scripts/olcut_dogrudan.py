"""Grafın bildiği bir olguyu DOĞRUDAN sorar: biliyor musun, evet mi hayır mı.

`olcut.py` açık uçlu soruyor ("köşe nasıldır") ve beklediği olgunun cevapta
GEÇMESİNİ bekliyor. O ölçü, graf büyüdükçe yapısal olarak düşüyor — kavram
başına olgu artınca belirli birinin ilk sekizde çıkma olasılığı azalıyor.
Ölçüldü ve bu projeyi uzun süre yanılttı:

    graf 140k -> 252k     olcut  %95,8 -> %75,0
    aynı olgular DOĞRUDAN sorulduğunda:
                          140k   %96,0 doğrulandı · 0 yanlış
                          252k   %96,0 doğrulandı · 0 yanlış

Yani kaybolan bilgi değil, DÖKÜMDE GÖRÜNME ŞANSI. İki ölçü iki ayrı şey
soruyor ve ikisi de gerekli:

    olcut.py           söylüyor mu   — cevabın seçimi iyi mi
    olcut_dogrudan.py  biliyor mu    — grafta duran şey erişilebilir mi

İkincisi düşerse gerçek bir gerileme vardır. Birincisi tek başına düşerse
sorulacak soru "hangi sekizini söylüyoruz" olur, "ne kaybettik" değil.

Kullanım:
    python3 scripts/olcut_dogrudan.py [--graf ...] [--sayi 300] [--tohum 7]
"""
import collections
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm import phrasing                                    # noqa: E402
from lmm.cli import Session                                 # noqa: E402
from lmm.memory import Memory                               # noqa: E402
from lmm.relations import CAN, CANNOT, HAS_PROPERTY, IS_A   # noqa: E402


def question_for(edge):
    """Olguyu, sistemin kendi ağzından bir evet/hayır sorusuna çevirir."""
    if edge.relation == IS_A:
        return f"{edge.concept} bir {edge.target} mıdır"
    if edge.relation in (CAN, CANNOT):
        return phrasing.ability_question(edge.concept, edge.target)
    return phrasing.property_question(edge.concept, edge.target)


def run(graph, count, seed):
    memory = Memory.load(graph)
    edges = [edge for edge in memory.edges
             if edge.relation in (IS_A, CAN, CANNOT, HAS_PROPERTY)
             and edge.target]
    sample = random.Random(seed).sample(edges, min(count, len(edges)))
    working = os.path.join(tempfile.mkdtemp(), "dogrudan.lmm")
    shutil.copy(graph, working)
    session = Session(working)
    before = len(session.memory.edges)
    tally = collections.Counter()
    wrong = []
    for edge in sample:
        said = session.respond(question_for(edge)).lower()
        expected = edge.relation != CANNOT
        if said.startswith("evet"):
            tally["DOĞRULADI" if expected else "YANLIŞ"] += 1
            if not expected:
                wrong.append((edge, said))
        elif said.startswith("hayır"):
            tally["DOĞRULADI" if not expected else "YANLIŞ"] += 1
            if expected:
                wrong.append((edge, said))
        else:
            tally["bilmiyor"] += 1
    return tally, len(sample), len(session.memory.edges) - before, wrong


def main(argv):
    graph = (argv[argv.index("--graf") + 1] if "--graf" in argv
             else "models/graph/birlesik.lmm")
    count = int(argv[argv.index("--sayi") + 1]) if "--sayi" in argv else 300
    seed = int(argv[argv.index("--tohum") + 1]) if "--tohum" in argv else 7
    if not os.path.exists(graph):
        print(f"yok: {graph}")
        return 1
    tally, total, written, wrong = run(graph, count, seed)
    print(f"{total} olgu doğrudan soruldu, tohum {seed}\n")
    for name in ("DOĞRULADI", "YANLIŞ", "bilmiyor"):
        print(f"  {name:<11} {tally[name]:>4}  %{tally[name] / total * 100:5.1f}")
    print(f"\n  hafızaya yazılan {written}  "
          f"<- {'TEMİZ' if written == 0 else 'KİRLENDİ'}")
    for edge, said in wrong[:4]:
        print(f"    YANLIŞ: {edge.concept} {edge.relation} {edge.target}"
              f"  -> {said[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
