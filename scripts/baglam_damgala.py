"""Grafta duran olgulara geldikleri CÜMLENİN kimliğini basar.

Anlam ayrımının en güçlü sinyali provenanstaydı ve atılmıştı. Aynı cümleden
çıkan olgular aynı ANLAMA aittir — ölçüldü:

    tavla   cümle 1 -> oyun, iki, zar, pul, platform      (oyun)
            cümle 2 -> mahalle, hatay, defne              (mahalle)
    kartal  cümle 1 -> ilçe, banliyö                      (İstanbul)
            cümle 3 -> kasaba, kurulu, nüfus, alan        (Macaristan)

Vektörle ayırmak da denendi: 20 bin cümlelik gömme altı nitelikten beşini
doğru ayırdı, `hızlı`yı kaçırdı ve marjlar zayıftı. Künye hiçbir tahmin
gerektirmiyor — geometri değil PROVENANS.

Bu betik yeniden alım yapmıyor; okuma çıktısıyla eşleştirip damgalıyor.
Yeniden alım elli dakika, damgalama dakikalar.

Eşleşme (kavram, ilişki, hedef) üçlüsünden. Hedef alım sırasında
normalleşebiliyor ("gölü" -> "göl", "semttir" -> "semt"), o yüzden birebir
eşleşmeyenler için hedefin ilk harfleriyle ikinci bir tur yapılıyor.

Kullanım:
    python3 scripts/baglam_damgala.py okuma.jsonl [--graf ...] [--yaz]
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.memory import Memory, context_mark                 # noqa: E402


def contexts(path):
    """(kavram, ilişki, hedef) -> cümle kimliği."""
    exact, loose = {}, {}
    for line in open(path, encoding="utf-8"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        concept = (row.get("kavram") or "").strip().lower()
        relation = (row.get("ilişki") or "").strip()
        target = (row.get("hedef") or "").strip().lower()
        sentence = row.get("cümle")
        if not (concept and relation and target and sentence):
            continue
        mark = context_mark(sentence)
        exact.setdefault((concept, relation, target), mark)
        loose.setdefault((concept, relation, target[:4]), mark)
    return exact, loose


def main(argv):
    given = [a for a in argv[1:] if not a.startswith("--")]
    if not given:
        print(__doc__.strip().splitlines()[-1])
        return 1
    path = given[0]
    graph = (argv[argv.index("--graf") + 1] if "--graf" in argv
             else "models/graph/birlesik.lmm")
    if not os.path.exists(path) or not os.path.exists(graph):
        print("dosya yok")
        return 1

    exact, loose = contexts(path)
    memory = Memory.load(graph)
    print(f"{os.path.basename(path)} — {len(exact):,} okuma")
    print(f"{os.path.basename(graph)} — {len(memory.edges):,} olgu\n")

    tally = collections.Counter()
    for edge in memory.edges:
        if edge.context is not None:
            tally["zaten damgalı"] += 1
            continue
        key = (edge.concept, edge.relation, str(edge.target).lower())
        mark = exact.get(key)
        if mark is None:
            mark = loose.get((edge.concept, edge.relation,
                              str(edge.target).lower()[:4]))
            if mark is not None:
                tally["gevşek eşleşti"] += 1
        else:
            tally["birebir eşleşti"] += 1
        if mark is None:
            tally["eşleşmedi"] += 1
            continue
        edge.context = mark

    total = max(len(memory.edges), 1)
    for name, count in tally.most_common():
        print(f"  {name:<18} {count:>8,}  %{count / total * 100:5.1f}")

    # Kaç kavramın olguları birden çok cümleye dağılıyor — anlam ayrımının
    # işe yarayacağı yerler tam bunlar.
    spread = collections.defaultdict(set)
    for edge in memory.edges:
        if edge.context is not None:
            spread[edge.concept].add(edge.context)
    many = sum(1 for marks in spread.values() if len(marks) > 1)
    print(f"\n  birden çok cümleden beslenen kavram: {many:,}"
          f" / {len(spread):,}")

    if "--yaz" not in argv:
        print("\n  (yazmak için --yaz ekle)")
        return 0
    memory.save(graph)
    print(f"\n  -> {graph}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
