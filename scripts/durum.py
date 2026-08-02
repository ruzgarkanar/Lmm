"""Sistemin tek komutluk hâli: ne biliyor, ne kadar doğru, ne kadar hızlı.

Bu proje boyunca en pahalı hatalar ölçülmediği için görünmedi. Bir sınav
yükselirken başkası düşebiliyor, ve tek tek bakınca fark edilmiyor. Burası
hepsini bir arada ve aynı anda basıyor ki bir kazanç başka bir kaybı
gizleyemesin.

Beş ölçü, beşi de ayrı şey söylüyor:

    GRAF          ne biliyor. İlişki dağılımı önemli: yalnız `type` bilen bir
                  graf "X bir Y'dir"den başka cümle kuramaz.
    OLCUT         grafın kendisinden kurulan sınav. Sorular bizim
                  seçmediğimiz kavramlar hakkında, ezberlenemez.
    SOHBET        çok turlu. Tek soruda %95 alıp sohbette dağılmak mümkün ve
                  bu bir kez gerçekten oldu.
    GERÇEK CÜMLE  `data/tr-gercek-niyet.txt` — bizim yazmadığımız, graftan
                  gelmeyen cümleler. Kapsamın dürüst ölçüsü, ve UYDURMA
                  denetiminin tek gerçek yeri.
    HIZ           soru başına milisaniye. Graf büyüdükçe bu bozulur ve
                  bozulduğu an kimse fark etmez.

Kullanım:
    python3 scripts/durum.py [--graf models/graph/birlesik.lmm] [--hızlı]
"""
import collections
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.cli import Session                                 # noqa: E402
from lmm.intuition import UNKNOWN, UNKNOWN_WORD, AMBIGUOUS  # noqa: E402
from lmm.memory import Memory                               # noqa: E402
from lmm.phrasing import is_a_refusal                       # noqa: E402

UNREAD = (UNKNOWN, UNKNOWN_WORD, AMBIGUOUS)
UNPARSED_MARKS = ("anlamadım", "demek istedin")


def _reader():
    try:
        from core.intent import Reader
        found = Reader("cekirdek")
        return found.read if found.ready else None
    except Exception:                                       # noqa: BLE001
        return None


def graph_report(memory):
    counted = collections.Counter(edge.relation for edge in memory.edges)
    total = max(len(memory.edges), 1)
    print(f"  {len(memory.edges):>9,} olgu   {len(memory.concepts()):>8,} kavram"
          f"   {len(memory.patterns):>4} kalıp")
    for name, count in counted.most_common(6):
        print(f"    {name:<14} {count:>8,}  %{count / total * 100:5.1f}")
    sources = collections.Counter(edge.source for edge in memory.edges)
    print("    kaynaklar:", ", ".join(f"{name} {count:,}"
                                      for name, count in sources.most_common(4)))


def _exam(script, graph, extra):
    out = subprocess.run([sys.executable, f"scripts/{script}", "--graf", graph]
                         + extra, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith(("DOĞRU", "YANLIŞ", "isabet", "cevapladığ",
                            "TAŞIDI", "UNUTTU", "bağlam")):
            print(f"    {line}")


def real_sentences(graph, reader, count=150, seed=21):
    """Bizim yazmadığımız cümlelerde kapsam ve UYDURMA."""
    path = "data/tr-gercek-niyet.txt"
    if not os.path.exists(path):
        print("    (veri yok)")
        return
    rows = [line.rstrip("\n").split("\t", 1)
            for line in open(path, encoding="utf-8") if "\t" in line]
    inside = [text for kind, text in rows if kind != "DISARIDA"]
    sample = random.Random(seed).sample(inside, min(count, len(inside)))
    working = os.path.join(tempfile.mkdtemp(), "gerçek.lmm")
    shutil.copy(graph, working)
    session = Session(working, intent=reader)
    tally = collections.Counter()
    before = len(session.memory.edges)
    for text in sample:
        if session.language.understand(text).kind not in UNREAD:
            tally["okundu"] += 1
        said = session.respond(text)
        if any(mark in said.lower() for mark in UNPARSED_MARKS):
            tally["anlamadı"] += 1
        elif is_a_refusal(said):
            tally["dürüst ret"] += 1
        else:
            tally["cevapladı"] += 1
    written = len(session.memory.edges) - before
    for name in ("okundu", "cevapladı", "dürüst ret", "anlamadı"):
        print(f"    {name:<12} {tally[name]:>4}  %{tally[name] / len(sample) * 100:5.1f}")
    mark = "TEMİZ" if written == 0 else "KİRLENDİ"
    print(f"    hafızaya yazılan {written}  <- {mark}")


def speed(graph, reader, count=40, seed=5):
    memory = Memory.load(graph)
    names = [name for name in sorted(memory.concepts())
             if len(name) > 3 and " " not in name]
    random.Random(seed).shuffle(names)
    asked = ([f"{name} nedir" for name in names[:count // 2]]
             + [f"zzz{index}qq nedir" for index in range(count // 2)])
    working = os.path.join(tempfile.mkdtemp(), "hız.lmm")
    shutil.copy(graph, working)
    started = time.time()
    session = Session(working, intent=reader)
    setup = time.time() - started
    started = time.time()
    for question in asked:
        session.respond(question)
    each = (time.time() - started) / len(asked) * 1000
    print(f"    kurulum {setup:.2f} sn   soru başına {each:.0f} ms")


def main(argv):
    graph = (argv[argv.index("--graf") + 1] if "--graf" in argv
             else "models/graph/birlesik.lmm")
    quick = "--hızlı" in argv
    memory = Memory.load(graph)
    reader = _reader()
    print(f"\n=== GRAF — {os.path.basename(graph)}")
    graph_report(memory)
    print("\n=== OLCUT (graftan kurulan sınav)")
    size = "120" if quick else "300"
    for seed in ("7", "31"):
        print(f"  tohum {seed}")
        _exam("olcut.py", graph, ["--sayi", size, "--tohum", seed, "--anla"])
    print("\n=== SOHBET SINAVI (çok turlu)")
    _exam("sohbet_sinavi.py", graph,
          ["--sohbet", "30" if quick else "60", "--tohum", "7", "--anla"])
    print("\n=== GERÇEK CÜMLELER (bizim yazmadığımız)")
    real_sentences(graph, reader, count=60 if quick else 150)
    print("\n=== HIZ")
    speed(graph, reader)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
