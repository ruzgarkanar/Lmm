"""Kelime etiketleri çıkarır: hangi kelime kavram, hangisi hedef.

Bugüne kadar ağ cümleyi **tek bir sınıfa** indiriyordu (13 niyet) ve kavramı
"ilk tanınan kelime" kuralıyla seçiyorduk. Ölçüldü ve yetmiyor: "penguenin
özellikleri neler" sorusunda kavram olarak `özellik` seçiliyordu, çünkü o da
grafta bir düğüm.

Sorun kuralda değil, **bilginin nereden geldiğinde**: kavramın hangi kelime
olduğu cümlenin bütününden anlaşılır, tek tek kelimelere bakarak değil. Bunu
öğrenmenin yolu diziyi etiketlemek — her konum için bir etiket — ve bunun
eğitim verisi bedava çıkarılabilir.

**Öğretmen kendi kalıp eşleştiricimiz.** Bir cümleyi ayrıştırabildiğinde hangi
kelimenin kavram olduğunu zaten biliyor. Ağ, kalıpların bildiğini öğrenir ama
kalıpların göremediği söyleyişlere genelleyerek.

Ama öğretmen kusurlu ve bunu bilerek kullanıyoruz: kalıp bazen kavramı yanlış
seçiyor (yukarıdaki örnek). O yüzden **süzgeç var** — bir ayrıştırma ancak
grafta GERÇEKTEN cevap üretiyorsa öğretmen sayılıyor. Cevap üretemeyen bir
okuma yanlış okumadır ve öğretilmemeli. Aynı ölçüt sistemin her yerinde.

Etiketler:

    KAVRAM   hakkında konuşulan şey
    HEDEF    sorulan nitelik / eylem / tür
    NESNE    ikinci kavram ("kartal SERÇEDEN hızlıdır")
    DIS      cümlenin geri kalanı

Kullanım:
    python3.11 scripts/etiket_cikar.py --girdi sorular.json --sayi 200000
"""
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.cli import Session                                 # noqa: E402
from lmm.intuition import UNKNOWN, UNKNOWN_WORD, AMBIGUOUS, tokenize  # noqa: E402
from lmm.serialize import is_refusal as is_a_refusal                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from core.intent import _where                              # noqa: E402
OUTPUT = os.path.join(ROOT, "data", "tr-kelime-etiketi.txt")
GRAPH = _where("graph", "wikipedia")

OUTSIDE = "DIS"
LABELS = ("KAVRAM", "HEDEF", "NESNE", OUTSIDE)
LONGEST = 12        # daha uzun cümlede hizalama güvenilmez oluyor


def align(tokens, value, morphology):
    """Bu değeri üreten kelimenin sırası. Bulunamazsa None.

    Hizalama yaklaşık: kelime ekleriyle geçiyor, değer köküyle duruyor
    ("penguenin" -> "penguen"). Önek karşılaştırması ikisini birleştiriyor.
    """
    if not value:
        return None
    value = value.split()[-1] if " " in value else value
    for index, token in enumerate(tokens):
        if token == value or token.startswith(value):
            return index
        stripped = morphology.strip_plural(token)
        if stripped == value or stripped.startswith(value):
            return index
    return None


def labelled(session, sentence):
    """(kelimeler, etiketler) ya da öğretilemeyecekse None.

    Süzgeç burada: ayrıştırma grafta cevap üretmiyorsa yanlış okumadır ve
    öğretilmez. Kusurlu bir öğretmenden öğrenmenin tek çaresi, öğrettiğini
    denetlemek.
    """
    tokens = tokenize(sentence)
    if not 2 < len(tokens) <= LONGEST:
        return None
    intent = session.language.understand(sentence)
    if intent.kind in (UNKNOWN, UNKNOWN_WORD, AMBIGUOUS) or not intent.concept:
        return None
    if is_a_refusal(session.gate.answer(intent)):
        return None         # okuma işe yaramıyor: öğretmen olamaz

    morphology = session.language.grammar.morphology
    marks = [OUTSIDE] * len(tokens)
    at = align(tokens, intent.concept, morphology)
    if at is None:
        return None         # kavram hizalanamadı: hangi kelime olduğu belirsiz
    marks[at] = "KAVRAM"
    for value, name in ((intent.target, "HEDEF"), (intent.object, "NESNE")):
        where = align(tokens, value, morphology) if value else None
        if where is not None and marks[where] == OUTSIDE:
            marks[where] = name
    return tokens, marks


def main(argv):
    source = argv[argv.index("--girdi") + 1] if "--girdi" in argv else None
    wanted = int(argv[argv.index("--sayi") + 1]) if "--sayi" in argv else 200000
    graph = argv[argv.index("--graf") + 1] if "--graf" in argv else GRAPH
    if not source or not os.path.exists(source):
        print("kullanım: --girdi sorular.json [--sayi 200000] [--graf model.lmm]")
        return 1
    if not os.path.exists(graph):
        print(f"eksik graf: {graph}")
        return 1

    session = Session(graph)
    print(f"graf: {len(session.memory.edges)} bilgi, "
          f"{len(session.memory.concepts())} kavram")
    questions = json.load(open(source, encoding="utf-8"))
    print(f"{len(questions):,} cümle taranacak, hedef {wanted:,} etiketli")

    kept = []
    counts = collections.Counter()
    started = time.time()
    for index, sentence in enumerate(questions):
        found = labelled(session, sentence)
        if found is not None:
            kept.append(found)
            counts.update(found[1])
        if index % 100000 == 0 and index:
            print(f"  {index:,} tarandı, {len(kept):,} etiketli "
                  f"(%{len(kept)/index*100:.1f})  {time.time()-started:.0f} sn",
                  flush=True)
        if len(kept) >= wanted:
            break

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for tokens, marks in kept:
            out.write(" ".join(tokens) + "\t" + " ".join(marks) + "\n")
    print(f"\n{len(kept):,} etiketli cümle -> {OUTPUT}")
    print(f"  verim %{len(kept)/max(index,1)*100:.1f}")
    for name, count in counts.most_common():
        print(f"    {name:<8} {count:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
