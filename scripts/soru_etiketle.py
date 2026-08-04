"""Soru cümlelerini etiketler — grafla hizalayarak, elle yazmadan.

Öğrenilmiş etiketleyici bildirme cümlelerinde çalışıyor ama SORULARDA hiç
çalışmıyor, ve sebebi ölçüldü: eğitim verisinin tamamı web düzyazısıydı, yani
bildirme. Ağ hiç soru görmedi.

    "Kartal, yırtıcı bir kuş türüdür."   -> kavram=Kartal   ✓
    "kartal nedir"                        -> kavram='rtal'   ✗

Soru için etiketli veri de elle yazılmadan üretilebilir: grafın BİLDİĞİ bir
kavram soruda geçiyorsa, kavram odur. Doğrulayan şey graf — 90 bin kavramlık
bir kayıt, ve o kayıt da metinden gelmiş.

    "kartal nedir"          graf `kartal`ı tanıyor      -> kavram = kartal
    "Ormanların önemi..."   graf `orman`ı tanıyor       -> kavram = ormanların

Hangi kavramın seçileceği de sorulmuyor: cümledeki, graf hakkında EN ÇOK
BİLDİĞİ ad alınıyor. Bu bir dil kuralı değil, bir sayım.

Kullanım:
    python3 scripts/soru_etiketle.py data/tr-soru-adaylari.txt [...] --yaz sorular.jsonl
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.memory import Memory                               # noqa: E402
from lmm.intuition import tokenize                          # noqa: E402
from lmm import frequency                                   # noqa: E402
from lmm.verbs import is_structural                         # noqa: E402

# Grafın bir adı "biliyor" sayması için kaç olgusu olmalı. Altında seçim
# gürültü: bir kez geçen bir ad, sorunun konusu olmayabilir.
LEAST_FACTS = 3


def lines_of(path):
    """Satır ya da 'etiket<TAB>cümle' — ikisi de kabul."""
    for line in open(path, encoding="utf-8", errors="ignore"):
        text = line.rstrip("\n")
        if "\t" in text:
            text = text.split("\t", 1)[1]
        text = text.strip()
        if 8 < len(text) < 300:
            yield text


def subject_of(sentence, memory, counts):
    """Cümlenin konusu: grafın EN ÇOK bildiği ad.

    Baskınlık aranıyor — iki ad yarışıyorsa hangisinin sorulduğu belli
    değildir ve belirsiz örnek öğretmez.
    """
    scored = []
    for word in set(tokenize(sentence)):
        if is_structural(word, counts):
            continue
        size = len(memory.query(word))
        if size >= LEAST_FACTS:
            scored.append((size, word))
    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[1][0] * 2 > scored[0][0]:
        return None
    return scored[0][1]


def main(argv):
    paths = [a for a in argv[1:] if not a.startswith("--") and os.path.exists(a)]
    out = (argv[argv.index("--yaz") + 1] if "--yaz" in argv
           else "sorular.jsonl")
    graph = (argv[argv.index("--graf") + 1] if "--graf" in argv
             else "models/graph/birlesik.lmm")
    if not paths:
        print(__doc__.strip().splitlines()[-1])
        return 1

    memory = Memory.load(graph)
    counts = frequency.counts()
    seen = set()
    found = 0
    with open(out, "w", encoding="utf-8") as handle:
        for path in paths:
            for sentence in lines_of(path):
                if sentence in seen:
                    continue
                seen.add(sentence)
                subject = subject_of(sentence, memory, counts)
                if not subject:
                    continue
                # Okuma çıktısıyla AYNI biçim: aynı hattan geçsin diye.
                handle.write(json.dumps({"cümle": sentence,
                                         "kavram": subject,
                                         "ilişki": "soru",
                                         "hedef": ""},
                                        ensure_ascii=False) + "\n")
                found += 1
    print(f"  {len(seen):,} cümle tarandı -> {found:,} etiketlendi "
          f"(%{found / max(len(seen), 1) * 100:.1f})")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
