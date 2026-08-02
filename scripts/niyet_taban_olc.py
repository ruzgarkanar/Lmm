"""Elle yazılmış şekil kodlayıcısının TABANINI ölçer — eğitmez.

Üretilmiş cümlelerden damıtma iki kez denendi ve ikisinde de gerçek işi
düşürdü — şekil özellikleriyle %72'den %56'ya, kimlik korunan özelliklerle
%39'a. Teşhis her seferinde aynıydı: **dağılım gerçek değil.** Dil modeline
"çeşitli cümleler yaz" demek, insanların gerçekten nasıl sorduğunu vermiyor.

Oysa 30 GB Türkçe derlemin içinde milyonlarca gerçek soru duruyor ve dört
saniyede altmış bin tanesi çıkarılabiliyor. Onlar üretilmiş değil; insanlar
yazmış.

Bu, LLM'i yapan fikrin bize uyarlanmış hâli: **denetim sinyali bedava ve
gerçek olmalı.** Sonraki-token tahmini yazılmış her cümleyi eğitim verisine
çeviriyor; burada da veri üretilmiyor, bulunuyor. Etiketi dil modeli koyuyor
ama cümleyi insan yazmış — ve ölçülen fark tam da orada.

Sınıflardan biri DIŞARIDA: gerçek soruların çoğu bizim niyetlerimizden hiçbiri
değil ve sınıflandırıcı bunu bilmek zorunda. "Ne oluyor annecim?" sorusuna
niyet uydurmak, cevap uydurmaktan farksız.

Bu betik bir eğitim betiği değil; sinir ağına geçmeden önce aşılması
gereken çizgiyi (%68,8) veren ölçüm. Eğitim `scripts/niyet_egitim.py`.

Kullanım: python3 scripts/niyet_taban_olc.py
"""
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm import frequency                                   # noqa: E402
from lmm.intuition import tokenize                          # noqa: E402
from lmm.lexicon import Lexicon                             # noqa: E402
from lmm.network import CLASSES, MiniNetwork, features      # noqa: E402
from lmm.verbs import discover                              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_SOURCE = os.path.join(ROOT, "data", "tr-gercek-niyet.txt")
MADE_SOURCE = os.path.join(ROOT, "data", "tr-niyet.txt")

OUTSIDE = "DISARIDA"
EVERY = list(CLASSES) + [OUTSIDE]

# İnsanın yazdığı sınav — iki üreticinin de hiç görmediği cümleler.
EXAM = [
    ("kartal nedir", "ASK_DEFINITION"),
    ("penguen uçar mı", "ASK_ABILITY"),
    ("kuşlar ne yapabilir", "ASK_ABILITIES"),
    ("bana penguenlerden bahseder misin", "ASK_DESCRIBE"),
    ("penguen neden uçamıyor", "ASK_WHY"),
    ("kuşların özellikleri nelerdir", "ASK_PROPERTIES"),
    ("penguen anlat", "ASK_DESCRIBE"),
    ("kartal hızlı mı", "ASK_PROPERTY"),
    ("kartal nasıldır", "ASK_PROPERTIES"),
    ("penguen tüylü müdür", "ASK_PROPERTY"),
    ("bana kuşları anlatır mısın", "ASK_DESCRIBE"),
    ("acaba kartal yüzebilir mi", "ASK_ABILITY"),
    ("kartalın özellikleri neler", "ASK_PROPERTIES"),
    ("penguen neden yüzemiyor", "ASK_WHY"),
    ("balina nedir", "ASK_DEFINITION"),
    ("kartal ne yapar", "ASK_ABILITIES"),
]


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if "\t" in line:
                kind, sentence = line.rstrip("\n").split("\t", 1)
                if kind in EVERY:
                    rows.append((sentence, kind))
    return rows


def lexicon_from_corpus():
    lexicon = Lexicon()
    words = frequency.counts()
    for infinitive, positive, negative in discover(words, minimum=20):
        lexicon.learn_verb(infinitive, positive, negative)
    return lexicon


def score(network, rows, lexicon):
    right = 0
    missed = collections.Counter()
    for sentence, wanted in rows:
        got, _ = network.predict(tokenize(sentence), lexicon)
        if got == wanted:
            right += 1
        else:
            missed[(wanted, got)] += 1
    return right / max(len(rows), 1), missed


def train_on(rows, lexicon, rounds=60, rate=0.3):
    network = MiniNetwork(EVERY)
    network.train([(features(tokenize(s), lexicon), k) for s, k in rows],
                  epochs=rounds, learning_rate=rate)
    return network


def main(argv):
    real = load(REAL_SOURCE)
    made = load(MADE_SOURCE)
    if not real:
        print(f"eksik: {REAL_SOURCE}")
        return 1
    lexicon = lexicon_from_corpus()
    counts = collections.Counter(k for _, k in real)
    print(f"{len(real)} GERÇEK etiketli soru, {len(made)} üretilmiş")
    for kind, count in counts.most_common():
        print(f"  {kind:<18} {count}")

    random.Random(11).shuffle(real)
    split = int(len(real) * 0.8)
    train, held = real[:split], real[split:]

    print(f"\n{'':<24}{'ayrılmış sınav':>16}{'GERÇEK İŞ':>14}")
    today = MiniNetwork.default()
    print(f"  {'bugünkü (21 şekil)':<22}{'—':>16}"
          f"{'%' + format(score(today, EXAM, lexicon)[0]*100, '.1f'):>14}")

    if made:
        from_made = train_on(made, lexicon)
        print(f"  {'üretilmiş cümleler':<22}"
              f"{'%' + format(score(from_made, held, lexicon)[0]*100, '.1f'):>16}"
              f"{'%' + format(score(from_made, EXAM, lexicon)[0]*100, '.1f'):>14}")

    from_real = train_on(train, lexicon)
    exam_score, _ = score(from_real, held, lexicon)
    real_score, missed = score(from_real, EXAM, lexicon)
    print(f"  {'GERÇEK sorular':<22}{'%' + format(exam_score*100, '.1f'):>16}"
          f"{'%' + format(real_score*100, '.1f'):>14}")

    if made:
        both = train_on(train + made, lexicon)
        print(f"  {'ikisi birlikte':<22}"
              f"{'%' + format(score(both, held, lexicon)[0]*100, '.1f'):>16}"
              f"{'%' + format(score(both, EXAM, lexicon)[0]*100, '.1f'):>14}")

    if missed:
        print("\n  gerçek işte karışanlar:")
        for (wanted, got), count in missed.most_common(6):
            print(f"    {wanted} -> {got}  x{count}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
