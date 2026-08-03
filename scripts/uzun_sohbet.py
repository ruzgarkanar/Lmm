"""Uzun soluklu sohbet sınavı — LMM ile bir dil modeli yan yana.

Bu projenin bütün sohbet ölçümleri kısaydı: "kartal nedir", "penguen uçar mı".
Onlar dilbilgisini ölçüyor, KONUŞMAYI değil. Bir insan böyle konuşmuyor —
konuyu açıyor, dallandırıyor, geri dönüyor, uzun soru soruyor, kısa soru
soruyor, ve cevabın uzunluğunun soruya uymasını bekliyor.

Ölçülen altı şey, altısı da ayrı:

    CEVAPLADI     ret mi cevap mı. Tek başına yanıltıcı — döküm de cevaptır.
    UZUNLUK       karakter. Tek sayı değil DAĞILIM: hep aynı uzunlukta cevap
                  vermek, hiç cevap verememek kadar bozuk.
    UYUM          uzunluk soruya göre değişiyor mu. "nedir" kısa, "anlat"
                  uzun olmalı. Sabitse sistem soruyu değil kendini konuşuyor.
    HAMLE         sohbeti yürüten sözler: konu açma, tepki, itiraz, vazgeçme.
    BAĞLAM        eksiltili soru önceki konuya bağlanıyor mu.
    UYDURMA       oturum boyunca hafızaya yazılan olgu sayısı. Sıfır olmalı.

Sorular ELLE YAZILMIYOR. Konular grafın en zengin kavramlarından seçiliyor,
sorular sistemin kendi söyleyiş organından (`lmm/phrasing.py`) kuruluyor, ve
gerçek sorular `data/sinav/tr-sinav.txt`ten geliyor — hiçbir eğitim dosyasında
geçmeyen satırlar. Böylece sınav, sınavı yazanın beklentisini ölçmüyor.

Dil modeli isteğe bağlı (`--llm`). Yoksa sınav yine çalışır; kıyas kaybolur.

Kullanım:
    python3 scripts/uzun_sohbet.py [--graf ...] [--sohbet 6] [--llm]
"""
import collections
import json
import os
import random
import shutil
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.cli import Session                                 # noqa: E402
from lmm.memory import Memory                               # noqa: E402
from lmm.phrasing import is_a_refusal                       # noqa: E402

# Bir konu hakkında sorulacaklar. Soru METİNLERİ burada değil: her biri
# `phrasing`in kurduğu bir kalıp ve beklenen uzunluk sınıfı. Sınıflar
# uyum ölçümü için — "nedir" ile "anlat" aynı uzunlukta cevap alıyorsa
# sistem soruya değil kendine bakıyor demektir.
SHAPES = [
    ("{} nedir", "kısa"),
    ("{} nasıldır", "orta"),
    ("{} ne yapabilir", "orta"),
    ("{} anlat", "uzun"),
    ("{} hakkında bildiğin her şeyi uzun uzun anlat", "uzun"),
]

# Sohbeti yürüten hamleler — cevabı bilgi değil, davranış.
MOVES = ["hmm ilginç", "peki başka", "yanlış biliyorsun bence", "neyse boşver"]

# Eksiltili takipler: öznesi cümlede yok, önceki turdan gelmeli.
FOLLOW_UPS = ["peki ne yapabilir", "nasıldır peki", "başka ne biliyorsun"]


def topics(memory, count, seed):
    """Grafın en çok şey bildiği kavramlar — konuşulacak bir şey olsun."""
    scored = [(len(memory.query(name)), name) for name in memory.concepts()
              if " " not in name and len(name) > 3]
    scored.sort(reverse=True)
    best = [name for _, name in scored[:400]]
    random.Random(seed).shuffle(best)
    return best[:count]


def real_questions(count, seed):
    """Bizim yazmadığımız uzun sorular."""
    path = "data/sinav/tr-sinav.txt"
    if not os.path.exists(path):
        return []
    rows = [line.rstrip("\n").split("\t", 1)
            for line in open(path, encoding="utf-8") if "\t" in line]
    long_ones = [text for kind, text in rows
                 if kind != "DISARIDA" and len(text.split()) >= 7]
    return random.Random(seed).sample(long_ones, min(count, len(long_ones)))


def _llm(history, ask):
    body = [{"role": "system",
             "content": "Türkçe konuş. Cevabın uzunluğunu soruya göre ayarla: "
                        "kısa soruya kısa, 'anlat' diyene uzun. "
                        "Emin değilsen 'bilmiyorum' de."}] + history
    return ask(body)


def _asker():
    """Dil modeli çağrısı — yoksa None."""
    try:
        from lmm.harvest import _config
        import urllib.request
        url, headers, _ = _config()
    except Exception:                                       # noqa: BLE001
        return None

    def ask(messages):
        body = json.dumps({"messages": messages, "temperature": 0,
                           "max_tokens": 400}).encode("utf-8")
        request = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.load(response)
            return payload["choices"][0]["message"]["content"].strip()
        except Exception:                                   # noqa: BLE001
            return ""
    return ask


def run(graph, rounds, seed, ask=None, show=0):
    memory = Memory.load(graph)
    working = os.path.join(tempfile.mkdtemp(), "sohbet.lmm")
    shutil.copy(graph, working)
    session = Session(working)
    before = len(session.memory.edges)

    chosen = topics(memory, rounds, seed)
    asked = real_questions(rounds * 2, seed)
    lengths = collections.defaultdict(list)
    tally = collections.Counter()
    transcript = []
    llm_lengths = collections.defaultdict(list)
    history = []

    for at, topic in enumerate(chosen):
        for mould, size in SHAPES:
            question = mould.format(topic)
            said = session.respond(question)
            tally["soru"] += 1
            if is_a_refusal(said):
                tally["ret"] += 1
            else:
                tally["cevap"] += 1
                lengths[size].append(len(said))
            transcript.append((question, said))
            if ask is not None:
                history.append({"role": "user", "content": question})
                theirs = _llm(history, ask)
                history.append({"role": "assistant", "content": theirs})
                if theirs:
                    llm_lengths[size].append(len(theirs))
                transcript[-1] = (question, said, theirs)
        # Eksiltili takip: özne cümlede yok.
        for follow in FOLLOW_UPS:
            said = session.respond(follow)
            tally["takip"] += 1
            if not is_a_refusal(said):
                tally["takip tuttu"] += 1
            transcript.append((follow, said))
        # Hamleler
        for move in MOVES:
            said = session.respond(move)
            tally["hamle"] += 1
            if not is_a_refusal(said) and "anlamadım" not in said.lower():
                tally["hamle tuttu"] += 1
            transcript.append((move, said))
        # Bizim yazmadığımız uzun soru
        for question in asked[at * 2:at * 2 + 2]:
            said = session.respond(question)
            tally["gerçek soru"] += 1
            if not is_a_refusal(said):
                tally["gerçek cevap"] += 1
            transcript.append((question, said))

    written = len(session.memory.edges) - before
    return tally, lengths, llm_lengths, written, transcript


def report(tally, lengths, llm_lengths, written, transcript, show):
    print(f"\n  {tally['soru']} konu sorusu · {tally['takip']} eksiltili "
          f"· {tally['hamle']} hamle · {tally['gerçek soru']} gerçek soru")
    print(f"    cevapladı        {tally['cevap']:>4} / {tally['soru']}")
    print(f"    eksiltili tuttu  {tally['takip tuttu']:>4} / {tally['takip']}")
    print(f"    hamle tuttu      {tally['hamle tuttu']:>4} / {tally['hamle']}")
    print(f"    gerçek soru      {tally['gerçek cevap']:>4} / {tally['gerçek soru']}")

    print("\n  UZUNLUK (karakter) — soruya uyuyor mu")
    print(f"    {'soru sınıfı':<8} {'LMM':>18}   {'LLM':>18}")
    for size in ("kısa", "orta", "uzun"):
        ours = lengths.get(size, [])
        theirs = llm_lengths.get(size, [])
        one = (f"{statistics.median(ours):>6.0f} ({len(ours)})" if ours
               else "     — ")
        two = (f"{statistics.median(theirs):>6.0f} ({len(theirs)})" if theirs
               else "     — ")
        print(f"    {size:<8} {one:>18}   {two:>18}")
    ours = [statistics.median(lengths[s]) for s in ("kısa", "orta", "uzun")
            if lengths.get(s)]
    if len(ours) >= 2:
        spread = max(ours) / max(min(ours), 1)
        mark = "UYUYOR" if spread >= 1.8 else "SABİT — soruya bakmıyor"
        print(f"    en uzun / en kısa = {spread:.1f}x   <- {mark}")

    print(f"\n    hafızaya yazılan {written}  "
          f"<- {'TEMİZ' if written == 0 else 'KİRLENDİ'}")

    if show:
        print("\n  — dökümden —")
        for row in transcript[:show]:
            print(f"    BEN : {row[0][:88]}")
            print(f"    LMM : {row[1][:150]}")
            if len(row) > 2 and row[2]:
                print(f"    LLM : {row[2][:150]}")
            print()


def main(argv):
    graph = (argv[argv.index("--graf") + 1] if "--graf" in argv
             else "models/graph/birlesik.lmm")
    rounds = int(argv[argv.index("--sohbet") + 1]) if "--sohbet" in argv else 6
    seed = int(argv[argv.index("--tohum") + 1]) if "--tohum" in argv else 7
    show = int(argv[argv.index("--göster") + 1]) if "--göster" in argv else 0
    ask = _asker() if "--llm" in argv else None
    if "--llm" in argv and ask is None:
        print("  (dil modeli yok; yalnız LMM ölçülüyor)")
    if not os.path.exists(graph):
        print(f"yok: {graph}")
        return 1
    print(f"=== UZUN SOHBET — {os.path.basename(graph)}, {rounds} konu")
    report(*run(graph, rounds, seed, ask, show), show=show)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
