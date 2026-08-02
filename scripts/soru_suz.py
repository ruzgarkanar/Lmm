"""Etiket bütçesini doğru yere harcamak: ağı süzgeç olarak kullanmak.

Derlemde 4,2 milyon gerçek soru var ve hepsini dil modeline etiketletmek
haftalar sürer. Rastgele örneklemek de işe yaramıyor — soruların %61'i kapsam
dışı ve elimizde zaten 18 bin kapsam dışı örnek var. Rastgele etiketlemek,
bütçenin çoğunu zaten bol olan sınıfa harcamak demek.

Eğitilmiş ağ burada bir karar verici değil, bir **süzgeç**: hangi soruların
kapsam içi olabileceğini tahmin ediyor, etiketi yine dil modeli koyuyor.
Ağ yanılırsa maliyeti boşa bir etiket; doğruluk kararı hiçbir zaman ona
bırakılmıyor.

Az bulunan sınıflara öncelik veriliyor — 315 örnekli ASK_DESCRIBE'ın bir
örneği, 3.203 örnekli ASK_ABILITY'nin bir örneğinden değerli.

Kullanım: python3 scripts/soru_suz.py --girdi sorular.json --sayi 60000
"""
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "data", "tr-soru-adaylari.txt")
HAVE = os.path.join(ROOT, "data", "tr-gercek-niyet-buyuk.txt")

OUTSIDE = "DISARIDA"
LEAST = 0.30        # bu güvenin altını süzgeç bile geçirmiyor


def scarcity(path=HAVE):
    """Sınıf başına elimizde kaç örnek var — azına öncelik verilecek."""
    counts = collections.Counter()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if "\t" in line:
                    counts[line.split("\t", 1)[0]] += 1
    return counts


def main(argv):
    from core.intent import Reader
    source = argv[argv.index("--girdi") + 1] if "--girdi" in argv else None
    wanted = int(argv[argv.index("--sayi") + 1]) if "--sayi" in argv else 60000
    if not source or not os.path.exists(source):
        print("kullanım: --girdi sorular.json [--sayi 60000]")
        return 1

    reader = Reader("cekirdek")
    if not reader.ready:
        print("niyet ağı yüklenemedi — önce scripts/niyet_egitim.py")
        return 1
    questions = json.load(open(source, encoding="utf-8"))
    counts = scarcity()
    print(f"{len(questions):,} soru süzülecek")
    print("  elimizdeki dağılım:")
    for kind, count in counts.most_common():
        print(f"    {kind:<18} {count}")

    # Tarama sınırlı: yeterli aday bulununca durur. 4,2 milyonun tamamını
    # geçirmek gereksiz — aradığımız şey birkaç on bin aday.
    scan = int(argv[argv.index("--tara") + 1]) if "--tara" in argv else 1_500_000
    # İkinci dalgada ilk dalganın taradığı bölge atlanıyor; aynı soruları
    # yeniden süzmek bütçeyi tekrara harcamak olurdu.
    skip = int(argv[argv.index("--atla") + 1]) if "--atla" in argv else 0
    questions = questions[skip:]
    picked = collections.defaultdict(list)
    started = time.time()
    step = 20000
    for start in range(0, min(scan, len(questions)), step):
        piece = questions[start:start + step]
        for question, (name, confidence) in zip(piece, reader.read_many(piece)):
            if name and confidence >= LEAST:
                picked[name].append(question)
        total = sum(len(v) for v in picked.values())
        print(f"  {start+len(piece):,}  aday {total:,}  "
              f"({time.time()-started:.0f} sn)", flush=True)
        if total >= wanted * 2:
            break

    # Azınlık sınıflarına öncelik: her sınıftan sırayla alınıyor, böylece
    # 315 örnekli sınıf 3.203 örnekli sınıfla aynı payı görüyor.
    order = sorted(picked, key=lambda k: counts.get(k, 0))
    chosen, at = [], 0
    while len(chosen) < wanted and any(len(picked[k]) > at for k in order):
        for kind in order:
            if at < len(picked[kind]) and len(chosen) < wanted:
                chosen.append((kind, picked[kind][at]))
        at += 1

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for kind, question in chosen:
            out.write(f"{kind}\t{question}\n")
    print(f"\n{len(chosen):,} aday -> {OUTPUT}")
    for kind, count in collections.Counter(k for k, _ in chosen).most_common():
        print(f"  {kind:<18} {count:>6}   (elimizde {counts.get(kind,0)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
