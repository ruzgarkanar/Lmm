"""Süzülmüş adayları dil modeline etiketletir.

Ağ süzgeç, dil modeli etiketçi, insan sınavı. Üç ayrı rol ve hiçbiri diğerinin
işini yapmıyor: ağ hangi soruların bakmaya değer olduğunu söylüyor, model ne
olduklarını, insan da sonucun işe yarayıp yaramadığını.

Süzgeç sayesinde bütçe az bulunan sınıflara gidiyor — 315 örnekli
ASK_DESCRIBE için 9.527 aday, 3.203 örnekli ASK_ABILITY için de 9.526.
Rastgele örneklemede bu oran 1'e 10 olurdu.

Kullanım: python3 scripts/aday_etiketle.py [--sayi 60000]
"""
import collections
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.compiler import CompileError, model_reader          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "data", "tr-soru-adaylari.txt")
OUTPUT = os.path.join(ROOT, "data", "tr-niyet-genis.txt")
BATCH = 25

BRIEF = """Sana numaralı Türkçe sorular verilecek. Her biri için HANGİ TÜR soru
olduğunu söyle.

  TANIM      bir şeyin ne olduğu soruluyor      "kartal nedir"
  YETENEK    belli bir şeyi yapıp yapmadığı      "penguen uçar mı"
  NITELIK    belli bir niteliği olup olmadığı    "kartal hızlı mı"
  YETENEKLER yapabildiklerinin tamamı           "kartal ne yapabilir"
  NITELIKLER niteliklerinin tamamı              "kartal nasıldır"
  ANLAT      serbest tarif isteniyor            "bana kartaldan bahset"
  NEDEN      sebebi soruluyor                   "penguen neden uçamaz"
  KIM        belli bir şeyi KİMİN yaptığı        "kimler uçar"
  DISARIDA   bunlardan hiçbiri

Emin değilsen DISARIDA yaz. HER SORU İÇİN bir satır, hiçbirini atlama.
Biçim:  NUMARA. TÜR"""

KINDS = {"TANIM": "ASK_DEFINITION", "YETENEK": "ASK_ABILITY",
         "NITELIK": "ASK_PROPERTY", "YETENEKLER": "ASK_ABILITIES",
         "NITELIKLER": "ASK_PROPERTIES", "ANLAT": "ASK_DESCRIBE",
         "NEDEN": "ASK_WHY", "KIM": "ASK_WHO", "DISARIDA": "DISARIDA"}
LINE = re.compile(r"^\s*(\d+)\s*[.=:)\-]?\s*([A-ZİĞÜŞÖÇ]+)\s*$")


def main(argv):
    wanted = int(argv[argv.index("--sayi") + 1]) if "--sayi" in argv else 60000
    global OUTPUT
    if "--cikti" in argv:
        given = argv[argv.index("--cikti") + 1]
        OUTPUT = given if os.path.isabs(given) else os.path.join(ROOT, given)
    if not os.path.exists(SOURCE):
        print(f"eksik: {SOURCE}\nönce: python3 scripts/soru_suz.py")
        return 1
    questions = []
    with open(SOURCE, encoding="utf-8") as handle:
        for line in handle:
            if "\t" in line:
                questions.append(line.rstrip("\n").split("\t", 1)[1])
    questions = questions[:wanted]
    reader = model_reader(temperature=0.0, brief=BRIEF)

    labelled = []
    started = time.time()
    for start in range(0, len(questions), BATCH):
        chunk = questions[start:start + BATCH]
        try:
            reply = reader(["\n".join(f"{i+1}. {q}"
                                      for i, q in enumerate(chunk))])
        except CompileError:
            continue
        for line in reply.splitlines():
            found = LINE.match(line.strip())
            if not found:
                continue
            at, kind = int(found.group(1)), found.group(2).upper()
            if kind in KINDS and 1 <= at <= len(chunk):
                labelled.append((chunk[at - 1], KINDS[kind]))
        if start % (BATCH * 40) == 0:
            with open(OUTPUT, "w", encoding="utf-8") as out:
                for question, kind in labelled:
                    out.write(f"{kind}\t{question}\n")
            print(f"  {start:,}/{len(questions):,}  etiketli {len(labelled):,}  "
                  f"({(time.time()-started)/60:.0f} dk)", flush=True)

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for question, kind in labelled:
            out.write(f"{kind}\t{question}\n")
    print(f"\nBİTTİ {len(labelled):,} etiketli -> {OUTPUT}")
    for kind, count in collections.Counter(k for _, k in labelled).most_common():
        print(f"  {kind:<18} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
