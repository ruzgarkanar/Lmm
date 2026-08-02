"""Niyet örnekleri üretir: elle yazılmış şablonların yerine geçecek veri.

Dil organı bugüne kadar 48 elle yazılmış cümle kalıbıydı ve ölçüldü: 12 doğal
sorudan 5'ini anlıyor. Kalıp eklemek çözüm değil — 48'den 200'e çıkmak da
görülmemiş bir söyleyişte çöker.

Ölçtüğümüz şey şuydu: niyet elle verildiğinde graf zaten doğru cevabı veriyor.
Yani eksik olan bilgi değil, **cümleyi niyete bağlamak**. Ve bu bir dil
modelleme problemi değil, on üç sınıflı bir sınıflandırma problemi.

Sınıflandırıcı zaten var (`lmm/network.py`, elle yazılmış softmax, kütüphanesiz).
Eksik olan tek şey eğitim verisiydi ve o bedavaya üretilebilir: dil modeline her
niyeti yüzlerce farklı şekilde söyletip küçük sınıflandırıcıya damıtmak.

Dil modeli burada bir otorite değil, bir **söyleyiş kaynağı**. Ne olduğuna karar
vermiyor — sınıfı biz veriyoruz, o yalnızca o sınıfın kaç türlü söylenebileceğini
gösteriyor. Yanılırsa maliyeti bir yanlış etiket, bir yanlış olgu değil.

Kullanım: python3 scripts/niyet_uret.py [--sayi 60]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.compiler import CompileError, model_reader          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "data", "tr-niyet.txt")

# Her sınıf: ne demek, ve iki örnek. Örnekler kalıp değil, YÖN gösteriyor —
# üretilen cümleler bunlara benzemek zorunda değil, tam tersi isteniyor.
KINDS = {
    "TEACH_TYPE": ("Bir şeyin ne olduğunu SÖYLEME (soru değil, bildirme).",
                   ["Kartal bir kuştur.", "Balina memelidir."]),
    "TEACH_NOT_TYPE": ("Bir şeyin belli bir tür OLMADIĞINI söyleme.",
                       ["Balina bir balık değildir.", "Örümcek böcek değil."]),
    "TEACH_ABILITY": ("Bir şeyin ne yaptığını / yapabildiğini söyleme.",
                      ["Kuşlar uçar.", "Penguen yüzemez."]),
    "TEACH_PROPERTY": ("Bir şeyin niteliğini söyleme.",
                       ["Kar beyazdır.", "Kartal hızlıdır."]),
    "TEACH_NOT_PROPERTY": ("Bir şeyin belli bir niteliği olmadığını söyleme.",
                           ["Penguen hızlı değildir.", "Buz sıcak değil."]),
    "ASK_DEFINITION": ("Bir şeyin NE OLDUĞUNU sorma.",
                       ["Kartal nedir?", "Vaşak ne demek?"]),
    "ASK_ABILITY": ("Belli bir şeyi yapıp yapamadığını sorma.",
                    ["Penguen uçar mı?", "Kartal yüzebilir mi?"]),
    "ASK_PROPERTY": ("Belli bir niteliği olup olmadığını sorma.",
                     ["Kartal hızlı mı?", "Penguen tüylü müdür?"]),
    "ASK_WHO": ("Belli bir şeyi KİMİN/NEYİN yaptığını sorma.",
                ["Kimler uçar?", "Ne yüzer?"]),
    "ASK_ABILITIES": ("Bir şeyin yapabildiklerinin TAMAMINI sorma.",
                      ["Kartal ne yapabilir?", "Penguen neler yapar?"]),
    "ASK_PROPERTIES": ("Bir şeyin niteliklerinin TAMAMINI sorma.",
                       ["Kartal nasıldır?", "Kuşların özellikleri nelerdir?"]),
    "ASK_WHY": ("Bir şeyin NEDEN öyle olduğunu sorma.",
                ["Penguen neden uçamaz?", "Kartal niçin hızlıdır?"]),
    "ASK_DESCRIBE": ("Bir şeyi ANLATMASINI isteme (serbest tarif).",
                     ["Penguen anlat.", "Bana kartaldan bahseder misin?"]),
}

BRIEF = """Sana bir cümle TÜRÜ ve o türden iki örnek verilecek. Görevin, AYNI
şeyi söyleyen ama BAŞKA BAŞKA biçimlerde kurulmuş Türkçe cümleler yazmak.

Çeşitlilik en önemli şey. Şunları karıştır:
  - kısa ve uzun kurulumlar
  - kibar ve gündelik söyleyişler ("... misin", "... söyler misin", "yaa ...")
  - dolaylı anlatım ("merak ediyorum ...", "acaba ...", "bir de şunu sorayım ...")
  - devrik cümleler
  - farklı soru kelimeleri ve farklı fiiller
  - farklı konular (hayvan, nesne, meslek, doğa olayı — tek konuda kalma)

Verilen iki örneğe BENZEMEYE çalışma, tam tersine onlardan uzaklaş.

Her satıra bir cümle yaz. Numara koyma, açıklama yazma, başlık yazma."""


def generate(reader, kind, description, examples, count):
    request = (f"TÜR: {description}\n"
               f"ÖRNEKLER:\n  " + "\n  ".join(examples) +
               f"\n\nBu türden {count} farklı cümle yaz.")
    reply = reader([request])
    found = []
    for line in reply.splitlines():
        line = line.strip().lstrip("-•*0123456789. ").strip()
        if 3 <= len(line) <= 120 and " " in line:
            found.append(line)
    return found


def main(argv):
    count = int(argv[argv.index("--sayi") + 1]) if "--sayi" in argv else 60
    try:
        reader = model_reader(temperature=1.0, brief=BRIEF)
    except CompileError as error:
        print(f"model yok: {error}")
        return 1

    rows = []
    started = time.time()
    for kind, (description, examples) in KINDS.items():
        try:
            found = generate(reader, kind, description, examples, count)
        except CompileError as error:
            print(f"  {kind}: istek düştü ({error})")
            continue
        rows.extend((kind, sentence) for sentence in found)
        print(f"  {kind:<20} {len(found):>3} cümle", flush=True)

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for kind, sentence in rows:
            out.write(f"{kind}\t{sentence}\n")
    print(f"\n{len(rows)} cümle -> {OUTPUT}  ({time.time()-started:.0f} sn)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
