"""Çevrimdışı okuyucuyu küçük bir örneklemde dener ve verimini ölçer.

96 bin tanımı okutmadan önce cevaplanması gereken tek soru var: bu okuyucu
tanım başına kaç olgu çıkarıyor, ve o olguların kaçı kapıdan geçiyor?

Karşılaştırma noktası elimizde: mevcut düzenli-ifade hasadı tanım başına
**0,15** olgu veriyor ve hepsi TÜR. Bir okuyucunun değeri, o sayıyı ne kadar
aştığıyla ölçülür — model adıyla değil.

Asıl ölçüt "olgu başına maliyet", "çağrı başına maliyet" değil. Tanım başına
3 sağlam olgu veren pahalı bir okuyucu, 0,5 veren ucuzdan hem iyi hem ucuz
olabilir. Bu betik o hesabı yapılabilir kılıyor.

Kimlik bilgileri YALNIZ ortamdan okunuyor; hiçbir değer bu dosyada durmuyor:
    set -a; . /tmp/lmm_env.sh; set +a

Kullanım:
    python3 scripts/okuyucu_dene.py [--sayi 100] [--çıktı okuma.jsonl]
                                    [--veri data/tr-tanimlar.txt]

`--sayi` verilmezse tamamını okur ve kaldığı yerden devam eder.
"""
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.harvest import _config, HarvestError                # noqa: E402

# İstem `notebooks/lmm_okuyucu.ipynb` ile BİREBİR aynı olmalı, yoksa
# karşılaştırma modeli değil istemi ölçer.
#
# Yerleştirme `.format()` ile DEĞİL `replace` ile yapılıyor: istem örnek JSON
# nesneleri taşıyor ve `.format()` süslü parantezi biçim alanı sanıp patlıyor.
# İlk yazışta bu gözden kaçtı ve model kendi şemasını uydurdu — çıktı biçimini
# hiç göstermemiştim, yalnız ilişki tablosunu vermiştim. On iki tanımdan sıfır
# olgu çıktı ve kusur modelde değil istemdeydi.
ISTEM = """Aşağıdaki ansiklopedi tanımından OLGULAR çıkar.

Her olgu için TAM OLARAK şu biçimde bir JSON satırı yaz:
{"kavram": "...", "ilişki": "...", "hedef": "..."}

`ilişki` yalnız şunlardan biri olabilir:
  type      X bir Y'dir            {"kavram":"kartal","ilişki":"type","hedef":"kuş"}
  property  X niteliği taşır       {"kavram":"kartal","ilişki":"property","hedef":"hızlı"}
  can       X şunu yapabilir       {"kavram":"kartal","ilişki":"can","hedef":"avlanmak"}
  cannot    X şunu yapamaz         {"kavram":"penguen","ilişki":"cannot","hedef":"uçmak"}
  has       X'in şu parçası var    {"kavram":"kuş","ilişki":"has","hedef":"kanat"}

Kurallar:
- `kavram` her satırda tanımın KONUSU olsun, küçük harfle. Cümle başındaki
  zarfı özne sanma: "Matematikte, grup bir yapıdır" -> kavram `grup`.
- `hedef` TEK kelime olsun: sıfat sıfat, fiil MASTAR halinde (uçmak, avlanmak).
- Metinde olmayan hiçbir şey yazma. Emin değilsen o satırı hiç yazma.
- Yalnız JSON satırları döndür, başka hiçbir şey yazma.

Tanım: {tanim}

JSON satırları:"""

TIMEOUT = 90
# Eşzamanlı istek. Ölçüldü: sıralı gidince 100 tanım 350 saniye, yani 96.099
# tanım 93 SAAT. Okuma işi ağ beklemesi, hesap değil — bekleyen bir isteğin
# yanında yirmi tane daha beklenebilir. Sınır sağlayıcının hız kotası;
# 429 gelirse düşür.
ESZAMANLI = 40


def _ask(sentence, url, headers):
    body = {"messages": [{"role": "user",
                          "content": ISTEM.replace("{tanim}", sentence)}],
            "temperature": 0}
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        payload = json.load(response)
    usage = payload.get("usage", {})
    return (payload["choices"][0]["message"]["content"],
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


def rows_in(text):
    """Modelin çıktısından JSON nesnelerini ayıklar."""
    found = []
    for piece in re.findall(r"\{[^{}]*\}", text):
        try:
            row = json.loads(piece)
        except ValueError:
            continue
        if row.get("kavram") and row.get("ilişki") and row.get("hedef"):
            found.append(row)
    return found


def main(argv):
    count = (int(argv[argv.index("--sayi") + 1]) if "--sayi" in argv
             else 10 ** 9)      # verilmezse HEPSİ
    out = (argv[argv.index("--çıktı") + 1] if "--çıktı" in argv
           else "deneme.jsonl")
    path = (argv[argv.index("--veri") + 1] if "--veri" in argv
            else "data/tr-tanimlar.txt")
    try:
        url, headers, model = _config()
    except HarvestError as reason:
        print(f"{reason}\n  set -a; . /tmp/lmm_env.sh; set +a")
        return 1

    lines = [line.strip() for line in open(path, encoding="utf-8")
             if 40 < len(line.strip()) < 240]
    # Kaldığı yerden devam: çıktıda hangi CÜMLELER okunmuşsa atlanıyor. Satır
    # SAYISINA bakmak yanlış olurdu — bir tanımdan birden çok olgu çıkıyor ve
    # sayı, atlanacak yeri şişirip okunmamış tanımları es geçiyordu.
    already = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            try:
                already.add(json.loads(line).get("cümle", ""))
            except ValueError:
                continue
        if already:
            print(f"  {len(already)} cümle zaten okunmuş, atlanıyor")
    lines = [line for line in lines if line[:160] not in already][:count]
    print(f"{model} — {len(lines):,} tanım, {ESZAMANLI} eşzamanlı\n")

    written, failed = [0], 0
    tokens = [0, 0]
    lock = threading.Lock()
    started = time.time()
    # Olgular ANINDA diske yazılıyor, sonda değil. İlk yazışta hepsi bellekte
    # birikip en sonda yazılıyordu ve bu, iki buçuk saatlik bir koşuda kabul
    # edilemez: kesinti her şeyi götürür ve "kaldığı yerden devam" da hiçbir
    # şey bulamaz — çıktı dosyası koşu bitene kadar boş kalıyor.
    handle = open(out, "a", encoding="utf-8")

    def one(sentence):
        nonlocal failed
        try:
            text, asked, said = _ask(sentence, url, headers)
        except Exception as reason:                         # noqa: BLE001
            with lock:
                failed += 1
                if failed <= 2:
                    print(f"  hata: {reason}", flush=True)
            return
        rows = rows_in(text)
        for row in rows:
            row["cümle"] = sentence[:160]
        with lock:
            tokens[0] += asked
            tokens[1] += said
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written[0] += len(rows)
            if written[0] % 500 < len(rows):
                handle.flush()

    with concurrent.futures.ThreadPoolExecutor(ESZAMANLI) as pool:
        for index, _ in enumerate(pool.map(one, lines), 1):
            if index % 500 == 0:
                print(f"  {index:,}/{len(lines):,}  {written[0]:,} olgu "
                      f"({index / max(time.time() - started, 1):.1f} tanım/sn, "
                      f"kalan ~{(len(lines) - index) / max(index / max(time.time() - started, 1), 0.1) / 3600:.1f} sa)",
                      flush=True)
    handle.flush()
    handle.close()
    facts = [None] * written[0]
    prompt_tokens, output_tokens = tokens

    read = len(lines) - failed
    elapsed = time.time() - started
    print(f"\n  {read} tanım, {len(facts)} olgu, {elapsed:.0f} sn")
    print(f"  TANIM BAŞINA {len(facts) / max(read, 1):.2f} olgu"
          f"   (düzenli-ifade hasadı: 0.15)")
    print(f"  token: {prompt_tokens:,} girdi + {output_tokens:,} çıktı")
    print(f"  tanım başına {prompt_tokens / max(read, 1):.0f} + "
          f"{output_tokens / max(read, 1):.0f}")
    print(f"\n  -> {out}   (kapıya sok: scripts/okuma_al.py {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
