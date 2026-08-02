"""Türkçe derlem toplama: web metnini indirip düz metne çevirir.

Wikipedia 94 milyon kelime verdi ve bu, 17 milyon parametrelik bir çekirdek
için yetti. Daha büyük bir çekirdek için yetmiyor: kabaca parametre başına 20
parça gerekiyor, yani 300 milyonluk bir model 6 milyar parça ister. Elimizdeki
130 milyon, kırk katından az.

FineWeb-2'nin Türkçe bölümü 125 GB ve kabaca 22 milyar kelime. Tamamı gereksiz;
birkaç parça dosyası hedefi fazlasıyla karşılıyor.

Dosyalar sırayla indiriliyor, metne çevriliyor ve **ham hâli siliniyor** —
yoksa 125 GB'lık bir depoya doğru gider. Diskte her an tek bir parça duruyor.

Kullanım:
    python3 scripts/derlem_topla.py --parca 3        # 3 dosya (~2,5 milyar kelime)
    python3 scripts/derlem_topla.py --parca 1 --dene # tek dosya, deneme
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "data", "tr-web.txt")
SCRATCH = os.path.join(ROOT, "data", "_indirilen.parquet")

REPO = "HuggingFaceFW/fineweb-2"
BRANCH = "main"
FOLDER = "data/tur_Latn/train"
LISTING = f"https://huggingface.co/api/datasets/{REPO}/tree/{BRANCH}/{FOLDER}"
FILE = f"https://huggingface.co/datasets/{REPO}/resolve/{BRANCH}/{{path}}"

SHORTEST = 200          # bundan kısa belge nesir değil, artık
CHUNK = 1 << 20


def listing():
    request = urllib.request.Request(LISTING, headers={"User-Agent": "LMM/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return [entry for entry in json.load(response)
                if entry["path"].endswith(".parquet")]


def fetch(path, size):
    """Tek dosyayı indirir. İlerleme gösterilir; sessiz bekleme kötü bir şey."""
    url = FILE.format(path=path)
    request = urllib.request.Request(url, headers={"User-Agent": "LMM/0.1"})
    started, got = time.time(), 0
    with urllib.request.urlopen(request, timeout=120) as response, \
            open(SCRATCH, "wb") as out:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            out.write(block)
            got += len(block)
            if got % (64 * CHUNK) < CHUNK:
                elapsed = time.time() - started
                print(f"     {got/2**30:5.1f}/{size/2**30:.1f} GB  "
                      f"{got/2**20/max(elapsed,1):5.1f} MB/sn", flush=True)
    return got


def convert(handle):
    """Parquet'i düz metne çevirir. Yalnızca metin sütunu alınır."""
    import pyarrow.parquet as parquet
    table = parquet.ParquetFile(SCRATCH)
    words = kept = 0
    for batch in table.iter_batches(batch_size=2000, columns=["text"]):
        for value in batch.column("text"):
            text = value.as_py()
            if not text or len(text) < SHORTEST:
                continue
            handle.write(text.replace("\n", " ").strip() + "\n")
            kept += 1
            words += text.count(" ") + 1
    return kept, words


def main(argv):
    count = int(argv[argv.index("--parca") + 1]) if "--parca" in argv else 3
    files = listing()
    print(f"{len(files)} parça mevcut, {count} tanesi alınacak")
    total_words = total_docs = 0
    started = time.time()
    mode = "a" if os.path.exists(OUTPUT) else "w"
    with open(OUTPUT, mode, encoding="utf-8") as out:
        for index, entry in enumerate(files[:count], 1):
            name = entry["path"].split("/")[-1]
            size = entry.get("size", 0)
            print(f"\n  [{index}/{count}] {name}  {size/2**30:.1f} GB", flush=True)
            try:
                fetch(entry["path"], size)
            except (urllib.error.URLError, OSError) as error:
                print(f"     indirilemedi: {error}")
                continue
            docs, words = convert(out)
            total_docs += docs
            total_words += words
            os.remove(SCRATCH)          # ham dosya diskte durmasın
            print(f"     {docs} belge, {words/1e6:.0f}M kelime  "
                  f"(toplam {total_words/1e6:.0f}M)", flush=True)

    print(f"\nBİTTİ  {total_docs} belge, {total_words/1e6:.0f}M kelime, "
          f"{(time.time()-started)/60:.0f} dakika")
    print(f"  {OUTPUT}  {os.path.getsize(OUTPUT)/2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
