"""Derlemi parçalara çevirir — çekirdek eğitiminin girdisi.

Elimizdeki `tr-parcalar.bin` yalnızca 171 milyon parça ve derlem 7,8 milyar.
Yani dil çekirdeği verinin %2'siyle eğitilmiş. Chinchilla ölçüsüyle 171 milyon
parça ~8,5 milyonluk bir modeli doyurur; bizimki 16,8M, yani **veri az, model
değil**.

Daha büyük ve düzgün eğitilmiş bir çekirdek için önce daha çok parça lazım.
Burada yapılan tam olarak bu ve başka hiçbir şey: metin okunur, parçalayıcıdan
geçirilir, uint16 olarak diske akıtılır. Bellekte tutulmaz — tamamı birkaç GB.

Kullanım: python3 scripts/parcala.py [--parca 1000000000] [--cikti data/tr-parcalar-buyuk.bin]
"""
import array
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "data", "tr-web.txt")
PIECES = os.path.join(ROOT, "data", "tr-parcalayici.model")
OUTPUT = os.path.join(ROOT, "data", "tr-parcalar-buyuk.bin")
WANTED = 1_000_000_000
FLUSH = 4_000_000        # kaç parçada bir diske yazılır


def main(argv):
    import sentencepiece as spm
    wanted = int(argv[argv.index("--parca") + 1]) if "--parca" in argv else WANTED
    output = argv[argv.index("--cikti") + 1] if "--cikti" in argv else OUTPUT
    if not os.path.exists(SOURCE):
        print(f"eksik: {SOURCE}")
        return 1
    pieces = spm.SentencePieceProcessor(model_file=PIECES)
    end = pieces.eos_id() if pieces.eos_id() >= 0 else 1

    buffer = array.array("H")
    written = 0
    started = time.time()
    with open(SOURCE, encoding="utf-8", errors="ignore") as source, \
            open(output, "wb") as out:
        for line in source:
            line = line.strip()
            if len(line) < 40:
                continue
            buffer.extend(pieces.encode(line))
            buffer.append(end)
            if len(buffer) >= FLUSH:
                buffer.tofile(out)
                written += len(buffer)
                del buffer[:]
                if written % (FLUSH * 25) < FLUSH:
                    hız = written / max(time.time() - started, 1)
                    kalan = (wanted - written) / max(hız, 1) / 60
                    print(f"  {written/1e6:.0f}M parça  "
                          f"{hız/1e3:.0f}K/sn  ~{kalan:.0f} dk kaldı", flush=True)
                if written >= wanted:
                    break
        if buffer:
            buffer.tofile(out)
            written += len(buffer)
    size = os.path.getsize(output)
    print(f"\nBİTTİ {written:,} parça -> {output}  "
          f"({size/2**30:.2f} GB, {(time.time()-started)/60:.0f} dk)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
