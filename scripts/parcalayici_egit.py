"""Parçalayıcıyı eğitir ve derlemi ikili parça dizisine çevirir.

Türkçe eklemeli bir dil: "göz", "gözlük", "gözlükçü", "gözlükçüden" hep aynı
köke dayanır. Kelime tabanlı bir sözlük bunların her birini ayrı bir giriş
sayar ve dağarcık patlar; harf tabanlı bir sözlük ise cümleleri gereksiz
uzatır. Alt-kelime parçalama (unigram) tam olarak bu ikisinin arasını bulur ve
eklemeli diller için doğru seçim odur.

Bir de LMM'e özgü bir sebep var: parçalar kök ve ekleri ayırdığında, çekirdeğin
öğrendiği şey kelime listesi değil **ek düzeni** olur. Bilgi grafta kalsın diye
çekirdeği daraltmak istiyoruz; parçalama bu daralmanın ilk adımı.

Kullanım: python3 scripts/parcalayici_egit.py [--sozluk 16000]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sentencepiece as spm                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = os.path.join(ROOT, "data", "tr-metin.txt")
PREFIX = os.path.join(ROOT, "data", "tr-parcalayici")
TOKENS = os.path.join(ROOT, "data", "tr-parcalar.bin")
SAMPLE_LINES = 2_000_000        # parçalayıcıyı eğitmek için bu kadarı yeter


def train_pieces(vocabulary):
    print(f"parçalayıcı eğitiliyor ({vocabulary} parça)...", flush=True)
    started = time.time()
    spm.SentencePieceTrainer.train(
        input=TEXT, model_prefix=PREFIX, vocab_size=vocabulary,
        model_type="unigram", character_coverage=0.9995,
        input_sentence_size=SAMPLE_LINES, shuffle_input_sentence=True,
        normalization_rule_name="nfkc",
        # Türkçe'nin kendi harfleri parça sözlüğünde mutlaka bulunsun.
        required_chars="çğıöşüÇĞİÖŞÜ",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3, num_threads=os.cpu_count())
    print(f"  {time.time()-started:.0f} sn, {PREFIX}.model", flush=True)


def encode(vocabulary):
    import numpy
    pieces = spm.SentencePieceProcessor(model_file=PREFIX + ".model")
    print("derlem parçalara çevriliyor...", flush=True)
    started = time.time()
    written = lines = 0
    with open(TEXT, encoding="utf-8") as source, open(TOKENS, "wb") as out:
        batch = []
        for line in source:
            batch.append(line.strip())
            if len(batch) < 2000:
                continue
            lines += len(batch)
            written += _flush(pieces, batch, out, numpy)
            batch = []
            if lines % 200000 == 0:
                elapsed = time.time() - started
                print(f"  {lines:>9} satır | {written/1e6:>7.1f}M parça | "
                      f"{elapsed:>5.0f} sn", flush=True)
        if batch:
            written += _flush(pieces, batch, out, numpy)
    print(f"\nBİTTİ  {written/1e6:.1f}M parça, {TOKENS} "
          f"({os.path.getsize(TOKENS)/2**20:.0f} MB), {time.time()-started:.0f} sn")
    print(f"  ortalama {written/max(lines,1):.1f} parça/satır")


def _flush(pieces, batch, out, numpy):
    encoded = pieces.encode(batch)
    flat = [token for row in encoded for token in row + [3]]   # 3 = cümle sonu
    numpy.array(flat, dtype=numpy.uint16).tofile(out)
    return len(flat)


def main(argv):
    if not os.path.exists(TEXT):
        print(f"eksik: {TEXT}\nönce: python3 scripts/veri_hazirla.py")
        return 1
    vocabulary = int(argv[argv.index("--sozluk") + 1]) if "--sozluk" in argv else 16000
    train_pieces(vocabulary)
    encode(vocabulary)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
