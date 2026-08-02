"""Web derlemini grafa okur — dil modeli değil, bilgi için.

Derlem dil çekirdeğini eğitmek için indirildi ve grafa okutmak akla gelmedi.
Oysa okuyucu 100 cümlede ~5 olgu çıkarıyor ve derlem 290 milyon cümle: kabaca
**14 milyon olgu**, yani bugünkü grafın 375 katı.

Bilginin ağırlıklara değil grafa yazılması bu projenin ana iddiası; o hâlde
büyümesi gereken yer de graf. Dil çekirdeğini büyütmek akıcılık kazandırır,
bilgi kazandırmaz.

Bellek dikkatle kullanılıyor: her olgu Python nesnesi olarak yüzlerce bayt
tutar, 14 milyonu birden tutmak birkaç gigabayt eder. Bu yüzden graf parça
parça yazılıyor ve sayaçlar akıtılarak toplanıyor.

Kullanım:
    python3 scripts/derlemi_oku.py --cumle 5000000 --cikti model-web.lmm
"""
import collections
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.clauses import readable                        # noqa: E402
from lmm.frames import read, to_fact                    # noqa: E402
from lmm.intuition import tokenize                      # noqa: E402
from lmm.memory import Memory, Edge                     # noqa: E402
from lmm.reading import sentences                       # noqa: E402
from lmm.verbs import discover, graded_pairs            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "data", "tr-web.txt")
SOURCE_NAME = "tr-web"
COUNTING = 300_000_000      # kelime sayımı için okunan karakter
REPORT = 200_000


def counters(limit=COUNTING):
    """Kelime ve derecelendirme sayaçları — tür testleri bunlara dayanıyor."""
    words = collections.Counter()
    graded = collections.Counter()
    got = 0
    with open(SOURCE, encoding="utf-8") as handle:
        for line in handle:
            tokens = re.findall(r"[a-zçğıöşü]+", line.lower())
            words.update(tokens)
            graded.update(graded_pairs(tokens))
            got += len(line)
            if got > limit:
                break
    return words, graded


def main(argv):
    if not os.path.exists(SOURCE):
        print(f"eksik: {SOURCE}\nönce: python3 scripts/derlem_topla.py")
        return 1
    wanted = (int(argv[argv.index("--cumle") + 1]) if "--cumle" in argv
              else 2_000_000)
    output = (argv[argv.index("--cikti") + 1] if "--cikti" in argv
              else os.path.join(ROOT, "model-web.lmm"))

    print("sayaçlar çıkarılıyor (tür testleri buna dayanıyor)...", flush=True)
    started = time.time()
    words, graded = counters()
    print(f"  {len(words)} kelime, {time.time()-started:.0f} sn", flush=True)

    memory = Memory()
    for infinitive, positive, negative in discover(words, minimum=5):
        memory.learn_word(infinitive, positive, negative)
    print(f"  {len(memory.vocabulary)} fiil metinden çıkarıldı", flush=True)

    seen = set()
    read_lines = kept = 0
    started = time.time()
    with open(SOURCE, encoding="utf-8") as handle:
        for document in handle:
            for whole in sentences(document):
                read_lines += 1
                for piece in readable(whole, lexicon=memory.lexicon):
                    frame = read(tokenize(piece), words, memory.lexicon)
                    if frame is None:
                        continue
                    fact = to_fact(frame, memory.lexicon, words, graded)
                    if fact is None or fact in seen:
                        continue
                    seen.add(fact)
                    concept, relation, target = fact
                    try:
                        memory.write(Edge(concept, relation, target,
                                          source=SOURCE_NAME))
                        kept += 1
                    except Exception:               # noqa: BLE001
                        pass
                if read_lines % REPORT == 0:
                    elapsed = time.time() - started
                    print(f"  {read_lines/1e6:5.1f}M cümle | {kept/1e6:5.2f}M olgu"
                          f" | {len(memory.concepts())/1e6:5.2f}M kavram"
                          f" | {read_lines/elapsed:.0f} cümle/sn", flush=True)
                if read_lines >= wanted:
                    break
            if read_lines >= wanted:
                break

    memory.save(output)
    print(f"\nBİTTİ  {read_lines} cümle -> {kept} olgu, "
          f"{len(memory.concepts())} kavram, {(time.time()-started)/60:.0f} dk")
    print(f"  {output}  {os.path.getsize(output)/2**20:.0f} MB")
    print(f"  100 cümlede {kept/max(read_lines,1)*100:.1f} olgu")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
