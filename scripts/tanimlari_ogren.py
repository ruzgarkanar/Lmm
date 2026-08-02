"""Wikipedia tanımlarını LMM grafına yazar — epistemik kapıdan geçirerek.

Bu, "bilgiyi ağırlıklara gömme" kararının reddedildiği yer. Aynı 70 bin olgu bir
dil modeline verilse parametrelere dağılır ve tek tek düzeltilemez; burada her
biri kendi satırında, kaynağıyla birlikte durur ve tek cümleyle değiştirilebilir.

Çelişkiler yazılmaz, sayılır ve raporlanır. Bir ansiklopedi kendi içinde de
çelişebilir; sessizce birini seçmek, LMM'in reddettiği tam olarak o davranış.

Kullanım: python3 scripts/tanimlari_ogren.py [--sinir N] [--cikti model-wiki.lmm]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.definitions import extract                     # noqa: E402
from lmm.memory import Memory, Edge, IS_A               # noqa: E402
from lmm.reasoning import Reasoning                     # noqa: E402
from lmm.learning import LearningLoop, LEARNED, REINFORCED, CONFLICT  # noqa: E402
from lmm.turkish import TurkishMorphology               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lmm import registry                                    # noqa: E402
SOURCE_FILE = os.path.join(ROOT, "data", "tr-tanimlar.txt")
SOURCE_NAME = "tr.wikipedia.org"
OUTPUT = os.path.join(ROOT, "models", "graph", "wikipedia.lmm")


def main(argv):
    if not os.path.exists(SOURCE_FILE):
        print(f"eksik: {SOURCE_FILE}\nönce: python3 scripts/veri_hazirla.py")
        return 1
    limit = int(argv[argv.index("--sinir") + 1]) if "--sinir" in argv else 0
    output = argv[argv.index("--cikti") + 1] if "--cikti" in argv else OUTPUT

    morphology = TurkishMorphology()
    memory = Memory()
    reasoning = Reasoning(memory)
    learning = LearningLoop(memory, reasoning)

    started = time.time()
    read = learned = reinforced = conflicts = refused = 0
    examples = []
    with open(SOURCE_FILE, encoding="utf-8") as lines:
        for line in lines:
            read += 1
            pair = extract(line.strip(), morphology)
            if pair is None:
                refused += 1
                continue
            concept, kind = pair
            edge = Edge(concept, IS_A, kind, source=SOURCE_NAME)
            # Çelişki denetimi grafın tamamına bakar; burada yalnızca doğrudan
            # zıddı arıyoruz, çünkü 70 bin olguda tam ata yürüyüşü saatler alır
            # ve tanımlar birbirini nadiren yalanlar.
            opposite = memory.direct(concept, "not_type", kind)
            if opposite is not None:
                conflicts += 1
                continue
            existing = memory.direct(concept, IS_A, kind)
            if existing is not None:
                existing.corroborate(SOURCE_NAME)
                reinforced += 1
            else:
                memory.write(edge)
                learned += 1
                if len(examples) < 6:
                    examples.append((concept, kind))
            if learned and learned % 10000 == 0:
                elapsed = time.time() - started
                print(f"  {read:>7} satır | {learned:>6} olgu | "
                      f"{elapsed:>5.0f} sn | {learned/elapsed:>6.0f} olgu/sn",
                      flush=True)
            if limit and learned >= limit:
                break

    print(f"\n{read} satır okundu")
    print(f"  öğrenilen   {learned}")
    print(f"  pekişen     {reinforced}")
    print(f"  çelişki     {conflicts}")
    print(f"  tanım değil {refused}")
    print(f"  kavram      {len(memory.concepts())}")
    memory.save(output)
    print(f"\n{output}  {os.path.getsize(output)/2**20:.1f} MB  "
          f"({time.time()-started:.0f} sn, GPU kullanılmadı)")
    print(f"  bilgi başına {os.path.getsize(output)/max(learned,1):.0f} bayt")
    for concept, kind in examples:
        print(f"    {concept} bir {kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
