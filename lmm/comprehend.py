"""Reading real prose, by restating it in sentences the system can hold.

A document written for people is nothing like the sentences LMM parses: subclauses,
objects, possessives, passive voice, noun phrases three words long. Run our own
banking paper through the reader and it learns one fact out of 1273 sentences.

So a model is used here for the one thing it is genuinely good at and that costs
us nothing to check: restating. It is asked to say what a passage already says,
in the shapes we understand, and forbidden to add anything. The facts still come
from the document — provenance stays with the file, not the model — and every
restated sentence goes through the same audit as any other source: parsed or
skipped, refused when it contradicts what is known, and beaten by a person.

Two samples must agree before anything is written, so a passage the model reads
differently twice teaches nothing.

Usage: python3 -m lmm.comprehend memory.lmm belge.txt [parça-sayısı]
"""
import json
import os
import sys

from lmm.memory import Memory
from lmm.reasoning import Reasoning
from lmm.reading import read_text, sentences
from lmm.harvest import agreed, ask_model, HarvestError
from lmm.lexicon import ACTIVE
import lmm.harvest as harvest

# İkisi de ölçülmedi ve hangisinin ne yaptığı karışıyordu, o yüzden yazılsın:
# `CHUNK_SENTENCES` modele bir seferde kaç cümle verildiği — bağlam ne kadar
# genişse yeniden yazım o kadar isabetli ama uydurma alanı da o kadar geniş.
# `DEFAULT_LIMIT` ise yalnızca komut satırının varsayılanı; bir belgenin kaç
# parçasının okunacağını çağıran belirliyor, bu sayı sessizce kesmiyor.
CHUNK_SENTENCES = 12
DEFAULT_LIMIT = 8

BRIEF = """Sana bir metin parçası verilecek. Bu parçanın SÖYLEDİKLERİNİ,
aşağıdaki basit kalıplarla yeniden yaz. Metinde OLMAYAN hiçbir şey ekleme.
Emin olmadığın hiçbir cümleyi yazma. Her cümle tek satırda, numarasız.

  <kavram> bir <tür>dür
  <kavram> bir <tür> değildir
  <kavram>lar <fiil>
  <kavram> <fiil-olumsuz>
  <kavram> <nitelik>dir
  <kavram> <nitelik> değildir

Kavramlar TEK kelime olmalı (örnek: "model", "bellek", "denetim").
Fiil kullanacaksan ve listede yoksa, önce şu satırı ekle:
  kelime: <mastar> = <olumlu> / <olumsuz>

Bildiğin fiiller: {verbs}

Az ama kesin cümle yaz. Metinde açıkça yazmayan bir şeyi asla yazma."""


def chunks(text, size=CHUNK_SENTENCES):
    found = sentences(text)
    for start in range(0, len(found), size):
        yield ". ".join(found[start:start + size]) + "."


def restate(passage, samples=2, asker=ask_model):
    """What the passage says, in shapes we can hold. Empty when unsure."""
    harvest.BRIEF = BRIEF
    verbs = sorted({value[0] for value in ACTIVE.verbs.values()})
    return agreed([asker([passage], verbs) for _ in range(samples)])


def comprehend(memory, text, source, limit=DEFAULT_LIMIT, asker=ask_model,
               corpus=None):
    """Fold a human document into a memory, one restated passage at a time.

    Every restatement is also recorded as a (prose, plain) pair when a corpus
    path is given. Those pairs are the training data for doing this ourselves:
    each one shows a sentence a person wrote next to the shapes it reduces to,
    which is exactly what a grammar has to learn. The model is scaffolding, and
    it builds the thing that replaces it.
    """
    reasoning = Reasoning(memory)
    pairs = []
    learned = refused = skipped = passages = 0
    for passage in chunks(text):
        if passages >= limit:
            break
        passages += 1
        restated = restate(passage, asker=asker)
        if not restated:
            continue
        pairs.append({"düzyazı": passage, "sade": restated})
        for line in restated.splitlines():
            if line.strip().lower().startswith("kelime:"):
                body = line.split(":", 1)[1]
                if "=" in body and "/" in body:
                    infinitive, forms = body.split("=", 1)
                    positive, negative = forms.split("/", 1)
                    memory.learn_word(infinitive.strip(), positive.strip(),
                                      negative.strip())
        without_words = "\n".join(l for l in restated.splitlines()
                                  if not l.strip().lower().startswith("kelime:"))
        report = read_text(without_words, memory, source, reasoning)
        learned += len(report.learned)
        refused += len(report.conflicts)
        skipped += len(report.skipped)
    if corpus is not None:
        _record(corpus, pairs)
    return passages, learned, refused, skipped


def _record(path, pairs):
    """Append to the corpus that will let us stop calling a model at all."""
    existing = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing + pairs, f, ensure_ascii=False, indent=1)


def main(argv):
    if len(argv) < 3:
        print("Kullanım: python3 -m lmm.comprehend memory.lmm belge.txt [parça]")
        return 1
    model_path, document = argv[1], argv[2]
    limit = int(argv[3]) if len(argv) > 3 else DEFAULT_LIMIT
    memory = Memory.load(model_path) if os.path.exists(model_path) else Memory()
    with open(document, encoding="utf-8") as f:
        text = f.read()
    try:
        corpus = os.path.splitext(model_path)[0] + "-cift.json"
        passages, learned, refused, skipped = comprehend(
            memory, text, os.path.basename(document), limit, corpus=corpus)
    except HarvestError as error:
        print(f"hata: {error}")
        return 1
    memory.save(model_path)
    print(f"{os.path.basename(document)}: {passages} parça yeniden yazıldı — "
          f"{learned} bilgi öğrenildi, {refused} çelişki reddedildi, "
          f"{skipped} cümle atlandı.")
    print(f"eşleşme külliyatı: {corpus}")
    print(f"bellek: {len(memory.edges)} bilgi, "
          f"{len(memory.vocabulary)} kelime, {len(memory.concepts())} kavram")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
