"""How much of natural Turkish does the parser actually understand?

Adding patterns by guesswork is how a parser becomes a pile. This asks a model
for the ways a person really phrases each intent, checks which ones we catch,
and prints what we miss — so every pattern added afterwards is one that earns
its place.

The dataset is cached, so the measurement can be repeated for free after each
change to the parser.

Usage: python3 scripts/dil_kapsam.py [--yenile]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.intuition import (Intuition, TEACH, ASK, ASK_WHO, ASK_ABILITIES,  # noqa: E402
                           ASK_WHY, ASK_PROPERTIES, ASK_DESCRIBE, UNKNOWN,
                           UNKNOWN_WORD)
from lmm.relations import IS_A, CAN, HAS_PROPERTY  # noqa: E402
from lmm.lexicon import ACTIVE  # noqa: E402
import lmm.harvest as harvest  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dil-ornekleri.json")

# (label, what it means, the phrasing we already accept)
INTENTS = [
    ("TEACH_TYPE", "bir şeyin ne olduğunu öğretmek", "penguen bir kuştur"),
    ("TEACH_ABILITY", "bir şeyin ne yapabildiğini öğretmek", "kuşlar uçar"),
    ("TEACH_PROPERTY", "bir şeyin nasıl olduğunu öğretmek", "kar beyazdır"),
    ("ASK_DEFINITION", "bir şeyin ne olduğunu sormak", "penguen nedir"),
    ("ASK_ABILITY", "bir şeyin bir şeyi yapıp yapamadığını sormak", "penguen uçar mı"),
    ("ASK_PROPERTY", "bir şeyin bir niteliği var mı diye sormak", "kar beyaz mı"),
    ("ASK_WHO", "bir eylemi kimlerin yaptığını sormak", "kimler uçar"),
    ("ASK_ABILITIES", "bir şeyin neler yapabildiğini sormak", "penguen ne yapabilir"),
    ("ASK_WHY", "bir şeyin sebebini sormak", "penguen neden uçamaz"),
    ("ASK_DESCRIBE", "bir şeyi anlatmasını istemek", "penguen anlat"),
]

EXPECTED = {
    "TEACH_TYPE": lambda i: i.kind == TEACH and i.relation == IS_A,
    "TEACH_ABILITY": lambda i: i.kind == TEACH and i.relation in ("can", "cannot"),
    "TEACH_PROPERTY": lambda i: i.kind == TEACH and i.relation in ("property",
                                                                   "not_property"),
    "ASK_DEFINITION": lambda i: i.kind == ASK and i.relation == IS_A,
    "ASK_ABILITY": lambda i: i.kind == ASK and i.relation == CAN,
    "ASK_PROPERTY": lambda i: i.kind == ASK and i.relation == HAS_PROPERTY,
    "ASK_WHO": lambda i: i.kind == ASK_WHO,
    "ASK_ABILITIES": lambda i: i.kind == ASK_ABILITIES,
    "ASK_WHY": lambda i: i.kind == ASK_WHY,
    "ASK_DESCRIBE": lambda i: i.kind == ASK_DESCRIBE,
}

BRIEF = """Türkçe konuşan bir insanın günlük dilde kullanacağı cümleler yaz.
Sana bir niyet ve o niyeti ifade eden bir örnek verilecek. AYNI niyeti ifade
eden 12 FARKLI cümle yaz. Her cümle tek satırda, numarasız, açıklamasız.
Sadece "penguen", "kuş", "kar", "uçmak", "beyaz" gibi basit kelimeler kullan.
Cümleler kısa olsun ve gerçek konuşma dilinden olsun."""


def generate():
    harvest.BRIEF = BRIEF
    dataset = {}
    for label, meaning, example in INTENTS:
        prompt = f"Niyet: {meaning}\nÖrnek: {example}\n12 farklı cümle:"
        reply = harvest.ask_model([prompt], [])
        lines = [line.strip(" -•*\t") for line in reply.splitlines() if line.strip()]
        dataset[label] = [line for line in lines if len(line.split()) <= 8]
        print(f"  {label}: {len(dataset[label])} cümle")
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=1)
    return dataset


def measure(dataset):
    """Three buckets, not two.

    A sentence the parser reads is understood. A sentence it cannot read only
    because a verb was never taught is not a hole in the grammar — the system
    says which word it needs, and one line fixes it. Counting those together
    hides where the work actually is.
    """
    intuition = Intuition()
    total = understood = needs_word = 0
    misses = []
    print(f"\n{'niyet':<16} {'anlaşılan':>10} {'kelime eksik':>13} {'kalıp yok':>10}")
    for label, sentences in dataset.items():
        hits = words = 0
        for sentence in sentences:
            intent = intuition.understand(sentence)
            if intent.kind not in (UNKNOWN, UNKNOWN_WORD) and EXPECTED[label](intent):
                hits += 1
            elif intent.kind == UNKNOWN_WORD:
                words += 1
            else:
                misses.append((label, sentence))
        count = len(sentences)
        total += count
        understood += hits
        needs_word += words
        print(f"{label:<16} {hits:>3}/{count:<3} %{hits / count * 100:>3.0f}"
              f"   {words:>8}      {count - hits - words:>7}")
    print(f"\nTOPLAM anlaşılan {understood}/{total} — %{understood / total * 100:.0f}")
    print(f"       kelime eksik {needs_word} (tek satırla çözülür)")
    print(f"       kalıp yok    {total - understood - needs_word}")
    print("\nKALIBI OLMAYANLAR (ilk 20):")
    for label, sentence in misses[:20]:
        print(f"  [{label}] {sentence}")
    return understood / total


def main(argv):
    if "--yenile" in argv or not os.path.exists(CACHE):
        print("model örnek üretiyor...")
        dataset = generate()
    else:
        with open(CACHE, encoding="utf-8") as f:
            dataset = json.load(f)
    measure(dataset)


if __name__ == "__main__":
    main(sys.argv)
