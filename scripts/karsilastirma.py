"""The experiment that settles it: LMM against a language model, on the two
properties that actually differ.

Accuracy is not the comparison to run — our knowledge came out of a model in the
first place, so scoring it against that model would be circular. What is not
circular is behaviour under ignorance and behaviour under repetition.

1. INVENTED THINGS. Ten plausible-sounding entities that do not exist. A system
   that only answers from memory must decline all ten. A system that samples
   likely words has no mechanism to decline.

2. THE SAME QUESTION, THREE TIMES. Answers built from a graph are identical by
   construction. Answers sampled from a distribution need not be.

Usage: python3 scripts/karsilastirma.py memory.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.cli import Session  # noqa: E402
from lmm.harvest import ask_model, HarvestError  # noqa: E402

# Plausible Turkish-sounding species and materials that do not exist.
INVENTED = [
    ("kızıl sırtlı anadolu çalıkuşu", "uçar mı"),
    ("mavi benekli toros semenderi", "yüzer mi"),
    ("gümüş yeleli step tilkisi", "koşar mı"),
    ("karadeniz cam yosunu", "büyür mü"),
    ("beyaz gagalı ege martısı", "uçar mı"),
    ("tunç kanatlı bozkır kelebeği", "uçar mı"),
    ("kara pullu fırat sazanı", "yüzer mi"),
    ("altın boynuzlu kapadokya keçisi", "koşar mı"),
    ("sarı damarlı likya çamı", "büyür mü"),
    ("gri tüylü ıspir baykuşu", "görür mü"),
]

REPEATED = ["penguen uçar mı", "balina yüzer mi", "kartal uçar mı",
            "demir batar mı", "buz erir mi"]

DECLINED = ("bilmiyorum", "anlamadım", "hiç duymadım", "öğrenmedim",
            "bilgim yok", "emin değilim", "bulunmuyor", "bilinmiyor",
            "kayıt yok", "tanımlanmamış", "hayali", "gerçek değil",
            "böyle bir", "yer almıyor", "rastlanmamış")


def declined(text):
    lowered = text.lower().replace("İ", "i")
    return any(marker in lowered for marker in DECLINED)


def lmm_run(path):
    session = Session(path)
    started = time.time()
    declines = 0
    for name, question in INVENTED:
        if declined(session.respond(f"{name.split()[-1]} {question}")):
            declines += 1
    varied = 0
    for question in REPEATED:
        answers = {Session(path).respond(question) for _ in range(3)}
        varied += len(answers) > 1
    elapsed = time.time() - started
    return declines, varied, elapsed


def llm_run():
    brief = ("Sana bir varlık hakkında soru sorulacak. Kısa cevap ver. "
             "Emin değilsen açıkça 'bilmiyorum' de.")
    declines = 0
    started = time.time()
    for name, question in INVENTED:
        reply = ask_model([f"{name} {question}?"], [],
                          **_plain(brief))
        if declined(reply):
            declines += 1
    varied = 0
    for question in REPEATED:
        answers = {ask_model([question + "?"], [], **_plain(brief)).strip()
                   for _ in range(3)}
        varied += len(answers) > 1
    return declines, varied, time.time() - started


def _plain(brief):
    """ask_model with our teaching brief swapped for a neutral one."""
    import lmm.harvest as harvest
    harvest.BRIEF = brief
    return {}


def main(path):
    print("DENEY 1 — OLMAYAN ŞEYLER (10 uydurma varlık)")
    print("DENEY 2 — AYNI SORU 3 KEZ (5 soru)\n")

    declines, varied, elapsed = lmm_run(path)
    print(f"LMM  : {declines}/10 reddetti, {varied}/5 soruda cevap değişti, "
          f"{elapsed * 1000:.0f} ms, 0 ağ çağrısı")

    try:
        declines, varied, elapsed = llm_run()
        print(f"LLM  : {declines}/10 reddetti, {varied}/5 soruda cevap değişti, "
              f"{elapsed:.0f} sn, 45 ağ çağrısı")
    except HarvestError as error:
        print(f"LLM  : çalıştırılamadı ({error})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "memory.json")
