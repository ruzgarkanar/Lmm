"""Harvesting: the system's own questions, answered by a language model.

Curiosity already knows what the memory is missing. This hands those questions
to a model, takes the answers back through the same audit every other source
goes through, and writes down what survives. The loop the research literature
keeps describing and nobody ships — notice a gap, go find out, verify, keep it,
never ask again — closes here, with a model standing in for the library.

Nothing is taken on faith. Answers arrive as ordinary sentences, are parsed by
our own parser, are refused when they contradict what is known, and are marked
as machine-sourced so a person always outranks them. What the model cannot be
trusted to do — decide what is true — it is never asked to do.

No dependencies: the API call is plain urllib.

Environment:
    OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION   (or LLM_MODEL for a plain OpenAI endpoint)

Usage: python3 -m lmm.harvest memory.json [soru-sayısı]
"""
import json
import os
import sys
import urllib.error
import urllib.request

from lmm.memory import Memory
from lmm.reasoning import Reasoning
from lmm.curiosity import Curiosity
from lmm.lexicon import ACTIVE
from lmm.distill import distill_text, generalise

TIMEOUT = 60
DEFAULT_BATCH = 8       # bir turda sorulan soru sayısı; ölçülmedi

# Sıcaklık ile örnek sayısı BİRLİKTE bir anlam taşıyor ve bu yazılı değildi.
# `agreed` iki örneğin aynı cümleyi söylemesini istiyor; sıcaklık düştükçe iki
# örneğin aynı çıkması kolaylaşır, yani çapraz denetim zayıflar. 0,2 ile iki
# örnek, "modelin ezberinde sağlam duranı al" demek — ama sıfıra yaklaştıkça
# denetim bir törene döner. Ölçülmedi; ölçüsü "aynı sorularda kaç satır
# eleniyor" olurdu.
TEMPERATURE = 0.2

BRIEF = """Sen bir bilgi kaynağısın. Sana sorulan sorulara SADECE aşağıdaki
kalıplarla, düz Türkçe cümlelerle cevap ver. Her cümle tek satırda olsun.
Açıklama, başlık, madde işareti, numara KULLANMA.

Kullanabileceğin kalıplar:
  <kavram> bir <tür>dür          örnek: karınca bir böcektir
  <kavram> bir <tür> değildir    örnek: karınca bir kuş değildir
  <kavram>lar <fiil>             örnek: karıncalar yürür
  <kavram> <fiil-olumsuz>        örnek: karınca uçamaz
  <kavram> <nitelik>dir          örnek: karınca küçüktür
  <kavram> <nitelik> değildir    örnek: karınca büyük değildir

Fiil kullanacaksan ve o fiil listede yoksa, cümlelerden ÖNCE şu satırı ekle:
  kelime: <mastar> = <olumlu> / <olumsuz>
örnek: kelime: kazmak = kazar / kazamaz

Bildiğin fiiller: {verbs}

Emin olmadığın hiçbir şeyi yazma. Az ama doğru cümle yaz."""


class HarvestError(Exception):
    pass


def _config():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HarvestError("OPENAI_API_KEY tanımlı değil.")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    if endpoint and deployment:
        url = (f"{endpoint}/openai/deployments/{deployment}"
               f"/chat/completions?api-version={version}")
        return url, {"api-key": key}, deployment
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    return ("https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {key}"}, model)


def ask_model(questions, verbs, url=None, headers=None):
    """Send the questions, get back plain sentences. No parsing of JSON replies."""
    if url is None:
        url, headers, _ = _config()
    body = {
        "messages": [
            {"role": "system", "content": BRIEF.format(verbs=", ".join(verbs))},
            {"role": "user", "content": "\n".join(questions)},
        ],
        "temperature": TEMPERATURE,
    }
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise HarvestError(f"HTTP {error.code}: {error.read()[:200].decode()}")
    except urllib.error.URLError as error:
        raise HarvestError(f"bağlanılamadı: {error.reason}")
    return payload["choices"][0]["message"]["content"]


def open_questions(memory, reasoning, count):
    """What curiosity would ask a teacher, asked of a model instead."""
    curiosity = Curiosity(memory, reasoning)
    questions = []
    while len(questions) < count:
        question = curiosity.next_question()
        if question is None:
            break
        questions.append(question.text)
    return questions


def agreed(answers):
    """Keep only what every sample said.

    The audit catches a claim that fights what the memory holds; it cannot catch
    a confident falsehood about something the memory knows nothing about. Asking
    twice does: a model that says animals cook in one sample and cannot cook in
    the next has told us it does not know, and neither line is kept.
    """
    seen = [{line.strip() for line in answer.splitlines() if line.strip()}
            for answer in answers]
    if not seen:
        return ""
    kept = set.intersection(*seen)
    order = [line.strip() for line in answers[0].splitlines() if line.strip()]
    return "\n".join(line for line in order if line in kept)


def harvest(memory, count=DEFAULT_BATCH, model_name=None, asker=ask_model,
            samples=2):
    """One round: ask, cross-check, audit, keep. Returns (questions, report)."""
    reasoning = Reasoning(memory)
    questions = open_questions(memory, reasoning, count)
    if not questions:
        return [], None
    verbs = sorted({value[0] for value in ACTIVE.verbs.values()})
    answers = [asker(questions, verbs) for _ in range(samples)]
    if model_name is None:
        model_name = _config()[2]
    return questions, distill_text(agreed(answers), memory, model_name, reasoning)


def main(argv):
    path = argv[1] if len(argv) > 1 else "memory.json"
    count = int(argv[2]) if len(argv) > 2 else DEFAULT_BATCH
    memory = Memory.load(path)
    try:
        questions, report = harvest(memory, count)
    except HarvestError as error:
        print(f"hata: {error}")
        return 1
    if report is None:
        print("merak edecek bir şey kalmamış.")
        return 0
    formed = generalise(memory)
    memory.save(path)

    print(f"sorulan: {len(questions)}")
    for question in questions:
        print(f"  ? {question}")
    print(f"alınan : {len(report.learned)} yeni bilgi, "
          f"{len(report.reinforced)} pekişen, {len(report.conflicts)} çelişkili, "
          f"{len(report.skipped)} atlandı")
    for edge, explanation in report.conflicts:
        print(f"  reddedildi: {edge.concept} — {explanation}")
    if formed:
        print(f"kendi çıkardığı kural: {len(formed)}")
    print(f"bellek : {len(memory.edges)} bilgi")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
