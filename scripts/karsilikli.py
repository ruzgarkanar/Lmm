"""LMM ile bir dil modeli karşılıklı konuşur; LMM konuşarak öğrenir.

Bu, sistemin insanlara açıldığındaki hâlinin provası. İnsan yerine bir dil
modeli konuşuyor, ama akış birebir aynı: karşı taraf cümle söylüyor, LMM
okuyor, kapıdan geçen grafa yazılıyor, LMM cevap veriyor, konuşma sürüyor.

Neden dil modeli: bir insan gecede yüz cümle söyler, bu binlerce söyler. Ve
insanların soracağı şeyleri sorar — derlemden gelen cümleler bunu yapmaz.
Toplu okuma grafı GENİŞLETİR, sohbet onu insanların KULLANDIĞI yere doğru
çeker. İkisi ayrı iş.

    KONU        LMM'in bildiği bir kavram + bilmediği bir komşusu
    KARŞI TARAF dil modeli: soru sorar, bilgi verir, düzeltir
    LMM         cevaplar; öğretilen cümleyi kendi dilbilgisiyle okur
    KAYNAK      sohbet:<model> — YABANCI basamağı, belge değil
    KAZANÇ      yazılan olgu · sorulan soru · LMM'in cevaplayabildiği

KAYNAK NEDEN YABANCI. Bir dil modeli burada bir insanın yerine geçiyor ve
insanların söylediği şey doğrulanmamış bir iddiadır. `lmm/trust.py` bu
basamağı belgenin de çıkarımın da altına koyuyor, ve doğrulanmamış kayıtlar
cevapta işaretle görünüyor. Uydurma riski buradan yönetiliyor: kapı geçirse
bile söz, tanıklı bilgiyle aynı ağırlıkta konuşmuyor.

Kullanım:
    set -a; . /tmp/lmm_env.sh; set +a
    python3 scripts/karsilikli.py [--tur 200] [--graf ...] [--yaz]
"""
import concurrent.futures
import json
import os
import random
import shutil
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.cli import Session                                 # noqa: E402
from lmm.harvest import _config, HarvestError               # noqa: E402
from lmm.memory import Memory                               # noqa: E402
from lmm.serialize import is_refusal as is_a_refusal                       # noqa: E402

# Karşı tarafa ne söyleniyor. İki şey isteniyor ve ikisi de bilerek: ÖĞRETMEK
# ve SORMAK. Yalnız soru sorsa graf büyümez; yalnız öğretse LMM'in neyi
# cevaplayabildiğini hiç görmeyiz.
#
# Cümle biçimi kısıtlanıyor çünkü LMM'in dilbilgisi düz cümle okuyor. Bu bir
# hile değil: karşı taraf da bir öğretmen gibi davranıyor, ve bir öğretmen
# öğrencisinin anlayacağı cümleyi kurar.
BRIEF = """Sen Türkçe konuşan bir öğretmensin. Karşındaki, bilgiyi bir graf
olarak tutan ve DÜZ cümlelerle öğrenen bir sistem.

Her turda TAM OLARAK şunu yap:
1. "{konu}" hakkında bildiğin, DOĞRU ve BASİT üç cümle yaz. Her cümle şu
   biçimlerden biri olsun:
      X bir Y'dir.        X şunu yapar.        X şu niteliktedir.
      X'in Y'si vardır.   X şunu yapamaz.
2. Sonra o konu hakkında bir SORU sor.

Kurallar:
- Emin olmadığın hiçbir şey yazma. Uydurma.
- Cümleler kısa olsun, virgülsüz, tek yargılı.
- Madde imi kullanma. Her cümle ayrı satırda.
- Soruyu son satıra yaz ve soru işaretiyle bitir."""

TIMEOUT = 90
# Eşzamanlı istek. Sıralı gidince konu başına 27 saniye ölçüldü, yani 200 konu
# 1,5 saat — sohbetle beslemenin anlamı kalmıyor. İş ağ beklemesi, hesap
# değil. Öğretme tarafı sıralı kalıyor ve kalmak zorunda: her cümle grafı
# değiştiriyor ve sonrakinin okunuşunu etkiliyor.
ESZAMANLI = 24


def _ask(messages, url, headers):
    body = json.dumps({"messages": messages, "temperature": 0.3,
                       "max_tokens": 300}).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.load(response)
        return payload["choices"][0]["message"]["content"].strip()
    except Exception:                                       # noqa: BLE001
        return ""


def topics(memory, count, seed):
    """LMM'in TANIDIĞI ama az bildiği kavramlar — öğrenilecek yer orası.

    En zengin kavramları seçmek boşuna: onlar zaten dolu. En fakirleri seçmek
    de boşuna: çoğu ayrıştırma artığı. Aradaki bant, hem gerçek hem eksik.
    """
    scored = [(len(memory.query(name)), name) for name in memory.concepts()
              if " " not in name and len(name) > 3]
    middle = [name for size, name in scored if 2 <= size <= 8]
    random.Random(seed).shuffle(middle)
    return middle[:count]


def main(argv):
    turns = int(argv[argv.index("--tur") + 1]) if "--tur" in argv else 200
    graph = (argv[argv.index("--graf") + 1] if "--graf" in argv
             else "models/graph/birlesik.lmm")
    seed = int(argv[argv.index("--tohum") + 1]) if "--tohum" in argv else 11
    write = "--yaz" in argv
    show = int(argv[argv.index("--göster") + 1]) if "--göster" in argv else 3
    if not os.path.exists(graph):
        print(f"yok: {graph}")
        return 1
    try:
        url, headers, model = _config()
    except HarvestError as reason:
        print(f"{reason}\n  set -a; . /tmp/lmm_env.sh; set +a")
        return 1

    working = graph if write else os.path.join(tempfile.mkdtemp(), "k.lmm")
    if not write:
        shutil.copy(graph, working)
    # Kaynak adı MODELİN adı: kim söylediyse o yazılsın. Yabancı basamağı
    # `lmm/trust.py`de tanımlı ve belgenin altında duruyor.
    session = Session(working, speaker=model)
    memory = Memory.load(graph)
    chosen = topics(memory, turns, seed)
    before = len(session.memory.edges)
    started = time.time()

    # ÖNCE hepsi paralel sorulur, SONRA sıralı öğretilir. Öğretmeyi
    # paralelleştirmek yanlış olurdu: her cümle grafı değiştiriyor ve
    # sonrakinin nasıl okunacağını etkiliyor.
    def fetch(topic):
        return topic, _ask([{"role": "system",
                             "content": BRIEF.replace("{konu}", topic)},
                            {"role": "user",
                             "content": f"{topic} hakkında anlat."}],
                           url, headers)

    with concurrent.futures.ThreadPoolExecutor(ESZAMANLI) as pool:
        fetched = list(pool.map(fetch, chosen))
    print(f"  {len(fetched)} konu soruldu ({time.time() - started:.0f} sn)")

    taught = asked = answered = shown = 0
    for at, (topic, said) in enumerate(fetched):
        if not said:
            continue
        lines = [line.strip(" -•\t") for line in said.splitlines()
                 if line.strip()]
        replies = []
        for line in lines:
            if not line:
                continue
            answer = session.respond(line)
            replies.append((line, answer))
            if line.endswith("?"):
                asked += 1
                if not is_a_refusal(answer):
                    answered += 1
            else:
                taught += 1
        if shown < show:
            shown += 1
            print(f"\n  — {topic} —")
            for line, answer in replies:
                print(f"    LLM: {line[:88]}")
                print(f"    LMM: {answer[:110]}")
        if (at + 1) % 50 == 0:
            grew = len(session.memory.edges) - before
            print(f"  {at + 1}/{len(chosen)} konu · +{grew} olgu · "
                  f"{answered}/{asked} soru cevaplandı")

    grew = len(session.memory.edges) - before
    print(f"\n  {len(chosen)} konu · {taught} cümle öğretildi · "
          f"{asked} soru soruldu")
    print(f"  grafa giren      {grew}")
    print(f"  cevaplanan soru  {answered}/{asked}"
          f"  %{answered / max(asked, 1) * 100:.0f}")
    print(f"  süre             {time.time() - started:.0f} sn")
    if write:
        session.memory.save(working)
        print(f"\n  -> {working}")
    else:
        print("\n  (yazmak için --yaz ekle)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
