"""LMM kendi eksiğini sorar, LLM cevaplar, kapı doğrular, graf büyür.

Toplu okuma KÖRDÜR: derlem ne veriyorsa onu alıyoruz. Bu döngü hedefli —
sorular grafın kendi boşluklarından geliyor. `lmm/curiosity.py` zaten üretiyor
ve bugüne kadar o soruları KULLANICIYA soruyordu:

    "bu arada, bunu hiç öğrenmedim: organ nedir?"

Ucunu bir dil modeline çevirmek döngüyü kapatıyor. Değişen tek şey soruyu
kimin cevapladığı; doğrulayan yol aynı kalıyor ve aynı kalmak zorunda.

    MERAK       graf neyi bilmiyor      -> curiosity.next_question()
    SORU        sistemin kendi ağzından -> phrasing (hiçbir kalıp burada değil)
    CEVAP       dil modeli               -> düz cümle, olgu değil
    KAPI        okuma hattının aynısı    -> scripts/okuma_al.py
    GRAF        yalnız geçenler          -> kaynak: merak:<model>

NEDEN GÜVENLİ. Dil modeli buraya OLGU yazmıyor, CÜMLE söylüyor. O cümle
sistemin kendi okuyucusundan geçiyor, kendi dilbilgisiyle geri okunuyor,
grafla çelişip çelişmediğine bakılıyor. Yani dil modeli bir kaynak, bir
yetke değil — tıpkı Vikipedi gibi. Uydurursa kapı eler; kapının bu gece
ölçülen geçirme oranı %57 ve elenenlerin çoğu gerçekten elenmeliydi.

NE KATMAZ. Merak ancak grafın TANIDIĞI kavramlar hakkında soru sorabilir;
yeni kavram getirmez, var olanı derinleştirir. Toplu okuma genişletir, bu
derinleştirir. İkisi ayrı iş ve ikisi de gerekli.

Kullanım:
    set -a; . /tmp/lmm_env.sh; set +a
    python3 scripts/merak_dongusu.py [--tur 200] [--graf ...] [--yaz]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util                                       # noqa: E402

from lmm.cli import Session                                 # noqa: E402
from lmm.harvest import _config, HarvestError               # noqa: E402


def _load(name):
    """Kardeş betiği olduğu yerden alır — kopyalamak iki kopya demek."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(here, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Dil modeline ne söyleniyor. Kısa ve düz cümle isteniyor, JSON değil: olguyu
# BİZİM okuyucumuz çıkaracak, çünkü çıkarımı da denetlenmeli. İki aşamayı tek
# isteğe sıkıştırmak, modelin hem kaynak hem hakem olması demekti.
ISTEM = """Şu soruya bir ya da iki DÜZ cümleyle cevap ver.

Kurallar:
- Emin değilsen yalnızca "bilmiyorum" yaz. Tahmin etme.
- Cümleyi basit kur: "X bir Y'dir", "X şunu yapar", "X şu niteliktedir".
- Madde imi, açıklama, giriş cümlesi yazma. Yalnız cevabın kendisi.

Soru: {soru}

Cevap:"""

REFUSAL = "bilmiyorum"


def ask(question, url, headers, harvest):
    """Soruyu sorar, düz cümleyi döndürür. Ağ hatası sessizce boş döner —
    bir sorunun kaybı, koşunun kaybı olmamalı."""
    try:
        said, _, _ = harvest._ask_plain(question, url, headers, ISTEM)
        return said
    except Exception:                                       # noqa: BLE001
        return ""


def main(argv):
    turns = int(argv[argv.index("--tur") + 1]) if "--tur" in argv else 200
    graph = (argv[argv.index("--graf") + 1] if "--graf" in argv
             else "models/graph/birlesik.lmm")
    out = (argv[argv.index("--çıktı") + 1] if "--çıktı" in argv
           else "merak.jsonl")
    if not os.path.exists(graph):
        print(f"yok: {graph}")
        return 1

    harvest = _load("okuyucu_dene")
    try:
        url, headers, _ = _config()
    except HarvestError as reason:
        print(f"{reason}\n  set -a; . /tmp/lmm_env.sh; set +a")
        return 1

    session = Session(graph)
    started = time.time()
    asked = answered = 0
    with open(out, "a", encoding="utf-8") as handle:
        for turn in range(turns):
            question = session.curiosity.next_question()
            if question is None:
                print("  merak bitti: grafta sorulacak boşluk kalmadı")
                break
            asked += 1
            said = ask(question.text, url, headers, harvest)
            if not said or REFUSAL in said.lower():
                continue
            answered += 1
            # Cevabın her cümlesi ayrı bir okuma. Sistemin KENDİ okuyucusuna
            # gidecek, o yüzden burada hiçbir çıkarım yapılmıyor.
            handle.write(json.dumps({"soru": question.text, "cümle": said},
                                    ensure_ascii=False) + "\n")
            handle.flush()          # çökme her şeyi kaybetmesin
            if asked % 25 == 0:
                each = (time.time() - started) / asked
                print(f"  {asked}/{turns} soruldu · {answered} cevap "
                      f"({each:.1f} sn/soru)")

    print(f"\n  {asked} soru, {answered} cevap, {time.time() - started:.0f} sn")
    print(f"  -> {out}")
    print(f"\n  şimdi okuyucuya ver, sonra kapıya:")
    print(f"    python3 scripts/okuyucu_dene.py --veri <cümleler> ...")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
