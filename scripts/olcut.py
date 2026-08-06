"""Dürüst ölçüt: elle seçilmiş cümlelerle değil, grafın kendisinden kurulan sınavla.

Bugüne kadarki ölçüm 40 sabit cümleydi ve hepsi üç kuş hakkındaydı — kartal,
penguen, kuş. Elle seçilmiş, hiç değişmeyen bir küme. Ona bakarak yapılan her
ayar, sınavı görüp ders çalışmaktır: skor yükselir, sistem yükselmez.

İkinci kusur daha sinsiydi: ölçüt yalnız "anlamadım"ı sayıyordu. Yanlış ama
kendinden emin bir cevap — "kuş bir kaynaktır" — başarı sayılıyordu. Oysa
uydurmayan bir sistemin asıl ölçüsü budur.

Burada üç şey değişiyor:

    sınav graftan kuruluyor   soru, grafın gerçekten bildiği bir kavram
                              hakkında; hangi kavram olduğunu biz seçmiyoruz
    doğruluk ölçülüyor        cevap grafta yazana uyuyor mu — "anlamadı" ile
                              "yanlış söyledi" ayrı sayılıyor
    örneklem her seferinde    tohum verilmezse rastgele; ezberlenecek bir
    başka                     sınav yok

Dört sonuç ayrı raporlanıyor ve toplamları anlamlıdır:

    DOĞRU     graftaki kayda uyan cevap
    YANLIŞ    kendinden emin ama kayda uymayan  <- en pahalısı
    BİLMİYOR  dürüst reddediş
    ANLAMADI  cümleyi çözemedi

Kullanım:
    python3 scripts/olcut.py [--graf models/graph/birlesik.lmm] [--sayi 200]
                             [--tohum 7] [--anla]
"""
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intent import _where                              # noqa: E402
from lmm.cli import Session                                 # noqa: E402
from lmm.memory import Memory                               # noqa: E402
from lmm.relations import CAN, CANNOT, HAS_PROPERTY, IS_A, SAME_AS  # noqa: E402

# Soru kalıpları ilişki türüne göre. Bunlar sınavın kendisi değil, sorunun
# nasıl sorulduğu — cevabın doğruluğu grafa bakılarak ölçülüyor.
ASKED = {
    IS_A: ("{c} nedir", "{c} bir {t} mu"),
    CAN: ("{c} {t} mı", "{c} ne yapabilir"),
    CANNOT: ("{c} {t} mı",),
    HAS_PROPERTY: ("{c} {t} mı", "{c} nasıldır"),
}

REFUSED = ("bilmiyorum", "öğretir misin", "öğrenmem lazım")
UNPARSED = ("anlamadım", "kelimesini bilmiyorum", "demek istedin")


def exam(memory, count, seed):
    """Graftan sınav kur: (soru, kavram, ilişki, hedef, beklenen_evet).

    Sorular grafın BİLDİĞİ kayıtlardan üretiliyor, yani cevabı bellidir ve
    doğruluk ölçülebilir. Elle cümle yazmak yerine kaydı sorguya çevirmek,
    sınavın ezberlenmesini imkânsız kılıyor.
    """
    edges = [edge for edge in memory.edges
             if edge.relation in ASKED and edge.target
             and len(edge.concept) > 2 and " " not in edge.concept]
    random.Random(seed).shuffle(edges)
    found = []
    for edge in edges:
        for shape in ASKED[edge.relation]:
            found.append((shape.format(c=edge.concept, t=edge.target),
                          edge.concept, edge.relation, edge.target,
                          edge.relation != CANNOT))
            if len(found) >= count:
                return found
    return found


INFINITIVE = ("mak", "mek")


def _stem(word):
    """Hedefi gövdesine indirir: "yüzmek" -> "yüz".

    Hakem mastarı çekimle karşılaştırıyordu ve doğru cevapları yanlış
    sayıyordu: kayıt "yüzmek", cevap "yüzer". İkisinin ortak paydası gövde.
    Ölçen aracın kendisi yanılırsa, ölçüm ölçümden kötüdür.
    """
    word = word.split()[-1]
    for ending in INFINITIVE:
        if word.endswith(ending) and len(word) > len(ending) + 1:
            return word[: -len(ending)]
    return word


def judge(said, concept, relation, target, positive):
    """Cevap doğru mu, yanlış mı, dürüst mü, anlaşılmadı mı."""
    lowered = said.lower()
    if any(mark in lowered for mark in UNPARSED):
        return "ANLAMADI"
    if any(mark in lowered for mark in REFUSED):
        return "BİLMİYOR"
    # Hedef cevapta geçiyor mu — gövde düzeyinde, çünkü kayıt mastar tutuyor
    # ama cevap çekimli konuşuyor.
    if target and _stem(target)[:4] not in lowered:
        return "YANLIŞ"
    # Kutup: olumlu kayda "hayır", olumsuz kayda "evet" denmemeli.
    if lowered.startswith("hayır") and positive:
        return "YANLIŞ"
    if lowered.startswith("evet") and not positive:
        return "YANLIŞ"
    return "DOĞRU"


def main(argv):
    graph = (argv[argv.index("--graf") + 1] if "--graf" in argv
             else _where("graph"))
    count = int(argv[argv.index("--sayi") + 1]) if "--sayi" in argv else 200
    seed = int(argv[argv.index("--tohum") + 1]) if "--tohum" in argv else None
    if seed is None:
        seed = random.randrange(10 ** 6)

    memory = Memory.load(graph)
    questions = exam(memory, count, seed)
    if not questions:
        print("graftan sınav kurulamadı")
        return 1

    reader = None
    if "--anla" in argv:
        try:
            from core.intent import Reader
            found = Reader("cekirdek")
            reader = found.read if found.ready else None
        except Exception:                                   # noqa: BLE001
            reader = None

    import shutil
    import tempfile
    working = os.path.join(tempfile.mkdtemp(), "sinav.lmm")
    shutil.copy(graph, working)
    session = Session(working, intent=reader)

    tally = collections.Counter()
    wrong = []
    for asked, concept, relation, target, positive in questions:
        said = session.respond(asked)
        verdict = judge(said, concept, relation, target, positive)
        tally[verdict] += 1
        if verdict == "YANLIŞ" and len(wrong) < 10:
            wrong.append((asked, said, f"{concept} --{relation}--> {target}"))

    total = sum(tally.values())
    print(f"{os.path.basename(graph)} — {len(memory.edges):,} olgu, "
          f"{len(memory.concepts()):,} kavram")
    print(f"{total} soru, tohum {seed}"
          f"{', niyet ağı açık' if reader else ''}\n")
    for name in ("DOĞRU", "YANLIŞ", "BİLMİYOR", "ANLAMADI"):
        share = tally[name] / max(total, 1) * 100
        print(f"  {name:<10} {tally[name]:>4}  %{share:>5.1f}")
    print(f"\n  cevapladığı sorularda isabet: "
          f"%{tally['DOĞRU'] / max(tally['DOĞRU'] + tally['YANLIŞ'], 1) * 100:.1f}")
    if wrong:
        print("\n  YANLIŞ örnekleri (en pahalı hata türü):")
        for asked, said, record in wrong[:6]:
            print(f"    soru : {asked}")
            print(f"    cevap: {said[:66]}")
            print(f"    kayıt: {record}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
