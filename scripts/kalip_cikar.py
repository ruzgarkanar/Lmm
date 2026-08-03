"""Kalıpları elle yazmayı bırak: etiketli gerçek cümlelerden çıkar.

Ölçüldü ve darboğaz buradaydı. 300 gerçek Türkçe cümlede:

    UNKNOWN        %67,7    hiçbir kalıp tutmadı
    UNKNOWN_WORD   % 4,0    kelime bilinmiyor

Yani sözlük sorun değil — dağarcık 805 mastar tanıyor. Sorun 94 kalıbın
gerçek Türkçenin çeşitliliğini tutmaması, ve o 94'ün 88'inin ELLE yazılmış
olması. Bir dilin söyleyiş biçimlerini elle saymak, kelimelerini elle saymak
kadar umutsuz.

Malzeme hazır duruyordu: `data/tr-gercek-niyet.txt` 1.570 etiketli gerçek
soru taşıyor. Etiket niyet TÜRÜNÜ veriyor; eksik olan yuvaların YERİ.

Yuvayı sıklık belirliyor, sözlük değil. `lmm/grammar.py`'deki `learn_pattern`
tanımadığı her kelimeyi SABİT yapıyor ve bu, tek cümleye uyan bir kalıp
üretiyor: "aile planlaması konusunda eğitimleriniz olacak mı" cümlesinden
çıkan kalıp yalnız o cümleyi tutar. Ölçüt tersine dönmeli —

    çok sık geçen kelime     dilbilgisidir, kalıbın SABİT parçası
    seyrek geçen kelime      içeriktir, kalıbın YUVASI

`lmm/verbs.py:is_structural` bu testi zaten yapıyor ve sırayla ölçüyor, oranla
değil; bir dilin en sık üç yüz kelimesi her derlemde aynı türdendir.

Bir şeklin inanılması için İKİ KEZ görünmesi gerekiyor. Bir kez görülen şekil
rastlantı olabilir ve bu, projenin olgulara uyguladığı ölçütün aynısı
(`grammar.induce` da `evidence=2` diyor).

Kullanım:
    python3 scripts/kalip_cikar.py [--veri data/tr-gercek-niyet.txt]
                                   [--kanit 2] [--enfazla 8] [--yaz model.lmm]
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm import asking, frequency                           # noqa: E402
from lmm.grammar import (KAVRAM, FIIL, NITELIK, SORU, KIM, Pattern,
                         FROM_VERB)                         # noqa: E402
from lmm.intuition import tokenize                          # noqa: E402
from lmm.memory import Memory                               # noqa: E402
from lmm.relations import CAN, HAS_PROPERTY, IS_A           # noqa: E402
from lmm.turkish import TurkishMorphology                   # noqa: E402
from lmm.verbs import is_structural                         # noqa: E402

# Etiketten (niyet türü, ilişki). Kapalı bir eşleme: veri yeni bir tür icat
# edemez, yalnızca sistemin zaten tanıdıklarından birini gösterebilir.
LABELS = {
    "ASK_DEFINITION": ("ASK", IS_A),
    "ASK_ABILITY": ("ASK", FROM_VERB),
    "ASK_PROPERTY": ("ASK", HAS_PROPERTY),
    "ASK_ABILITIES": ("ASK_ABILITIES", CAN),
    "ASK_PROPERTIES": ("ASK_PROPERTIES", HAS_PROPERTY),
    "ASK_DESCRIBE": ("ASK_DESCRIBE", None),
    "ASK_WHY": ("ASK_WHY", CAN),
}

LONGEST = 8         # bundan uzun cümleden çıkan kalıp yalnız kendini tutar


def _slot(token, morphology, counts, verbs):
    """Bu kelime kalıbın neresi: sabit mi, hangi yuva mı.

    Sıra önemli. Soru eki ve soru sözcüğü kapalı sınıf, önce onlar. Sonra fiil,
    çünkü fiil sık geçse de içerik taşır ve sabite dönüşmemeli. Sonra yapısal
    sözcük sabit kalır. Kalan içeriktir.
    """
    if asking.particle_of(token, morphology) is not None:
        return SORU
    if asking.interrogative_of(token, morphology) is not None:
        return KIM
    if token in verbs:
        return FIIL
    if counts and is_structural(token, counts):
        return token                    # dilbilgisi: kalıbın sabit parçası
    if morphology.has_copula(token):
        return NITELIK
    return KAVRAM


def _places(slots, relation):
    """Kavramın ve hedefin yeri — yuvaların kendisinden.

    Kavram ilk KAVRAM yuvasıdır: Türkçede özne başta durur. Hedef ilişkiye
    göre değişiyor ve bu bir tercih değil tanım: yetenek sorusunun hedefi
    fiil, nitelik sorusununki niteliktir.
    """
    concept = slots.index(KAVRAM) if KAVRAM in slots else None
    target = None
    if relation in (CAN, FROM_VERB):
        target = slots.index(FIIL) if FIIL in slots else None
    elif relation == HAS_PROPERTY:
        target = slots.index(NITELIK) if NITELIK in slots else None
    elif relation == IS_A:
        # "X nedir" hedefsizdir; "X bir Y mi" ikinci kavramı hedef alır.
        found = [i for i, s in enumerate(slots) if s == KAVRAM]
        target = found[1] if len(found) > 1 else None
    return concept, target


def shapes(rows, morphology, counts, verbs):
    """Etiketli cümlelerden (şekil -> kaç kez, örnek)."""
    found = collections.defaultdict(lambda: [0, None])
    for label, sentence in rows:
        if label not in LABELS:
            continue
        tokens = tokenize(sentence)
        if not 1 < len(tokens) <= LONGEST:
            continue
        kind, relation = LABELS[label]
        slots = [_slot(t, morphology, counts, verbs) for t in tokens]
        # Yuvası olmayan bir şekil kalıp değil, cümlenin kendisidir.
        if not any(s in (KAVRAM, FIIL, NITELIK) for s in slots):
            continue
        # ÇAPASI olmayan bir şekil de kalıp değil, JOKERDİR. Yalnız kavram
        # yuvaları ve bir soru ekinden ibaret bir şekil her cümleye uyar ve
        # niyeti keyfî atar. Ölçüldü: 6+ kelimelik gerçek sorularda okunanların
        # %38'i YANLIŞ niyetle okunuyordu ve yanlışların çoğu bu jokerlerden
        # geliyordu —
        #
        #   "Karbondioksid toksik bir gaz mıdır" -> ASK_WHY
        #   kalıp: {kavram}{kavram}{kavram}{kavram}{kavram}{soru}
        #
        # Soru eki soru olduğunu söyler ama HANGİ soru olduğunu söylemez.
        # Onu söyleyen şey ya bir soru sözcüğü (KIM) ya da sabit bir
        # kelimedir. İkisi de yoksa şekil hiçbir şey ayırt etmiyor.
        if not any(s == KIM or s not in (KAVRAM, FIIL, NITELIK, SORU)
                   for s in slots):
            continue
        concept, target = _places(slots, relation)
        if concept is None:
            continue
        key = (tuple(slots), kind, relation, concept, target)
        found[key][0] += 1
        if found[key][1] is None:
            found[key][1] = sentence
    return found


def main(argv):
    path = (argv[argv.index("--veri") + 1] if "--veri" in argv
            else "data/tr-gercek-niyet.txt")
    evidence = int(argv[argv.index("--kanit") + 1]) if "--kanit" in argv else 2
    if not os.path.exists(path):
        print(f"yok: {path}")
        return 1

    rows = [line.rstrip("\n").split("\t", 1)
            for line in open(path, encoding="utf-8") if "\t" in line]
    morphology = TurkishMorphology()
    counts = frequency.counts()
    verbs = frequency.verbs()
    if not counts:
        print("derlem sayacı yok: yapısal sözcük ayırt edilemez")
        return 1

    found = shapes(rows, morphology, counts, verbs)
    kept = {key: value for key, value in found.items() if value[0] >= evidence}
    print(f"{path} — {len(rows):,} satır")
    print(f"{len(found)} ayrı şekil, {len(kept)} tanesi en az {evidence} kez\n")
    for key, (times, example) in sorted(kept.items(),
                                        key=lambda item: -item[1][0])[:12]:
        slots, kind, relation, concept, target = key
        print(f"  {times:>3}x  {kind:<16} {' '.join(slots)}")
        print(f"       örnek: {example[:60]}")

    if "--yaz" not in argv:
        print("\n  (bir modele yazmak için --yaz model.lmm ekle)")
        return 0
    target_file = argv[argv.index("--yaz") + 1]
    memory = Memory.load(target_file)
    before = len(memory.patterns)
    known = {tuple(entry["tokens"]) for entry in memory.patterns}
    for key, (times, example) in kept.items():
        slots, kind, relation, concept, target = key
        if tuple(slots) in known:
            continue
        pattern = Pattern(list(slots), kind, relation, concept, target,
                          name=f"çıkarıldı ({times}x): {example[:40]}",
                          fallback=True)
        memory.learn_pattern(pattern)
    memory.save(target_file)
    print(f"\n  {len(memory.patterns) - before} yeni kalıp -> {target_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
