"""Çevrimdışı okuyucunun çıkardığı olguları KAPIDAN geçirerek alır.

Mimarinin kendi cevabı (`lmm/compiler.py`): dil modeli bir kez, çevrimdışı,
belge okunurken kullanılır; çıktısı denetlenir; sonra ortadan kalkar. Çalışma
anında graf yalnız kalır ve CPU'da çalışır. RAG'den farkı tam burası — orada
model her cevapta devrededir, burada hiç.

Bu betik o denetimin kendisi. Okuyucudan gelen hiçbir üçlü doğrudan grafa
girmiyor; her biri dört kapıdan geçiyor:

    KAVRAM MI          Özne, sistemin kavram sayabileceği bir şey mi. Yapısal
                       sözcük, fiil, durum eki taşıyan zarf ("matematikte")
                       kavram olamaz — bu üçü de bugün ölçüldü ve üçü de grafa
                       çöp düğüm yazmıştı.
    GERİ OKUNUYOR MU   Üçlü cümleye çevrilip sistemin KENDİ ayrıştırıcısıyla
                       yeniden okunuyor ve çıkan olgu aslıyla karşılaştırılıyor.
                       Okuyamadığı bir şeyi bilmesinin anlamı yok: cevap
                       veremeyecek. Kutbu da bu yakalıyor — kelime listesi
                       kutbu kodlayamaz.
    ÇELİŞİYOR MU       Graf zaten tersini söylüyorsa yazılmıyor, sayılıyor.
                       Sessizce üzerine yazmak, hafızayı kirletmenin en sinsi
                       yolu.
    KAYNAK KÜNYESİ     Her kenar "okuyucu:<model>" künyesiyle giriyor. Sonradan
                       tek komutla geri alınabilir olması şart — bir LLM'in
                       ağırlıklarından bilgi silinemez, buradan silinebilir.

Kullanım:
    python3 scripts/okuma_al.py okuma.jsonl [--graf models/graph/birlesik.lmm]
                                            [--yaz] [--kaynak okuyucu:qwen]

Girdi biçimi — satır başına bir JSON nesnesi:
    {"kavram": "kartal", "ilişki": "property", "hedef": "hızlı",
     "cümle": "Kartal, hızlı ve yırtıcı bir kuştur."}
"""
import collections
import json
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm import frequency, phrasing                         # noqa: E402
from lmm.intuition import TEACH                             # noqa: E402
from lmm.memory import Edge, Memory                         # noqa: E402
from lmm.reasoning import Reasoning                         # noqa: E402
from lmm.relations import CAN, CANNOT, HAS_PART, HAS_PROPERTY, IS_A  # noqa: E402
from lmm.turkish import TurkishMorphology                   # noqa: E402
from lmm.verbs import is_structural                         # noqa: E402

# Okuyucunun kullanabileceği ilişkiler. Kapalı bir küme: okuyucu yeni bir
# ilişki icat edemez, yalnızca envanterdekilerden birini gösterebilir.
ALLOWED = (IS_A, HAS_PROPERTY, CAN, CANNOT, HAS_PART)

SHORTEST = 3


def _a_concept(name, counts, verbs, morphology):
    """Bu ad kavram olabilir mi.

    Üçü de bugün ölçülmüş kusurlardan geliyor ve üçü de grafa çöp düğüm
    yazmıştı: yapısal sözcük ("bir kase" -> `bir`), fiil ("kuş ne demek"
    bozuluyordu), durum eki taşıyan zarf ("Matematikte, bir grup ..." ->
    `matematikte`, gerçek özne kayıp).
    """
    if not name or len(name) < SHORTEST or len(name) > 40:
        return False
    head = name.split()[-1]
    if head in verbs or name in verbs:
        return False
    if morphology.is_oblique(head):
        return False
    if counts and (is_structural(head, counts) or is_structural(name, counts)):
        return False
    return True


def _said(concept, relation, target):
    """Üçlüyü sistemin kendi ağzından bir cümleye çevirir."""
    if relation == IS_A:
        return phrasing.is_a_clause(concept, target)
    if relation == HAS_PROPERTY:
        return phrasing.property_clause(concept, target, True)
    if relation == HAS_PART:
        return phrasing.part_clause(concept, target, True)
    return phrasing.ability_clause(concept, target, relation == CAN)


def _round_trip(concept, relation, target, language):
    """Cümleye çevrilip GERİ okunuyor mu — ve aynı olgu mu çıkıyor.

    Bu kapı bir kez kaldırıldı ve gecenin en tehlikeli deliğini açtı: kelime
    listesi kutbu kodlayamaz, "uçar" ile "uçamaz" aynı kelimeyi taşır. Geri
    okuma kutbu yakalar çünkü ayrıştırıcı eki okur.

    Okuyucu SOHBETİN okuduğu yol olmalı. İlk yazışta `frames.read`
    kullanılmıştı ve "kuşun kanadı var" cümlesi reddediliyordu — oysa sohbette
    o cümle sorunsuz öğreniliyor. Sebep: sohbet `frames`'i değil dilbilgisi
    kalıplarını kullanıyor. Yanlış okuyucuyla yapılan denetim, sistemin
    gerçekte anladığı şeyleri eliyordu.
    """
    intent = language.understand(_said(concept, relation, target))
    return (intent.kind == TEACH and intent.concept == concept
            and intent.relation == relation
            and str(intent.target or "").startswith(str(target)[:4]))


_SHARED = {}


def _prepare(graph):
    """Her işçi sürecin kendi belleği ve dil organı — bir kez kuruluyor."""
    from lmm.discovered import words_of
    from lmm.grammar import Pattern
    from lmm.intuition import Intuition
    memory = Memory.load(graph)
    held = {}

    def concepts():
        marker = len(memory.edges)
        if held.get("at") != marker:
            held["at"], held["names"] = marker, memory.concepts()
        return held["names"]

    language = Intuition(lexicon=memory.lexicon, words=words_of(memory),
                         known=concepts)
    for entry in memory.patterns:
        pattern = Pattern.from_dict(entry)
        language.grammar.add(pattern, first=not pattern.fallback)
    _SHARED["memory"] = memory
    _SHARED["language"] = language


def _screen(chunk):
    """Bir öbeği KAVRAM ve GERİ OKUMA kapılarından geçirir.

    Bu iki kapı grafın o anki durumundan bağımsız — kavram sınaması derleme,
    geri okuma dilbilgisine bakıyor. Bağımsız oldukları için paralelleşebilir,
    ve pahalı olan da bunlar: profillendi, sürenin neredeyse tamamı geri
    okumadaki dilbilgisi eşleşmesinde geçiyor.

    Çelişki denetimi paralelleşemez: kabul edilen her olgu grafı değiştiriyor
    ve sonraki denetimi etkiliyor. O yüzden o aşama sıralı kalıyor — ama ucuz,
    çünkü sözlük araması.
    """
    memory = _SHARED["memory"]
    language = _SHARED["language"]
    counts = frequency.counts()
    verbs = frequency.verbs()
    morphology = TurkishMorphology()
    passed, tally, rejected = [], collections.Counter(), []
    for row in chunk:
        concept = (row.get("kavram") or "").strip().lower()
        relation = (row.get("ilişki") or row.get("iliski") or "").strip()
        target = (row.get("hedef") or "").strip().lower()
        if relation not in ALLOWED:
            tally["ilişki tanınmadı"] += 1
            continue
        if not _a_concept(concept, counts, verbs, morphology):
            tally["kavram değil"] += 1
            if len(rejected) < 4:
                rejected.append(("kavram", concept))
            continue
        if not target or len(target) < SHORTEST:
            tally["hedef yok"] += 1
            continue
        if relation == HAS_PART and not _round_trip(concept, relation,
                                                    target, language):
            target = phrasing.possessed(target)
        if not _round_trip(concept, relation, target, language):
            tally["geri okunamadı"] += 1
            if len(rejected) < 4:
                rejected.append(("geri okuma",
                                 f"{concept} {relation} {target}"))
            continue
        passed.append((concept, relation, target))
    return passed, tally, rejected


def take_parallel(rows, graph, memory, source, write=False, workers=None):
    """İki aşama: paralel eleme, sonra sıralı çelişki denetimi ve yazma."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    size = max(200, len(rows) // (workers * 8) + 1)
    chunks = [rows[at:at + size] for at in range(0, len(rows), size)]
    tally = collections.Counter()
    rejected = collections.defaultdict(list)
    survivors = []
    with multiprocessing.Pool(workers, _prepare, (graph,)) as pool:
        for passed, counted, notes in pool.imap_unordered(_screen, chunks):
            survivors.extend(passed)
            tally.update(counted)
            for kind, example in notes:
                if len(rejected[kind]) < 4:
                    rejected[kind].append(example)
    reasoning = Reasoning(memory)
    for concept, relation, target in survivors:
        known, _ = reasoning.about(concept, relation, target)
        if known is not None and known != (relation not in (CANNOT,)):
            tally["grafla çelişti"] += 1
            if len(rejected["çelişki"]) < 4:
                rejected["çelişki"].append(f"{concept} {relation} {target}")
            continue
        tally["GEÇTİ"] += 1
        if write:
            memory.write(Edge(concept, relation, target, source=source))
    return tally, rejected


def take(rows, memory, source, write=False, language=None):
    """Okumaları denetleyip sayar; `write` ise geçenleri grafa yazar."""
    if language is None:
        from lmm.discovered import words_of
        from lmm.intuition import Intuition
        # Kavram listesi ÖNBELLEKLİ geçiliyor. `memory.concepts` doğrudan
        # verilince `grammar.known()` her çağrıda yeni bir liste alıp kümeye
        # çeviriyor ve kimlik karşılaştırması hiç tutmuyor — `lmm/cli.py`
        # aynı düzeltmeyi taşıyor ama bu betik onu atlıyordu. Profillendi:
        # 600 okumanın 90 saniyesinin 37'si (%41) buradaydı, ve 375 bin
        # okumada bu üç saat demek.
        held = {}

        def concepts():
            marker = len(memory.edges)
            if held.get("at") != marker:
                held["at"], held["names"] = marker, memory.concepts()
            return held["names"]

        language = Intuition(lexicon=memory.lexicon, words=words_of(memory),
                             known=concepts)
        from lmm.grammar import Pattern
        for entry in memory.patterns:
            pattern = Pattern.from_dict(entry)
            language.grammar.add(pattern, first=not pattern.fallback)
    counts = frequency.counts()
    verbs = frequency.verbs()
    morphology = TurkishMorphology()
    reasoning = Reasoning(memory)
    tally = collections.Counter()
    rejected = collections.defaultdict(list)
    for row in rows:
        concept = (row.get("kavram") or "").strip().lower()
        relation = (row.get("ilişki") or row.get("iliski") or "").strip()
        target = (row.get("hedef") or "").strip().lower()
        if relation not in ALLOWED:
            tally["ilişki tanınmadı"] += 1
            rejected["ilişki"].append(relation)
            continue
        if not _a_concept(concept, counts, verbs, morphology):
            tally["kavram değil"] += 1
            rejected["kavram"].append(concept)
            continue
        if not target or len(target) < SHORTEST:
            tally["hedef yok"] += 1
            continue
        # Parça adı iyelik eki ister: graf "kanadı" tutuyor, okuyucu "kanat"
        # veriyor. Sistemin KONUŞTUĞU biçime çevrilip öyle sınanıyor.
        if relation == HAS_PART and not _round_trip(concept, relation,
                                                    target, language):
            target = phrasing.possessed(target)
        if not _round_trip(concept, relation, target, language):
            tally["geri okunamadı"] += 1
            rejected["geri okuma"].append(f"{concept} {relation} {target}")
            continue
        known, _ = reasoning.about(concept, relation, target)
        claimed = relation not in (CANNOT,)
        if known is not None and known != claimed:
            tally["grafla çelişti"] += 1
            rejected["çelişki"].append(f"{concept} {relation} {target}")
            continue
        tally["GEÇTİ"] += 1
        if write:
            memory.write(Edge(concept, relation, target, source=source))
    return tally, rejected


def main(argv):
    given = [a for a in argv[1:] if not a.startswith("--")]
    if not given:
        print(__doc__.strip().splitlines()[-1])
        return 1
    path = given[0]
    if not os.path.exists(path):
        print(f"yok: {path}")
        return 1
    graph = (argv[argv.index("--graf") + 1] if "--graf" in argv
             else "models/graph/birlesik.lmm")
    source = (argv[argv.index("--kaynak") + 1] if "--kaynak" in argv
              else "okuyucu")

    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue

    memory = Memory.load(graph)
    before = len(memory.edges)
    if "--tek" in argv:
        tally, rejected = take(rows, memory, source, write="--yaz" in argv)
    else:
        tally, rejected = take_parallel(rows, graph, memory, source,
                                        write="--yaz" in argv)

    print(f"{path} — {len(rows):,} okuma")
    print(f"{os.path.basename(graph)} — {before:,} olgu\n")
    total = sum(tally.values())
    for name, count in tally.most_common():
        print(f"  {name:<18} {count:>6}  %{count / max(total, 1) * 100:5.1f}")
    for name, examples in rejected.items():
        if examples:
            print(f"\n  reddedilen ({name}) örnekleri:")
            for example in examples[:4]:
                print(f"    {example}")
    if "--yaz" not in argv:
        print("\n  (yazmak için --yaz ekle)")
        return 0
    memory.save(graph)
    print(f"\n  {len(memory.edges) - before:,} yeni olgu -> {graph}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
