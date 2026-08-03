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
from lmm.memory import CycleError, Edge, Memory             # noqa: E402
from lmm.reasoning import Reasoning                         # noqa: E402
from lmm.relations import CAN, CANNOT, HAS_PART, HAS_PROPERTY, IS_A  # noqa: E402
from lmm.turkish import TurkishMorphology                   # noqa: E402
from lmm.verbs import is_structural                         # noqa: E402

# Okuyucunun kullanabileceği ilişkiler. Kapalı bir küme: okuyucu yeni bir
# ilişki icat edemez, yalnızca envanterdekilerden birini gösterebilir.
ALLOWED = (IS_A, HAS_PROPERTY, CAN, CANNOT, HAS_PART)

SHORTEST = 3


def _a_target(target, relation, memory, counts, grades, morphology):
    """Hedef bu ilişkiye yakışıyor mu — ve temiz mi.

    Kabul edilenlerden dört bin tanesi elle okundu ve üç sistematik kusur çıktı:

        tophane --type--> semttir            koşaç soyulmamış
        commodore --property--> 1953         yıl nitelik değil
        johann --property--> yazar           ad nitelik değil

    İlk ikisi biçim, üçüncüsü tür sorunu ve üçüncüsü zordu.

    Önce `verbs.is_adjective` denendi ve ÖLÇÜLÜP çürütüldü: o test
    DERECELENEBİLİRLİĞİ ölçüyor, sıfatlığı değil. Renkler derecelenmez ve
    sayılar bunu açıkça söylüyor —

        hızlı 0,279   güzel 0,319   sert 0,157      <- geçiyor
        beyaz 0,0059  siyah 0,0031  yuvarlak 0,011  <- kalıyor
        ülke  0,0061  yazar 0,0037  kuzey 0,0040    <- ad, ama beyaz'ın üstünde

    `ülke` `beyaz`dan yüksek. Yani hiçbir eşik ikisini ayıramaz; test yanlış
    testti. `is_noun` de ayırmıyor, çünkü Türkçede sıfat da durum eki alır.

    Ayıran şey grafın KENDİ bilgisi: bir nitelik, hakkında konuşulan bir şey
    değildir. `yazar`, `ülke`, `göl`, `cihaz` grafta kavram; `beyaz`, `siyah`,
    `hızlı`, `sert` değil. Ölçüt graf büyüdükçe GÜÇLENİYOR — her yeni tür
    olgusu bir sözcüğü daha kavram yapıyor ve nitelik yuvasından çıkarıyor.
    """
    target = morphology.strip_copula(target)
    if not target or target.isdigit() or any(ch.isdigit() for ch in target):
        return None
    # İyelik eki soyuluyor — AMA yalnız soyulmuş hâli grafın tanıdığı bir
    # kavramsa. Ölçüldü: okuyucu bazen tamlama içindeki biçimi veriyor
    # ("van gölü bir GÖLÜDÜR") ve graf `gölü` ile `göl`ü iki AYRI tür sanıp
    # olmayan bir belirsizlik bildiriyordu:
    #
    #   van gölü nedir -> "birden fazla şeye işaret ediyor: bir gölü ve bir göl"
    #
    # Koşul şart: "kedi" kelimesinden "ked" çıkarmak yanlış olurdu. Kararı
    # yine graf veriyor, kural değil.
    known = set(memory.concepts())
    for ending in sorted(getattr(morphology, "possessive_suffixes", ()),
                         key=len, reverse=True):
        if not target.endswith(ending) or len(target) - len(ending) < 3:
            continue
        stem = target[: -len(ending)]
        softened = (stem[:-1] + getattr(morphology, "softened", {}).get(
            stem[-1], stem[-1])) if stem else stem
        for candidate in (stem, softened):
            # KOŞUL İKİ TANE, ve ikincisi bu gece eklendi. "Grafta var mı"
            # tek başına DÖNGÜSEL: bir kez giren çöp kendini besliyor.
            # `bat` bir kez kavram oldu, ondan sonra her `batı` ona soyuldu;
            # `resmi` -> `resm` de aynı yoldan, ve sınav cevaplarında
            # görünüyorlardı:
            #
            #     güneşli nasıldır -> "güneşli bat, yeni, resm, eski..."
            #
            # İkinci koşul DERLEME soruyor, grafa değil: gerçek bir iyelik
            # soyulduğunda çıplak ad da yaygındır, çünkü o da bir kelimedir.
            # Ölçüldü:
            #
            #     gölü -> göl     0,62     batı  -> bat    0,03
            #     kanadı -> kanat 2,64     resmi -> resm   0,08
            #     adası -> ada    1,44     kişi  -> kiş    0,00
            #
            # İki küme arasında yedi kat boşluk var; eşik oraya konuyor.
            # Kelime listesi değil, sayım.
            if candidate in known and _a_word(candidate, target, counts):
                return candidate        # çıplak ad TERCİH ediliyor
        break
    if relation == HAS_PROPERTY and target not in set(memory.properties()):
        if target in known:
            return None
    return target


# Çıplak adın, çekimli hâline göre en az bu kadar sık geçmesi gerekiyor.
# Doğru soymalarda en düşük 0,62, yanlışlarda en yüksek 0,08 ölçüldü.
STEM_SHARE = 0.25


def _a_word(stem, inflected, counts):
    """Bu gövde derlemde KENDİ BAŞINA bir kelime gibi duruyor mu."""
    return counts.get(stem, 0) >= STEM_SHARE * counts.get(inflected, 0)


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


def _said(concept, relation, target, object=None, role=None):
    """Olguyu sistemin kendi ağzından bir cümleye çevirir."""
    if relation == IS_A:
        return phrasing.is_a_clause(concept, target)
    if relation == HAS_PROPERTY:
        return phrasing.property_clause(concept, target, True, object, role)
    if relation == HAS_PART:
        return phrasing.part_clause(concept, target, True)
    return phrasing.ability_clause(concept, target, relation == CAN,
                                   object, role)


def _an_infinitive(word, morphology):
    """Bu hedef bir fiil MASTARI mı — biçimden, listeden değil."""
    endings = getattr(morphology, "infinitive_suffixes", ())
    return any(word.endswith(end) and len(word) > len(end) + 1
               for end in endings)


def _verb_known(infinitive, lexicon):
    """Sözlük ya da derlem bu fiili GERÇEKTEN biliyor mu.

    `lexicon.surface()` sorulamaz: bilmediğinde sessizce mastardan türetip
    döndürüyor, yani "biliyor musun" sorusuna her zaman evet diyor. İlk
    yazışta ona sordum ve öğretme yolu hiç açılmadı — kurulmuş ama hiç
    çalışmayan bir kapı, çalıştığı sanıldığı için hiç olmamasından kötü.

    Bilinen bir fiilin çekimi ÜSTÜNE YAZILMAMALI: düzensizleri derlem doğru
    biliyor ("gitmek" -> "gider"), türetme yanlış verebilir.
    """
    if (infinitive, True) in getattr(lexicon, "_forms", {}):
        return True
    return any(name == infinitive
               for name, _ in frequency.verbs().values())


def _round_trip(concept, relation, target, language, object=None, role=None):
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
    intent = language.understand(_said(concept, relation, target,
                                       object, role))
    if not (intent.kind == TEACH and intent.concept == concept
            and intent.relation == relation
            and str(intent.target or "").startswith(str(target)[:4])):
        return False
    # NESNE DE GERİ OKUNMALI. Yoksa kapı nesnesiz cümleyi doğrulayıp nesneli
    # olguyu yazardı: yazılan şey sınanandan fazla olurdu ve fazlası
    # denetimsiz kalırdı. Ayrıştırıcı nesneyi çekimli döndürüyor ("kanı"),
    # okuyucu eksiz veriyor ("kan") — baş harflerden eşleşme yeterli.
    if object:
        read = str(intent.object or "")
        return bool(read) and read.startswith(str(object)[:4])
    return True


_SHARED = {}


def repeated_names(rows, least=2):
    """Okuyucunun EN AZ İKİ kez kavram diye adlandırdığı çok kelimeli adlar.

    "the beatles bir gruptur" cümlesinde ayrıştırıcı kavramı `the` diye
    okuyup gerisini hedefe katıyordu: çok kelimeli bir ad ancak TANIKLI ise
    kabul ediliyor (graf biliyor, biçimbilim işaretliyor ya da derlem
    tanıklıyor) ve "the beatles" hiçbirinden geçmiyor. Doğru bir olgu, kapıda
    bu yüzden ölüyordu — ölçüldü, geri okuma redlerinin büyük kısmı bu.

    Okuyucunun kendi tekrarı dördüncü tanık: aynı adı iki ayrı olguda özne
    yapmışsa o bir şeydir. Bir kez geçen ad rastlantı olabilir; aynı ölçüt
    kalıp çıkarımında ve `grammar.induce`'da da kullanılıyor.

    Tanıklık YALNIZ bu denetim için geçerli — grafa yazılan şey yine tek tek
    kapıdan geçen olgular.
    """
    counted = collections.Counter()
    for row in rows:
        name = (row.get("kavram") or "").strip().lower()
        if " " in name and 2 <= len(name.split()) <= 3:
            counted[name] += 1
    return {name for name, times in counted.items() if times >= least}


def _prepare(graph, extra=()):
    """Her işçi sürecin kendi belleği ve dil organı — bir kez kuruluyor."""
    from lmm.discovered import words_of
    from lmm.grammar import Pattern
    from lmm.intuition import Intuition
    memory = Memory.load(graph)
    held = {}

    def concepts():
        marker = len(memory.edges)
        if held.get("at") != marker:
            names = memory.concepts()
            if extra:
                # Aynı LİSTE nesnesi korunuyor: dilbilgisi kümesini o nesneye
                # asıyor ve yenisini kurmak önbelleği her cümlede düşürürdü.
                names.extend(name for name in sorted(extra)
                             if name not in set(names))
            held["at"], held["names"] = marker, names
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
    grades = frequency.grades()
    verbs = frequency.verbs()
    morphology = TurkishMorphology()
    passed, tally, rejected = [], collections.Counter(), []
    for row in chunk:
        concept = (row.get("kavram") or "").strip().lower()
        relation = (row.get("ilişki") or row.get("iliski") or "").strip()
        target = (row.get("hedef") or "").strip().lower()
        object = (row.get("nesne") or "").strip().lower() or None
        role = (row.get("rol") or "").strip().lower() or None
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
        target = _a_target(target, relation, memory, counts, grades, morphology)
        if not target or len(target) < SHORTEST:
            tally["hedef yakışmadı"] += 1
            if len(rejected) < 4:
                rejected.append(("hedef", f"{concept} {relation} "
                                          f"{row.get('hedef')}"))
            continue
        if relation == HAS_PART and not _round_trip(concept, relation,
                                                    target, language):
            target = phrasing.possessed(target)
        # BİLMEDİĞİ FİİLİ OKUYAMIYOR. Ölçüldü: 306 eylem olgusunun 77'si
        # "epinefrin dilde uygular" gibi kusursuz bir cümle üretti ve
        # ayrıştırıcı UNKNOWN dedi — `uygular` derlemde geçmiyor, sözlükte
        # yok, dolayısıyla o cümle kurulamıyor ve doğru bir olgu kapıda
        # reddediliyor. Sistemin kendi ağzı kendi bilgisini eliyordu.
        #
        # Çözüm uydurmak değil TÜRETMEK: okuyucu mastarı veriyor, biçimbilim
        # geniş zamanı zaten kuruyor (`phrasing.aorist`), ve sözcük dağarcığı
        # yeniden eğitimsiz büyüyor — bu projenin üçüncü şartı.
        #
        # Sınır: yalnız mastar EKİYLE biten bir hedef fiil sayılıyor. Fiil
        # öğrenmek olguyu geçirmiyor; olgu yine geri okunacak, çelişki
        # denetiminden geçecek ve ancak öyle yazılacak.
        if (relation in (CAN, CANNOT) and _an_infinitive(target, morphology)
                and not _verb_known(target, memory.lexicon)):
            memory.learn_word(target, phrasing.aorist(target, True),
                              phrasing.aorist(target, False))
        # Nesne AYRI bir iddia ve ayrı sınanıyor. Geri okunmuyorsa olgu
        # atılmıyor, nesnesi düşürülüyor: "kalp pompalar" hâlâ doğru ve
        # kaynağı var. Sınanamayanı yazmak ile sınananı atmak arasındaki
        # doğru yer burası.
        if object and not _round_trip(concept, relation, target, language,
                                      object, role):
            tally["nesne geri okunamadı"] += 1
            object = role = None
        if not _round_trip(concept, relation, target, language):
            tally["geri okunamadı"] += 1
            if len(rejected) < 4:
                rejected.append(("geri okuma",
                                 f"{concept} {relation} {target}"))
            continue
        # Cümlenin kendisi değil KİMLİĞİ taşınıyor: aynı cümleden çıkan
        # olguları eşleştirmeye yetiyor ve grafı şişirmiyor.
        context = row.get("cümle")
        # NESNE DE TAŞINMALI. İlk yazışta demet dörtlüydü ve nesne paralel
        # elemede doğrulanıp orada kalıyordu: kapı 79 nesneyi sınayıp
        # geçirdi, yazma yolu hiçbirini görmedi, graf yine %0 nesneli çıktı.
        # Sınanan ile yazılan aynı şey olmalı.
        passed.append((concept, relation, target,
                       hash(context) & 0xFFFFFFFF if context else None,
                       object, role))
    return passed, tally, rejected


def take_parallel(rows, graph, memory, source, write=False, workers=None):
    """İki aşama: paralel eleme, sonra sıralı çelişki denetimi ve yazma."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    size = max(200, len(rows) // (workers * 8) + 1)
    chunks = [rows[at:at + size] for at in range(0, len(rows), size)]
    tally = collections.Counter()
    rejected = collections.defaultdict(list)
    survivors = []
    names = repeated_names(rows)
    with multiprocessing.Pool(workers, _prepare, (graph, names)) as pool:
        for passed, counted, notes in pool.imap_unordered(_screen, chunks):
            survivors.extend(passed)
            tally.update(counted)
            for kind, example in notes:
                if len(rejected[kind]) < 4:
                    rejected[kind].append(example)
    reasoning = Reasoning(memory)
    for concept, relation, target, context, object, role in survivors:
        known, _ = reasoning.about(concept, relation, target)
        if known is not None and known != (relation not in (CANNOT,)):
            tally["grafla çelişti"] += 1
            if len(rejected["çelişki"]) < 4:
                rejected["çelişki"].append(f"{concept} {relation} {target}")
            continue
        if write:
            # Döngü grafın kendi kapısı ve DOĞRU çalışıyor: "medulla bir
            # omurilik soğanıdır" + "omurilik soğanı bir medulladır" bir
            # kalıtım halkası kurar ve `ancestors` orada sonsuza dek döner.
            # Ama hata yakalanmıyordu ve TÜM ALIM ÇÖKÜYORDU — iki buçuk
            # saatlik okumanın ardından tek bir kötü üçlü her şeyi
            # durduruyordu. Tek bir olgunun reddi, işin tamamının kaybı
            # olmamalı.
            try:
                memory.write(Edge(concept, relation, target, source=source,
                                  context=context, object=object, role=role))
            except CycleError:
                tally["döngü kurardı"] += 1
                if len(rejected["döngü"]) < 4:
                    rejected["döngü"].append(f"{concept} {relation} {target}")
                continue
        tally["GEÇTİ"] += 1
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
        object = (row.get("nesne") or "").strip().lower() or None
        role = (row.get("rol") or "").strip().lower() or None
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
        target = _a_target(target, relation, memory, counts,
                           frequency.grades(), morphology)
        if not target or len(target) < SHORTEST:
            tally["hedef yakışmadı"] += 1
            continue
        # Parça adı iyelik eki ister: graf "kanadı" tutuyor, okuyucu "kanat"
        # veriyor. Sistemin KONUŞTUĞU biçime çevrilip öyle sınanıyor.
        if relation == HAS_PART and not _round_trip(concept, relation,
                                                    target, language):
            target = phrasing.possessed(target)
        # BİLMEDİĞİ FİİLİ OKUYAMIYOR. Ölçüldü: 306 eylem olgusunun 77'si
        # "epinefrin dilde uygular" gibi kusursuz bir cümle üretti ve
        # ayrıştırıcı UNKNOWN dedi — `uygular` derlemde geçmiyor, sözlükte
        # yok, dolayısıyla o cümle kurulamıyor ve doğru bir olgu kapıda
        # reddediliyor. Sistemin kendi ağzı kendi bilgisini eliyordu.
        #
        # Çözüm uydurmak değil TÜRETMEK: okuyucu mastarı veriyor, biçimbilim
        # geniş zamanı zaten kuruyor (`phrasing.aorist`), ve sözcük dağarcığı
        # yeniden eğitimsiz büyüyor — bu projenin üçüncü şartı.
        #
        # Sınır: yalnız mastar EKİYLE biten bir hedef fiil sayılıyor. Fiil
        # öğrenmek olguyu geçirmiyor; olgu yine geri okunacak, çelişki
        # denetiminden geçecek ve ancak öyle yazılacak.
        if (relation in (CAN, CANNOT) and _an_infinitive(target, morphology)
                and not _verb_known(target, memory.lexicon)):
            memory.learn_word(target, phrasing.aorist(target, True),
                              phrasing.aorist(target, False))
        # Nesne AYRI bir iddia ve ayrı sınanıyor. Geri okunmuyorsa olgu
        # atılmıyor, nesnesi düşürülüyor: "kalp pompalar" hâlâ doğru ve
        # kaynağı var. Sınanamayanı yazmak ile sınananı atmak arasındaki
        # doğru yer burası.
        if object and not _round_trip(concept, relation, target, language,
                                      object, role):
            tally["nesne geri okunamadı"] += 1
            object = role = None
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
        if write:
            try:
                memory.write(Edge(concept, relation, target, source=source,
                                  object=object, role=role))
            except CycleError:
                tally["döngü kurardı"] += 1
                continue
        tally["GEÇTİ"] += 1
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
