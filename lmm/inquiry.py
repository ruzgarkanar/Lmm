"""Soruşturma: bilmediğini fark edip gidip okumak.

Üçüncü şart — *ağırlıklar değişmeden büyümek* — şimdiye kadar edilgen
karşılandı: biri bir şey öğretirse graf büyüyordu. Etkin hâli eksikti.
"Bilmiyorum" demek dürüstlük; **bilmediğini fark edip gidip öğrenmek** başka
bir şey ve LLM'in yapamadığı şey tam da bu. Bir dil modeli okuduğunu ertesi
soruda hatırlamaz; buradaki graf hatırlar, çünkü bilgi ağırlıkta değil.

`harvest.py` bu döngüyü bir dil modeline sorarak kapatıyor. Buradaki fark
küçük görünüp büyük: **soru sorulmuyor, kaynak okunuyor.** Modele sorulunca
gelen şey modelin ezberidir — gösterilebilir bir künyesi yoktur, yanıldığında
nereye bakılacağı bilinmez. Sayfa okununca her olgunun yanında `vikipedi:Kartal`
kalır ve insan gidip bakabilir. Dördüncü şartın (uydurmasın) asıl karşılığı
denetim değil, **künye**dir.

Okumayı kim yapıyor — ölçülerek seçildi, tercih edilerek değil. 24 ansiklopedi
girişi, 445 cümle:

    kendi çerçeve okuyucumuz     100 cümlede 41,6 olgu, hassasiyet ~%17
    derleyici + çapa denetimi    100 cümlede 43,3 olgu, hassasiyet yüksek

Verim aynı, hassasiyet değil. Kendi okuyucumuz basit bildirme cümlelerinde
çalışıyor ama gerçek düzyazıda çöküyor — "metre —can→ dimek", "laik —type→
müslüman". Aynı sayfalardan derleyici "kalp —type→ organ", "buzul —has→ tatlı
su" çıkarıyor.

Bu, gecenin dersinin aynısı: *bir cümlenin ne söylediğine karar vermek anlam
işidir ve yüzey biçiminden okunmuyor* (olgu sınıflandırıcısının üç başarısız
denemesi, bkz. `docs/DENEMELER.md`). Anlamsal yargı dil modelinin yapabildiği
şey; bizim yaptığımız onu denetlemek ve hatırlamak.

Denetim çalışıyor ve boş bir tören değil: aynı dört sayfada dil modelinin 45
iddiası **kaynak metinde geçmediği için** reddedildi. Model uydurduğunda
metne bakıp yakalıyoruz — dördüncü şartın gerçek karşılığı bu.

Okunan şey sayfanın tamamı değil giriş bölümü: tanım kipinin bulunduğu yer
orası.

Bağımlılık yok: istek düz `urllib`. Ağ testlerde hiç kullanılmıyor — hem
getirici hem okuyucu dışarıdan verilebiliyor.
"""
import collections
import json
import re
import urllib.parse
import urllib.request

from lmm.clauses import readable
from lmm.compiler import compile_text
from lmm import frequency
from lmm.frames import read, to_fact
from lmm.intuition import tokenize
from lmm.memory import Edge
from lmm.reading import sentences
from lmm.verbs import graded_pairs

ENDPOINT = "https://tr.wikipedia.org/w/api.php"
# HTTP başlıkları latin-1 taşır; Türkçe harf konursa istek hiç gitmez.
AGENT = "LMM/0.1 (local knowledge graph; personal research)"
TIMEOUT = 15
SOURCE_PREFIX = "vikipedi:"

# Bir soruşturmadan en çok bu kadar olgu alınır. Sınır hassasiyet için değil,
# **niyet** için: sorulan kavram hakkında okunuyor, sayfanın tamamı grafa
# boşaltılmıyor. Sınır aşılırsa `investigate` bunu söylüyor, sessizce kesmiyor.
MOST = 60


class Unreachable(Exception):
    """Kaynağa ulaşılamadı. Bilmemek ile ulaşamamak ayrı şeyler ve ayrı
    söylenmeli — biri grafın eksiği, diğeri ağın."""


def fetch(concept, endpoint=ENDPOINT, timeout=TIMEOUT):
    """Kavramın ansiklopedi girişi. (başlık, metin) ya da bulunamazsa None.

    Yalnızca giriş bölümü isteniyor (`exintro`): tanım kipinin bulunduğu yer
    orası. Yönlendirmeler izleniyor, çünkü "kartallar" ile "Kartal" aynı şeydir
    ve bunu bizim bilmemiz gerekmiyor — ansiklopedi zaten biliyor.
    """
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "extracts",
        "exintro": "1", "explaintext": "1", "redirects": "1",
        "titles": concept,
    })
    request = urllib.request.Request(f"{endpoint}?{query}",
                                     headers={"User-Agent": AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            payload = json.loads(answer.read().decode("utf-8"))
    except Exception as error:                              # noqa: BLE001
        raise Unreachable(f"{concept}: {error}") from error
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract", "").strip()
        if text and "missing" not in page:
            return page.get("title", concept), text
    return None


def _cleaned(text):
    """Parantez içleri ve dipnot işaretleri atılır.

    "(Latince: Aquila)" gibi araya giren parçalar cümlenin durum yapısını
    bozuyor ve okuyucu özneyi kaybediyor. Atmak bilgi kaybettirir ama
    kaybettirdiği bilgi zaten okunamıyordu.
    """
    text = re.sub(r"\([^()]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return re.sub(r"\s+", " ", text)


# Koşaçsız tanım: Türkçe koşacı düşürebiliyor ve ansiklopedi bunu sık yapıyor.
# "Vaşak, ... hayvan türlerinin ortak adı." cümlesinde yüklem eki yok, o yüzden
# çerçeve okuyucusu hiçbir şey bulamıyordu — sayfaya ulaşılıyor, okunuyor ve
# sıfır olgu çıkıyordu. Sessiz bir başarısızlık: organ "çalışıyor" görünüyor.
BARE_DEFINITION = re.compile(
    r"^(?P<konu>[^,.(]{2,40})\s*[,(].{0,200}?"
    r"\bbir\s+(?P<tur>[a-zçğıöşü]+)"
    r"(?:dır|dir|dur|dür|tır|tir|tur|tür)?\b", re.I | re.S)
# "... X türlerinin ortak adı" / "... X cinsinin genel adı" — ansiklopedinin
# en sık ikinci kalıbı ve içinde tür bilgisi var.
COMMON_NAME = re.compile(
    r"^(?P<konu>[^,.(]{2,40})\s*[,(].{0,200}?"
    r"\b(?P<tur>[a-zçğıöşü]+)\s+(?:tür|cins|familya)\w*\s+"
    r"(?:ortak|genel|bilimsel)\s+adı", re.I | re.S)


def definition_in(text, title=None):
    """Tanım kipinden (kavram, tür). Bulunamazsa None.

    Konu TAHMİN EDİLMİYOR: ansiklopedik tanımda başta durur ve zaten başlığı
    biliyoruz. Çerçeve okuyucusundan yalnız TÜR isteniyor — ve o bile koşaç
    düşünce bulunamıyordu.
    """
    first = text.split(".")[0].strip()
    for shape in (BARE_DEFINITION, COMMON_NAME):
        found = shape.search(first)
        if not found:
            continue
        concept = (title or found.group("konu")).strip().lower()
        kind = found.group("tur").strip().lower()
        # Koşaç türün parçası değil: "bir tatlıdır" -> "tatlı". Düzenli ifade
        # eki isteğe bağlı yakalıyor ve yakaladığında gövdeye yapışıyordu.
        for ending in ("dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür"):
            if kind.endswith(ending) and len(kind) > len(ending) + 2:
                kind = kind[: -len(ending)]
                break
        if len(concept) < 2 or len(kind) < 3 or concept == kind:
            continue
        return concept, kind
    return None


def facts_in(text, lexicon=None):
    """Metinden çıkan olgular: (kavram, ilişki, hedef). Tekrarsız, sıralı.

    Sayaçlar derlem tablosundan geliyor, metnin kendisinden değil. Bu ayrım
    ölçülerek öğrenildi: durum eki çözümü ve tür testleri sıklığa dayanıyor ve
    tek bir yazının içinde sıklık hiçbir şey söylemiyor. "Penguenler güney
    yarım kürede yaşar" cümlesinin öznesi `kürede` çıkıyordu, çünkü o yazıda
    "küre" hiç geçmiyor ve bulunma eki tanınamıyordu.

    Tablo yoksa metnin kendi sayacına düşülür — körleşir ama çalışır.
    """
    text = _cleaned(text)
    tokens = re.findall(r"[a-zçğıöşü]+", text.lower())
    words = frequency.counts() or collections.Counter(tokens)
    graded = frequency.grades() or collections.Counter(graded_pairs(tokens))
    found, seen = [], set()
    for whole in sentences(text):
        for piece in readable(whole, lexicon=lexicon):
            frame = read(tokenize(piece), words, lexicon, graded)
            if frame is None:
                continue
            fact = to_fact(frame, lexicon, words, graded)
            if fact is None or fact in seen:
                continue
            seen.add(fact)
            found.append(fact)
    return found


def read_page(text, lexicon=None, reader=None):
    """Sayfadan olgular. Okuyucu verilmezse kendi çerçeve okuyucumuz.

    Dönen: [((kavram, ilişki, hedef), kaç ayrı cümle söyledi)]. Sayı bir tekrar
    değil bir doğrulamadır — aynı olguyu iki cümle söylüyorsa tek okumanın
    hatası olma ihtimali düşer.

    `reader` verildiğinde derleyici çalışır ve çıkardığı her iddia kaynak metne
    karşı denetlenir: metinde geçmeyen iddia yazılmaz. Verilmediğinde sistem
    ağsız çalışmaya devam eder, yalnız hassasiyet düşer.
    """
    text = _cleaned(text)
    if reader is None:
        return [(fact, 1) for fact in facts_in(text, lexicon)]
    lines = list(sentences(text))
    report = compile_text(lines, text, reader)
    return report.facts(), report


class _Claim:
    """Öğrenme döngüsünün beklediği en küçük biçim."""

    def __init__(self, concept, relation, target):
        self.concept, self.relation, self.target = concept, relation, target


def investigate(memory, concept, fetcher=fetch, reader=None, most=MOST,
                reasoning=None):
    """Kavramı soruşturur ve öğrendiklerini grafa yazar.

    Dönen sözlük ne olduğunu anlatır: `kaynak` künyedir — `vikipedi:Kartal`
    gibi, gidip bakılabilir. `okunan` metinden çıkan olgu, `yazılan` grafa
    girebilen, `çelişen` bilinenle çakıştığı için alınmayan, `uydurma` kaynak
    metinde geçmediği için daha yazılmadan elenen.

    Yazma **öğrenme döngüsünden** geçiyor, doğrudan belleğe değil. İlk yazışta
    doğrudan yazılıyordu ve sessiz bir delik açılmıştı: insan "penguen uçamaz"
    demişken ansiklopedi "uçar" diyordu ve graf ikisini birden tutuyordu.
    Doğrudan yazmak kapıyı atlamak demek — güven sıralaması, çelişki denetimi
    ve istisna mantığının hepsi orada. Bir kaynak ne kadar iyi olursa olsun
    insanın söylediğini bozamaz.
    """
    page = fetcher(concept)
    if page is None:
        return {"kaynak": None, "okunan": 0, "yazılan": 0, "çelişen": 0,
                "uydurma": 0, "kesildi": False}
    title, text = page
    source = SOURCE_PREFIX + title
    outcome = read_page(text, memory.lexicon, reader)
    invented = 0
    if isinstance(outcome, tuple):
        facts, report = outcome
        invented = len(report.refused)
    else:
        facts = outcome

    from lmm.learning import CORRECTED, LEARNED, LearningLoop, REINFORCED
    from lmm.reasoning import Reasoning
    loop = LearningLoop(memory, reasoning or Reasoning(memory))
    accepted = (LEARNED, REINFORCED, CORRECTED)

    # Tanım kipi ayrıca okunuyor. Çerçeve okuyucusu koşaçsız cümleyi göremiyor
    # ve ansiklopedi girişlerinin çoğu öyle bitiyor: "Vaşak, ... türlerinin
    # ortak adı." Sayfaya ulaşılıyor, okunuyor ve sıfır olgu çıkıyordu —
    # sessiz bir başarısızlık, çünkü organ dışarıdan çalışıyor görünüyor.
    #
    # Konu tahmin edilmiyor: ansiklopedik tanımda başta durur ve başlığı
    # zaten biliyoruz.
    from lmm.relations import IS_A
    stated = definition_in(text, title)
    if stated is not None:
        pair = (stated[0], IS_A, stated[1])
        # `facts` her zaman (olgu, kaç cümle söyledi) çifti taşıyor. Tanım
        # kipinden geleni 2 sayıyoruz: kaynağın ilk cümlesi ve yapısı birlikte
        # söylüyor, yani tek okumanın hatası olma ihtimali düşük.
        if not any(item[0] == pair for item in facts):
            facts = [(pair, 2)] + list(facts)

    cut = len(facts) > most
    written = clashed = 0
    for fact, _times in facts[:most]:
        # Çoğul kavramı değiştirmez: "ahtapotlar" ile "ahtapot" aynı şey.
        # Soyulmadan yazıldığında graf ikisini ayrı kavram sanıyor ve
        # birinde öğrenilen diğerinde bulunamıyordu.
        name, relation, target = fact
        fact = (_singular(name, memory), relation, target)
        try:
            status, _said, _edge = loop.teach(_Claim(*fact), source=source)
        except Exception:                                   # noqa: BLE001
            clashed += 1
            continue
        if status in accepted:
            written += 1
        else:
            clashed += 1
    return {"kaynak": source, "okunan": len(facts), "yazılan": written,
            "çelişen": clashed, "uydurma": invented, "kesildi": cut}


PLURAL = ("lar", "ler")


def _singular(name, memory):
    """Kavramın tekil hâli — grafta zaten tekili varsa ona bağlanır."""
    known = set(memory.concepts())
    if name in known:
        return name
    for ending in PLURAL:
        if name.endswith(ending) and len(name) > len(ending) + 2:
            single = name[: -len(ending)]
            # `or True` yazılmıştı ve korumayı tamamen iptal ediyordu: çoğul
            # ekiyle BİTEN her kelimenin son üç harfi kesiliyordu. Sonuç,
            # ansiklopedi okumasından grafa var olmayan kavramlar girmesiydi —
            # "popüler" -> `popü`, "moleküler" -> `molekü`, "bahçelievler" ->
            # `bahçeliev`. Aday 2.402 kelimenin 22'si böyle bozuluyordu.
            #
            # Doğru ölçüt zaten yazılmış ama devre dışı bırakılmış: tekili
            # grafta VARSA ona bağlan, yoksa kelimeyi olduğu gibi bırak.
            # Derlem de ikinci bir tanık: gövde kelimeden sık geçiyorsa ek
            # gerçektir.
            if single in known:
                return single
            from lmm import frequency
            counts = frequency.counts()
            if counts and counts.get(single, 0) >= counts.get(name, 0):
                return single
    return name


def unknown_in(memory, concepts):
    """Bu kavramlardan grafın hiç duymadıkları — soruşturulacak olanlar.

    Soruşturma tembeldir ve öyle olmalı: bilinen bir kavram için ağa gidilmez.
    Graf zaten büyüyor; her soruda yeniden okumak onu büyütmez, yavaşlatır.
    """
    return [c for c in concepts if not memory.query(c)]
