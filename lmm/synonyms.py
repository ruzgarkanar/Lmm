"""Eş anlamlılığı okumak: aynı şeyin başka söylenişini öğrenmek.

Ayrıştırıcının tavanı ölçüldü ve sebebi bulundu: şekil kodlayıcı niyeti taşıyan
kelimeyi siliyor. "kartalın özellikleri neler" ile "kartalın yavruları neler"
birebir aynı özellik vektörünü veriyor, dolayısıyla hiçbir sınıflandırıcı
ayıramaz — ne kadar veri verilirse verilsin.

İlk çözüm önerisi kümelemeydi ve **ölçülüp çürütüldü**: dağılımsal benzerlik
somut eş anlamlıları buluyor (`doktor ~ hekim 0,466`) ama işlevsel/soyut
olanlarda çöküyor (`özellik ~ nitelik 0,192`, oysa `parça ~ organ` yalnızca
0,067 — yani aynı gruptaki çift, farklı gruptakinin altında). Niyet kelimeleri
tam da işlevsel olanlar. Sebebi de anlaşılır: soyut kelimeler her yerde geçer,
dağılımları yayvandır.

Buradaki yol başka: **eş anlamlılık bir olgudur, o hâlde yeri graftır.** Ağırlık
ya da küme değil — künyeli, denetlenebilir, düzeltilebilir bir kayıt.

İki işaret okunuyor ve ikisi de Vikisözlük'ten:

    tanım bağlantısı   tanım satırının kendisi tek bir bağlantıysa o bir eş
                       anlamlıdır — "anlatmak # [[nakletmek]]"
    çeviri kesişimi    aynı yabancı kelimeye giden iki Türkçe kelime eş
                       anlamlıdır — "özellik" ve "nitelik" ikisi de İngilizce
                       *property*, Fransızca *qualité*

İkincisi ölçüldü ve birincisinden güçlü çıktı; üstelik denetimler tam sıfır
verdi:

    özellik ~ nitelik    %62,5        anlatmak ~ koşmak    %0,0
    hekim ~ doktor       %62,5        özellik ~ koşmak     %0,0
    anlatmak ~ bahsetmek %25,0

Yöntem yeni değil ve olduğunu iddia etmiyoruz: çeviri pivotu Bannard &
Callison-Burch (2005), PPDB (2013) ve BabelNet (2010) aynı fikri kullanıyor.
Buradaki fark yöntemde değil, sonucun nereye yazıldığında — ağırlığa değil,
kaynağı gösterilebilir bir kayda.

Bağımlılık yok: istek düz `urllib`. Ağ testlerde kullanılmıyor — getirici
dışarıdan verilebiliyor.
"""
import collections
import json
import re
import urllib.parse
import urllib.request

ENDPOINT = "https://tr.wiktionary.org/w/api.php"
AGENT = "LMM/0.1 (local knowledge graph; personal research)"
TIMEOUT = 15
SOURCE_PREFIX = "vikisözlük:"

# Kaç dilde ortak çeviri paylaşılırsa eş anlamlı sayılır. Ölçülen ayrım keskin:
# gerçek çiftler %12,5-62,5, alakasız çiftler tam %0. Eşik sıfırın hemen
# üstünde duruyor, çünkü ayıran şey oranın büyüklüğü değil VARLIĞI.
LEAST_SHARE = 0.10
LEAST_LANGUAGES = 3     # tek dilde kesişme tesadüf olabilir


class Unreachable(Exception):
    """Sözlüğe ulaşılamadı. Bilmemek ile ulaşamamak ayrı şeyler."""


def fetch(word, endpoint=ENDPOINT, timeout=TIMEOUT):
    """Kelimenin sözlük maddesi, ham hâliyle. Yoksa None."""
    query = urllib.parse.urlencode({
        "action": "parse", "format": "json", "page": word,
        "prop": "wikitext", "redirects": "1",
    })
    request = urllib.request.Request(f"{endpoint}?{query}",
                                     headers={"User-Agent": AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            payload = json.loads(answer.read().decode("utf-8"))
    except Exception as error:                              # noqa: BLE001
        raise Unreachable(f"{word}: {error}") from error
    if "parse" not in payload:
        return None
    return payload["parse"]["wikitext"]["*"]


def translations(entry):
    """{dil: {kelime}} — maddedeki çeviriler."""
    found = collections.defaultdict(set)
    for code, word in re.findall(r"\{\{ç\|([a-z\-]+)\|([^}|]+)", entry or ""):
        found[code].add(word.strip().lower())
    return found


def stated(entry):
    """Maddenin kendi söylediği eş anlamlılar.

    İki yer: başlıklı "Eş anlamlılar" bölümü, ve tanımın kendisi tek bir
    bağlantıysa o satır. İkincisi Türkçe Vikisözlük'te daha yaygın.
    """
    if not entry:
        return []
    found = []
    section = re.search(r"={2,}\s*Eş ?anlamlı[^=]*={2,}(.*?)(?:\n={2,}|\Z)",
                        entry, re.S | re.I)
    if section:
        found.extend(re.findall(r"\[\[([^\]|#]+)", section.group(1)))
    for line in entry.splitlines():
        line = line.strip()
        if not line.startswith("#") or line.startswith(("#*", "#:")):
            continue
        body = line.lstrip("# ").strip()
        whole = re.fullmatch(r"\[\[([^\]|#]+)\]\]\.?", body)
        if whole:
            found.append(whole.group(1))
    return [word.strip().lower() for word in found if word.strip()]


def shared_translations(first, second):
    """İki maddenin çeviri kesişimi: (oran, hangi dillerde).

    Oran, ORTAK dillere göre alınıyor — bir kelimenin kırk dilde, diğerinin
    üç dilde çevirisi olması onları uzaklaştırmamalı. Ölçülen şey "ikisinin de
    çevirisi olan dillerde ne sıklıkla aynı kelimeye gidiyorlar".
    """
    left, right = translations(first), translations(second)
    languages = set(left) & set(right)
    if len(languages) < LEAST_LANGUAGES:
        return 0.0, []
    agreeing = [(code, left[code] & right[code])
                for code in sorted(languages) if left[code] & right[code]]
    return len(agreeing) / len(languages), agreeing


def alike(first_entry, second_entry, first, second):
    """Bu iki kelime eş anlamlı mı, ve neden. (evet_mi, gerekçe).

    Gerekçe yazılıyor çünkü bir olgunun künyesi kadar önemli olan şey, hangi
    kanıtla yazıldığı. "vikisözlük çeviri: en, fr, uz" cümlesi insanın gidip
    bakabileceği bir şey; bir küme numarası değil.
    """
    said = stated(first_entry)
    if second in said:
        return True, "vikisözlük tanımı"
    share, agreeing = shared_translations(first_entry, second_entry)
    if share >= LEAST_SHARE:
        codes = ", ".join(code for code, _ in agreeing[:5])
        return True, f"vikisözlük çeviri ({codes}) — %{share*100:.0f}"
    return False, ""


def read(words, fetcher=fetch):
    """Bu kelimeler arasındaki eş anlamlılıklar: [(a, b, gerekçe)].

    Her madde bir kez getiriliyor. Sözlükte olmayan kelime bir hata değil bir
    sonuçtur — atlanır.
    """
    entries = {}
    for word in words:
        try:
            entries[word] = fetcher(word)
        except Unreachable:
            entries[word] = None
    found = []
    ordered = [w for w in words if entries.get(w)]
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            yes, why = alike(entries[first], entries[second], first, second)
            if yes:
                found.append((first, second, why))
    return found


INFINITIVE = ("mak", "mek")


def stem(word):
    """Mastarı gövdeye indirir: "anlatmak" -> "anlat".

    Eş anlamlılık gövde düzeyinde saklanıyor, mastar düzeyinde değil. Sebebi
    ölçüldü: kalıplardaki sözcükler zaten gövde ("anlat"), ve fiil keşfi bazı
    mastarları yanlış çıkarıyor ("bahseder" -> "bahsedemek", oysa "bahsetmek").
    Gövde ikisinin de ortak paydası ve çekimli hâller ona önek olarak uyuyor.
    """
    for ending in INFINITIVE:
        if word.endswith(ending) and len(word) > len(ending) + 1:
            return word[: -len(ending)]
    return word


def into(memory, words, fetcher=fetch):
    """Bulunan eş anlamlılıkları grafa yazar. Kaç tanesinin yazıldığını döner.

    Kayıt gövde düzeyinde: kalıplardaki sözcükler gövde ("anlat") ve çekimli
    hâller ona önek olarak uyuyor ("bahseder" -> "bahset").
    """
    from lmm.memory import Edge
    from lmm.relations import SAME_AS
    written = 0
    for first, second, why in read(words, fetcher):
        try:
            memory.write(Edge(stem(first), SAME_AS, stem(second),
                              source=SOURCE_PREFIX + why))
            written += 1
        except Exception:                                   # noqa: BLE001
            pass
    return written


def main(argv):
    """python3 -m lmm.synonyms model.lmm anlatmak bahsetmek özellik nitelik"""
    import sys
    import time
    from lmm.memory import Memory
    if len(argv) < 3:
        print(main.__doc__)
        return 1
    path, words = argv[1], [w.lower() for w in argv[2:]]
    memory = Memory.load(path)
    before = len(memory.edges)
    print(f"{len(words)} kelime soruluyor...")

    def polite(word):
        found = fetch(word)
        time.sleep(1.1)         # sözlüğe nazik davranmak
        return found

    written = into(memory, words, polite)
    memory.save(path)
    print(f"{written} eş anlamlılık yazıldı  ({before} -> {len(memory.edges)})")
    from lmm.relations import SAME_AS
    for edge in memory.edges:
        if edge.relation == SAME_AS:
            print(f"  {edge.concept} = {edge.target}   [{edge.source}]")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
