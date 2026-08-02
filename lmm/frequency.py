"""Derlem sıklıkları — çalışma anında yüklenen sayım.

`case_of` durum ekini kurala göre değil derleme sorarak çözüyor: kök kelimenin
kendisi kadar sık geçiyorsa ek gerçektir. Fikir işliyor ama sayaç bugüne kadar
yalnız toplu okuma betiklerinde vardı. Çalışma anında hiç yoktu ve eksikliği
sessizdi — soruşturma "Penguenler güney yarım kürede yaşar" cümlesinin öznesini
`kürede` sanıyordu, çünkü tek bir yazının içinde "küre" hiç geçmiyor.

Tablo bir model değil. Satır satır okunabilir, gradyanı yok, eğitimi yok:
kelime ve kaç kez görüldüğü. Derlem değişirse yeniden sayılır
(`scripts/sıklık_çıkar.py`).

Yoksa sistem çalışmaya devam eder, yalnız durum çözümü körleşir. Bu bilinçli:
tablo bir kolaylık, bir bağımlılık değil.
"""
import collections
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(HERE, "data", "tr-siklik.txt")
GRADES = os.path.join(HERE, "data", "tr-derece.txt")
VERBS = os.path.join(HERE, "data", "tr-fiiller.txt")

_loaded = None
_graded = None
_verbs = None


def load(path=TABLE):
    """{kelime: sayı}. Dosya yoksa boş sayaç — çağıran çökmez.

    `Counter` döner, düz sözlük değil: `is_structural` en sık kelimeleri
    `most_common` ile alıyor ve düz sözlükte o yok. Tür uyuşmazlığı sessiz
    değildi, ama sessiz olabilirdi — sayaç arayüzü sözleşmenin parçası.
    """
    found = collections.Counter()
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            word, _, count = line.rstrip("\n").partition("\t")
            if count.isdigit():
                found[word] = int(count)
    return found


def counts(path=TABLE):
    """Bir kez yüklenir, sonra paylaşılır. 8 MB dosya, ~0,3 sn."""
    global _loaded
    if _loaded is None:
        _loaded = load(path)
    return _loaded


def ready(path=TABLE):
    return os.path.exists(path)


def load_grades(path=GRADES):
    """{(derecelendirici, kelime): sayı} — sıfat testinin sayacı.

    Türkçe'de yalnızca sıfat ve zarf derecelenir: "daha zor" olur, "daha banka"
    olmaz. Test oran bakıyor, ham sayıya değil. Sayaç olmadan test her kelimeye
    "sıfat değil" diyordu ve özne seçimi sıfatlara takılıyordu — "evcil kedi
    etobur memeli" cümlesinin öznesi `evcil` çıkıyordu.
    """
    found = collections.Counter()
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and parts[2].isdigit():
                found[(parts[0], parts[1])] = int(parts[2])
    return found


def grades(path=GRADES):
    """Bir kez yüklenir, sonra paylaşılır."""
    global _graded
    if _graded is None:
        _graded = load_grades(path)
    return _graded


def load_verbs(path=VERBS):
    """Derlemden çıkarılmış fiiller: {çekim: (mastar, olumlu mu)}.

    Oturumun kendi sözlüğü yalnızca ÖĞRETİLMİŞ fiilleri bilir ve bu doğru —
    o, sistemin öğrendiği kelime dağarcığı. Ama bir okumanın cümleye uyup
    uymadığını denetlerken daha geniş bir soru sorulur: "bu kelime Türkçe'de
    bir fiil mi?" Cevabı derlem verir, öğretmen değil.
    """
    found = {}
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                infinitive, positive, negative = parts
                found[positive] = (infinitive, True)
                found[negative] = (infinitive, False)
    return found


def verbs(path=VERBS):
    """Bir kez yüklenir, sonra paylaşılır."""
    global _verbs
    if _verbs is None:
        _verbs = load_verbs(path)
    return _verbs
