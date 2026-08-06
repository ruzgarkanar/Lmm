"""Eğitim kümesi: ham metinden okuyucunun öğreneceği biçime.

Okuyucunun öğreneceği iki şey var ve ikisi de aynı cümleden çıkar:

    ne yapmak istiyor      PASS · ASK · WRITE
    hangi harf ne rolde    0 dışarısı · 1 özne · 2 yüklem · 3 değer

Etiketler ELLE YAZILMIYOR, HİZALAMAYLA çıkıyor: elimizde (cümle, olgu) çifti
var; olgunun parçalarının cümlede nerede geçtiği bulunuyor ve harfler o role
işaretleniyor. Hiçbir ek listesi, kalıp ya da sözcük sınıfı kullanılmıyor.

    "Kartal, yırtıcı bir kuş türüdür."   +   (kartal, tür, kuş)
     1111110000000000000033300000000         WRITE

Soru cümlelerinin etiketi de aynı yoldan gelir: bir soru, bir olgunun EKSİK
hâlidir — öznesi söylenmiş, değeri sorulmuştur.

Bu dosyada dile ait hiçbir şey yoktur. Hizalama, dizginin dizgide nerede
geçtiğini aramaktan ibarettir ve hangi dilde olduğunu bilmez.
"""
import json
import os
import random

OUT, SUBJECT, PREDICATE, VALUE = 0, 1, 2, 3
PASS, ASK, WRITE = 0, 1, 2


def span_of(text, name):
    """Adın metindeki yeri — çekimli hâliyle de olsa.

    Kök bulunur, kelime sonuna kadar uzatılır: ek de parçanın kendisi sayılır,
    çünkü ağ ekli hâli görecek ve kökü çıkarmayı KENDİSİ öğrenmeli. Hangi ekin
    olduğu hiç sorulmuyor — sorulsaydı bu dosya bir ek listesine muhtaç olurdu.
    """
    if not name:
        return None
    lowered, name = text.lower(), name.lower()
    at = lowered.find(name)
    if at < 0:
        return None
    end = at + len(name)
    while end < len(text) and (text[end].isalpha() or text[end] == "'"):
        end += 1
    return at, min(end, len(text))


def label(text, subject, predicate, value):
    """Cümlenin her harfi için rol. Hizalanamıyorsa None.

    Özne şart: öznesi bulunamayan cümleden öğrenilecek bir şey yok. Yüklem ve
    değer bulunamazsa yerleri boş kalır — soru cümlelerinde zaten öyle olur.
    """
    roles = [OUT] * len(text)
    place = span_of(text, subject)
    if place is None:
        return None
    for at in range(*place):
        roles[at] = SUBJECT
    for name, role in ((predicate, PREDICATE), (value, VALUE)):
        if not name:
            continue
        where = span_of(text, name)
        if where is None:
            continue
        if any(roles[at] != OUT for at in range(*where)):
            continue        # çakışma: aynı harf iki role verilemez
        for at in range(*where):
            roles[at] = role
    return roles


def from_facts(paths, longest=256, most=None):
    """(cümle, olgu) dosyalarından WRITE örnekleri.

    Cümle başına TEK örnek: aynı cümle birden çok olgu verdiğinde her birini
    ayrı örnek yapmak veriyi kendiyle çeliştiriyordu — eski mimaride ölçüldü,
    kavram+değer birlikte doğruluğu %0,9'da kalmıştı. Bir cümlenin bir öznesi
    vardır; en çok tekrarlanan özne alınır, ötekiler düşer.
    """
    held = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            text = row.get("cümle")
            subject = (row.get("kavram") or "").strip()
            predicate = (row.get("ilişki") or "").strip()
            value = (row.get("hedef") or "").strip()
            if not (text and subject and predicate) or len(text) > longest:
                continue
            held.setdefault(text, []).append((subject, predicate, value))

    found = []
    for text, facts in held.items():
        counted = {}
        for subject, _, _ in facts:
            counted[subject] = counted.get(subject, 0) + 1
        subject = max(counted, key=counted.get)
        chosen = next((one for one in facts if one[0] == subject), None)
        if chosen is None:
            continue
        roles = label(text, *chosen)
        if roles is None:
            continue
        found.append((text, WRITE, roles))
        if most and len(found) >= most:
            return found
    return found


def from_dialogue(paths, longest=256, most=None, keep=0.25):
    """Konuşma satırlarından PASS örnekleri.

    Bir sohbet satırının çoğu bilgi taşımaz: selamlaşma, tepki, geçiş. Bunlar
    PASS'tir ve okuyucunun bunları tanıması, olgu taşıyanları tanıması kadar
    önemli — tanımazsa her söze bir olgu uydurmaya çalışır.

    `keep` payı örnekleniyor çünkü konuşma derlemi olgu derleminden çok daha
    büyük ve dengesiz bir küme, çoğunluk sınıfına çökmeyi öğretir.
    """
    found = []
    chance = random.Random(7)
    for path in paths:
        if not os.path.exists(path):
            continue
        opener = _open(path)
        for line in opener:
            text = line.strip()
            if not (4 < len(text) <= longest):
                continue
            if chance.random() > keep:
                continue
            found.append((text, PASS, [OUT] * len(text)))
            if most and len(found) >= most:
                return found
    return found


def _open(path):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def alphabet(rows):
    """Derlemin kendi harfleri — hiçbir yerde bildirilmiyor, sayılıyor."""
    seen = set()
    for text, _, _ in rows:
        seen.update(text)
    return {letter: at + 1 for at, letter in enumerate(sorted(seen))}


def build(fact_paths, dialogue_paths=(), most_facts=None, most_dialogue=None,
          seed=7):
    """Karışık, dengeli, karıştırılmış eğitim kümesi."""
    rows = from_facts(fact_paths, most=most_facts)
    if dialogue_paths:
        share = most_dialogue or max(1, len(rows) // 3)
        rows += from_dialogue(dialogue_paths, most=share)
    random.Random(seed).shuffle(rows)
    return rows
