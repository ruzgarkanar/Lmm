"""Ansiklopedi tanımlarından olgu çıkarma.

Genel düzyazıyı ayrıştırmak, ölçtüğümüz kadarıyla, yüz cümlede 2,8 olgu veriyor
ve gevşetildiğinde çöp üretiyor. Ama bir tür cümle var ki yapısı sabit: bir
ansiklopedi maddesinin açılışı. "Futbol, ... bir takım sporudur." Bu bir tanım
ve tanımın şekli baştan bellidir.

DBpedia milyonlarca olguyu tam olarak bu düzenlilikten çıkardı — bütün metni
anlamaya çalışarak değil, anlamı zaten belli olan cümleyi seçerek. Buradaki iş
de o: 126 bin tanım cümlesinden yaklaşık 64 bin "X bir Y'dir" olgusu.

İşin tamamı yüklemin baş ismini bulmakta ve orası Türkçe'de eklerin altında
kalıyor. İki tuzak ölçüldü ve ikisi de sessizce yanlış bilgi üretiyordu:

    "cinsidir"  -> "cinsi" -> kör soyma "cin" verir; doğrusu "cins"
    "güveci"    -> kör soyma "güvec" verir; ünsüz yumuşaması geri alınmalı

İlki 3621 sahte "cin" üretmişti. İkisi de morfolojinin zaten bildiği şeyler;
eksik olan onlara sormaktı.
"""
import re

from lmm.intuition import lower

VOWELS = set("aeıioöuü")
HARDENED = {"c": "ç", "ğ": "k", "b": "p", "d": "t"}
HEAD = re.compile(r"\b(?:bir|birer)\b\s+(.{2,70}?)\s*\.?\s*$")
# "Kartal, ... bir kuştur." — virgülden önceki ad, "bir"den sonraki tür.
OPENING = re.compile(r"^(.{2,40}?),\s+(.{10,240}?\b(?:bir|birer)\s+.{2,60}?"
                     r"(?:dir|dır|dur|dür|tir|tır|tur|tür))\b")

MINIMUM = 3
MAXIMUM = 25


def is_definition(sentence):
    return bool(OPENING.match(sentence))


def strip_possessive(word):
    """"arabası" -> "araba", ama "cinsi" -> "cins" ve "güveci" -> "güveç".

    "-sı" yalnızca ünlüyle biten gövdelere gelir. Ünsüzden sonra gelen "s"
    gövdenin kendisidir ve soyulursa kelime bozulur.
    """
    if len(word) < 4:
        return word
    if word[-2:] in ("sı", "si", "su", "sü") and word[-3] in VOWELS:
        return word[:-2]
    if word[-1] in "ıiuü" and word[-2] not in VOWELS:
        stem = word[:-1]
        return stem[:-1] + HARDENED.get(stem[-1], stem[-1])
    return word


def head_noun(sentence, morphology):
    """Yüklemin baş ismi — tanımın söylediği tür."""
    match = HEAD.search(sentence.rstrip(". "))
    if not match:
        return None
    words = match.group(1).split()
    if not words:
        return None
    word = lower(words[-1].strip("().,;:\"'«»"))
    if not word or not word.replace("'", "").isalpha():
        return None
    if morphology.has_copula(word):
        word = morphology.strip_copula(word)
    word = strip_possessive(word)
    return word if MINIMUM <= len(word) <= MAXIMUM else None


def subject(sentence):
    """Tanımlanan şey: ilk virgülden önceki ad.

    Birden çok kelimeyse atlanır. "Çayır papatyası" bir kavram olabilir ama
    hangi kelimenin başı olduğuna karar vermek ayrı bir iştir ve tahminle
    yapılırsa graf bozulur.
    """
    name = lower(sentence.split(",")[0].strip())
    name = re.sub(r"\s*\([^)]*\)", "", name).strip()
    if not name or " " in name or not name.replace("'", "").replace("-", "").isalpha():
        return None
    return name if MINIMUM <= len(name) <= 40 else None


def extract(sentence, morphology):
    """(kavram, tür) ya da None. Emin olunamayan her şey None."""
    if not is_definition(sentence):
        return None
    name = subject(sentence)
    kind = head_noun(sentence, morphology)
    if not name or not kind or name == kind:
        return None
    return name, kind


def extract_all(lines, morphology):
    found = []
    for line in lines:
        pair = extract(line.strip(), morphology)
        if pair:
            found.append(pair)
    return found
