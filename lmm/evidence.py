"""KANIT DEPOSU — dokümanın cümleleri + fold'lu ters-indeks (embedding YOK).

Temsil darlığının çözümünün yarısı: graf yapıyı tutar (indeks, türetim,
çelişki, çok-adım), CÜMLE kanıtı tutar (sayılar, aralıklar, nüans — üçlüye
sığmayan her şey). Cevap graf-öncelikli kurulur ama kanıt cümleleri de
bağlama girer; kapı yine denetler. RAG'den farkı: getirme embedding-benzerlik
kumarı değil, fold'lu sözcük kesişimi — deterministik, dil-bağımsız
(fold + önek toleransı), açıklanabilir ("şu sözcükler şu cümlede geçti").

Kalıcılık: bellek yolunun yanına `<yol>.kanit` (JSON) — v3 çekirdeğine
dokunulmaz, format kapalı kalır.
"""
import json
import os
import re
import unicodedata

from v3.dataset import fold

_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text):
    """İçerik sözcükleri: fold'lu, birleşen-imsiz, >=3 harf YA DA rakam.
    Rakam istisnası kritik: "Tier 3"ün 3'ü elenirse kapsama kapısı sayı
    değişimini ("Tier 1" uydurmasını) göremez."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [fold(w) for w in _WORD.findall(text)
            if len(w) >= 3 or w.isdigit()]


def covered(answer, given_text):
    """Cevabın TÜM içerik-sözcükleri verilen metinden mi geliyor — ÖNEK
    toleranslı (çekim: "bandındadır"~"bandında"; tam-eşleşme kapıyı gereksiz
    kapatıyordu). Yeni içerik-sözcük yok = yeni iddia yok = uydurma imkânsız.
    Rakamlar aynen eşleşmeli (sayı değişimi = uydurma)."""
    given = set(_words(given_text))
    words = set(_words(answer))
    if not words:
        return False
    for w in words:
        if w in given:
            continue
        if w.isdigit():
            return False                    # rakamda tolerans YOK
        if not any((len(g) >= 4 and w.startswith(g))
                   or (len(w) >= 4 and g.startswith(w)) for g in given):
            return False
    return True


class SentenceStore:
    """Cümle deposu + ters-indeks. Küçük ve saf: liste + dict."""

    def __init__(self):
        self.sentences = []                 # id -> (cümle, kaynak)
        self.index = {}                     # fold'lu sözcük -> set(id)

    def add(self, sentence, source=""):
        sid = len(self.sentences)
        self.sentences.append((sentence, source))
        for w in set(_words(sentence)):
            self.index.setdefault(w, set()).add(sid)
        return sid

    def find(self, query, most=4):
        """Sorguyla EN ÇOK içerik-sözcüğü kesişen cümleler. Önek toleranslı
        (çekim: "doluluğu"~"doluluk"). Skor = kesişen sözcük sayısı; eşitlikte
        kısa cümle önde (daha yoğun kanıt)."""
        qwords = set(_words(query))
        if not qwords:
            return []
        scores = {}
        for qw in qwords:
            hits = set()
            if qw in self.index:
                hits |= self.index[qw]
            else:
                # önek toleransı — iki yönlü, kök>=4 (indeks küçükken ucuz)
                for w, ids in self.index.items():
                    if len(qw) >= 4 and w.startswith(qw):
                        hits |= ids
                    elif len(w) >= 4 and qw.startswith(w):
                        hits |= ids
            for sid in hits:
                scores[sid] = scores.get(sid, 0) + 1
        if not scores:
            return []
        ranked = sorted(scores.items(),
                        key=lambda kv: (-kv[1], len(self.sentences[kv[0]][0])))
        # tek-sözcük kesişimli gürültüyü ele: en iyi skorun yarısından azı düşer
        best = ranked[0][1]
        keep = [sid for sid, sc in ranked if sc >= max(2, best // 2)][:most]
        return [self.sentences[sid][0] for sid in keep]

    # --- kalıcılık (JSON yan-dosya) ------------------------------------
    def save(self, memory_path):
        if not memory_path:
            return
        with open(memory_path + ".kanit", "w", encoding="utf-8") as f:
            json.dump([[s, src] for s, src in self.sentences], f,
                      ensure_ascii=False)

    @classmethod
    def load(cls, memory_path):
        store = cls()
        path = (memory_path + ".kanit") if memory_path else None
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for sentence, source in json.load(f):
                    store.add(sentence, source)
        return store
