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


def _tokens(text):
    """Sıralı token dizisi (fold'lu, kısa dahil) — komşuluk denetimi için."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [fold(w) for w in _WORD.findall(text)]


def digits_ok(answer, block):
    """RAKAM disiplini tek başına: cevaptaki her rakam blokta VE en az bir
    aynı-komşulu ikilide. Sözcük-kapsaması başarısız olsa bile bu şart
    pazarlıksız — motor-destek denetimi (generate.supported) bile rakam
    ihlalini KURTARAMAZ (sayı değişimi = uydurma, nokta)."""
    bt = _tokens(block)
    bigrams = set(zip(bt, bt[1:]))
    block_digits = {t for t in bt if t.isdigit()}
    at = _tokens(answer)
    for i, t in enumerate(at):
        if not t.isdigit():
            continue
        if t not in block_digits:
            return False
        prev = at[i - 1] if i > 0 else None
        nxt = at[i + 1] if i + 1 < len(at) else None
        if prev is None and nxt is None:
            continue
        if (prev, t) not in bigrams and (t, nxt) not in bigrams:
            return False
    return True


def digits_present(answer, block):
    """Gevşek rakam şartı (ikinci kademe için): cevaptaki her rakam blokta VAR
    olmalı — komşuluk aranmaz. Meşru KOMPOZİSYON ("07 yüksek önceliklidir" iki
    ayrı satırın birleşimi) rakamı yeni komşulara taşır; bigram şartı onu
    öldürüyordu. Komşuluğun yerini motor-destek denetimi (supported) alır."""
    bt = _tokens(block)
    block_digits = {t for t in bt if t.isdigit()}
    return all(t in block_digits for t in _tokens(answer) if t.isdigit())


def covered(answer, block, question=""):
    """Cevabın TÜM içerik-sözcükleri verilen bloktan mı geliyor — önek
    toleranslı (çekim: "bandındadır"~"bandında"). Yeni içerik-sözcük yok =
    yeni iddia yok. RAKAM kuralları SIKI (code-review #1):
      - rakam yalnız BLOK'tan sayılır (sorudaki rakam sayılmaz — "tier 1 mi?"
        yanlış-öncül yankısı kapıdan geçmesin);
      - cevaptaki her rakam, blokta EN AZ BİR aynı-komşulu ikilide geçmeli
        ((önceki,rakam) ya da (rakam,sonraki)) — bloktaki başka satırın
        rakamının başka özneye yapışması (Tier-karışması) zorlaşır.
    Bilinen kalıntı: önek toleransı olumsuzluk ekini ayırt edemez
    ("azalma"~"azalmaz") — dil-listesi yasak olduğundan sözcük-küme düzeyinde
    kapatılamaz; motor-düzeyi denetim v-sonraki."""
    if not digits_ok(answer, block):
        return False
    given = set(_words(block + " " + question))
    words = set(_words(answer))
    if not words:
        return False
    for w in words:
        if w in given or w.isdigit():
            continue
        if not any((len(g) >= 5 and w.startswith(g) and len(w) - len(g) <= 3)
                   or (len(w) >= 5 and g.startswith(w) and len(g) - len(w) <= 3)
                   for g in given):
            return False
    return True


class SentenceStore:
    """Cümle deposu + ters-indeks. Küçük ve saf: liste + dict."""

    def __init__(self):
        self.sentences = []                 # id -> (cümle, kaynak)
        self.index = {}                     # fold'lu sözcük -> set(id)
        self._seen = set()                  # fold'lu cümle — çift kanıt engeli
        self.last_sources = []              # son find()'ın kaynakları (hedge)

    def add(self, sentence, source=""):
        key = fold(sentence)
        if key in self._seen:               # aynı doküman iki kez okunursa
            return None                     # kanıt katlanmasın (review #2)
        self._seen.add(key)
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
        # eşitlikte UZUN kazanır: kısa tablo kırıntısı ("Orta · 3") bağlam
        # taşıyan düzyazı penceresini geçmesin — hücre değeri bağlamsız gezince
        # yanlış niteliğe yapışıyordu (kamera/KVKK vakası).
        ranked = sorted(scores.items(),
                        key=lambda kv: (-kv[1], -len(self.sentences[kv[0]][0])))
        # gürültü eşiği: en iyi skorun yarısından azı düşer. KISA sorgu istisnası
        # (review #4): 1-2 içerik-sözcüklü soruda ("karvel nedir") skor 1 meşru —
        # taban 2 olsaydı kanıt varken boş dönerdi. Pencere/tablo satırları best'i
        # şişirebilir; taban, sorgu kısaldıkça iner.
        best = ranked[0][1]
        floor = max(1 if len(qwords) <= 2 else 2, best // 2)
        keep = [sid for sid, sc in ranked if sc >= floor][:most]
        self.last_sources = [self.sentences[sid][1] for sid in keep]
        return [self.sentences[sid][0] for sid in keep]

    # --- kalıcılık (JSON yan-dosya) ------------------------------------
    def save(self, memory_path):
        if not memory_path:
            return
        # ATOMİK (review #2): tmp + os.replace — yarım .kanit dosyası load'u
        # (dolayısıyla Session.__init__'i) çökertmesin.
        path = memory_path + ".kanit"
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([[s, src] for s, src in self.sentences], f,
                      ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, memory_path):
        store = cls()
        path = (memory_path + ".kanit") if memory_path else None
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    rows = json.load(f)
            except (json.JSONDecodeError, OSError):
                return store        # bozuk yan-dosya → boş depo (graf sağlam;
                #                     kanıt yeniden yutulabilir, uydurma riski yok)
            for sentence, source in rows:
                store.add(sentence, source)
        return store
