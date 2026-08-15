"""ASIL KAPI — cevabın OLGU iddiaları, o soru için ENJEKTE EDİLEN olgulara
karşı denetlenir.

Belkemiği kural: "Qwen grafta olmayanı söyleyemez" → daha da sıkısı: "Qwen bu
soru için verilmeyeni söyleyemez". Cevaptaki her olgu iddiasının hem öznesi hem
değeri `allowed` (enjekte edilen kimlikler) içinde olmalı; değilse düşer.

Bu, "kalp → kan akışı" gibi Qwen'in parametrik sızıntısını yakalar: "kan akışı"
enjekte edilmedi, hiçbir kimliğe çözülmez → allowed'da yok → cümle düşer. Değer
ikamesini de yakalar (Paris yerine Berlin: Berlin allowed'da yok).

Eski hâlin iki deliği kapandı:
  F2  sohbette allowed=∅ → olgu iddiası taşıyan sohbet cevabı düşer (ham dönmez)
  F5  gevşek önek toleransı yok — TAM eşleşme (yanlış-pozitif = uydurma geçmesi
      tehlikeli; yanlış-negatif = "bilmiyorum" güvenli)

DÜRÜST SINIR (F3 artığı): iddia ÇIKARIMI hâlâ Qwen'e dayanıyor (reextract).
Qwen bir uydurmayı reextract'ta gizlerse kapı görmez. allowed-denetimi F3'ü
büyük ölçüde DARALTIR (karar allowed kümesinde, Qwen'de değil) ama tam
bağımsızlık Qwen-dışı bir varlık-çıkarıcı ister — sonraki iş.

Mod: STRICT (desteksiz cümle düşer) · ASSIST (‹doğrulanmamış› işaretlenir).
"""
import re

from lmm import extract, link
from v3.dataset import fold


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def allowed_of(memory, records):
    """Enjekte edilen kayıtların özne+değer kimlikleri — 'izinli' küme."""
    keys = set()
    for r in records:
        if isinstance(r.subject, int):
            keys.add(r.subject)
        if isinstance(r.value, int):
            keys.add(r.value)
    return keys


def edges_of(records):
    """Enjekte edilen kayıtların YÖNSÜZ kenar çiftleri (geriye-uyum için tutulur;
    verify artık grafın tamamına `_has_edge` ile bakıyor)."""
    pairs = set()
    for r in records:
        if isinstance(r.subject, int) and isinstance(r.value, int):
            pairs.add(frozenset((r.subject, r.value)))
    return pairs


def _has_edge(memory, sk, vk, v_label=None):
    """Grafta sk—vk arasında GERÇEK bir kenar var mı (YÖNSÜZ). Türetilmiş
    (#inference) kenarları da görür — `about` güven süzmez — böylece is-a gibi
    GEÇİŞLİ yüklemlerde türetilmiş "kartal→hayvan" geçer, ama türetilMEYEN
    "fransa→eyfel" (başkent geçişli değil) düşer.

    v_label verilirse (reextract'ın değer METNİ): anahtar eşleşmezse ÖZNENİN
    KENDİ değer etiketleriyle KELİME düzeyinde eşleşme dener (çekim/çok-kelime
    toleransı: "kuş" ~ "deniz kuşu"). GÜVENLİ: yalnız öznenin kendi değerlerine
    bakar — farklı bir düğüme ilişki UYDURAMAZ, yeniden-birleşim deliğini açmaz
    (web'den öğrenilen çok-kelimeli kavramların cevaplanabilmesi için)."""
    if sk is None:
        return False
    if vk is not None:
        if any(r.value == vk for r in memory.about(sk, touch=False)):
            return True
        if any(r.value == sk for r in memory.about(vk, touch=False)):
            return True
    if v_label:
        want = {fold(w) for w in v_label.split() if len(w) >= 3}
        if want:
            for r in memory.about(sk, touch=False):
                have = {fold(w) for w in link.label_of(memory, r.value).split()
                        if len(w) >= 3}
                for a in want:
                    for b in have:
                        if a == b or a.startswith(b) or b.startswith(a):
                            return True
    return False


def verify(memory, answer, allowed, mode="STRICT", anchor="edge", edges=None):
    """Cevabı cümle cümle denetle. Olgu taşımayan cümle (selam/görüş) geçer.

    `anchor`:
      "edge"  (VARSAYILAN, ASK yolu) — iddianın (özne,değer)'i enjekte edilen
              GERÇEK bir kenar olmalı (`edges`). Düğüm-üyeliği yetmez → yeniden
              birleşim uydurması bloklanır. `edges` verilmezse güvenli düşüş:
              eski düğüm-üyeliği (allowed) denetimi.
      "value" (sohbet-KİMLİK yolu) — nesne allowed'da olmalı; özne öz-referanslı
              zamir (ben/beni/I) çözülemediğinden anchor DEĞİL, AMA özne gerçek
              bir düğüme çözülüyorsa ya izinli olmalı ya da kenar bulunmalı —
              "Python'u Rüzgar yazdı" (python≠izinli, {python,rüzgar} kenarı yok)
              böylece düşer; "Beni Rüzgar yaptı" (ben→None) geçer."""
    edges = edges or set()
    kept = []
    for sentence in _sentences(answer):
        claims = extract.reextract(sentence)
        if not claims:
            kept.append(sentence)          # olgu iddiası yok → geçer
            continue
        grounded = True
        for subject, _predicate, value in claims:
            sk = link.resolve(memory, subject)
            vk = link.resolve(memory, value)
            if anchor == "value":
                ok = (vk in allowed) and (sk is None or sk in allowed
                                          or _has_edge(memory, sk, vk, value))
            else:                          # "edge": grafta gerçek kenar olmalı
                ok = _has_edge(memory, sk, vk, value)
            if not ok:
                grounded = False
                break
        if grounded:
            kept.append(sentence)
        elif mode == "ASSIST":
            kept.append(sentence + " ‹doğrulanmamış›")
        # STRICT: desteksiz cümle atılır
    return " ".join(kept).strip()
