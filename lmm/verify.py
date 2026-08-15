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


def verify(memory, answer, allowed, mode="STRICT"):
    """Cevabı cümle cümle denetle. Her olgu iddiasının özne+değeri allowed'da
    olmalı. Olgu taşımayan cümle (selam/görüş) geçer."""
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
            if sk not in allowed or vk not in allowed:
                grounded = False
                break
        if grounded:
            kept.append(sentence)
        elif mode == "ASSIST":
            kept.append(sentence + " ‹doğrulanmamış›")
        # STRICT: desteksiz cümle atılır
    return " ".join(kept).strip()
