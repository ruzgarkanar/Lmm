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


def verify(memory, answer, allowed, mode="STRICT", anchor="both"):
    """Cevabı cümle cümle denetle. Olgu taşımayan cümle (selam/görüş) geçer.

    `anchor`:
      "both"  (VARSAYILAN, ASK yolu) — iddianın hem öznesi hem değeri allowed'da
              olmalı. Değer-ikamesini ve parametrik sızıntıyı yakalar.
      "value" (sohbet-KİMLİK yolu) — yalnız iddia edilen DEĞER (nesne) allowed'da
              olmalı. "Beni Rüzgar yaptı" → değer rüzgar∈allowed geçer; "Beni
              Google yaptı" → google∉allowed düşer. Özne çoğu dilde öz-referanslı
              zamir (ben/beni/I) olup güvenilir çözülemez; onu ANCHOR yapmak iyi
              kimlik cevabını yanlışlıkla düşürüyordu (refusal'a). Nesneyi anchor
              yapmak dış uydurmayı yine bloklar — kayıp yalnız güvenli yönde
              (fazla-tutucu = 'bilmiyorum'), asla uydurma sızmaz."""
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
            ok = (vk in allowed) if anchor == "value" else (
                sk in allowed and vk in allowed)
            if not ok:
                grounded = False
                break
        if grounded:
            kept.append(sentence)
        elif mode == "ASSIST":
            kept.append(sentence + " ‹doğrulanmamış›")
        # STRICT: desteksiz cümle atılır
    return " ".join(kept).strip()
