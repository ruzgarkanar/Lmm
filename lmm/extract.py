"""Çıkarım: mesaj → {kind, triples}. Qwen çıkarır ama YAZAMAZ — çıkan üçlüler
KAPIYA aday olarak girer (session/verify karar verir). reader.py'nin yerini
alır; sözleşme aynı kalır (kind + üçlüler) ki session mantığı bozulmasın.
"""
import json
import unicodedata

from v3.dataset import fold
from lmm import prompts, runtime


def _fold(text):
    """fold + görünmez-birleşen-im temizliği. Qwen JSON'unda 'i̇çecek' gibi
    (i + U+0307) diziler ÜRETEBİLİYOR — görünmez nokta ayrı graf düğümü açar,
    zincirler sessizce kopar. Önce NFC (gerçek aksanlar tek koda birleşir:
    ç/ö/ü/é korunur), sonra ARTAKALAN birleşen imler atılır (birleşemeyenler
    bu tür çöptür). Dil kuralı değil — Unicode'un kendi tablosu."""
    text = unicodedata.normalize("NFC", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return fold(text)

WRITE, ASK, CHAT = "WRITE", "ASK", "CHAT"


def _json(raw):
    """Qwen çıktısından İLK TAM JSON nesnesini çeker — dengeli parantez tarama.

    Eski greedy `\\{.*\\}` en baştaki { ile en sondaki } arasını yutuyordu:
    Qwen olgudan sonra açıklama + ikinci bir {...} verirse (küçük modellerde
    sık) iki nesne birleşip geçersiz JSON oluyor, çıkarım sessizce sıfırlanıyordu.
    Şimdi ilk açılan parantezin dengelendiği yerde durur.
    """
    start = raw.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except ValueError:
                    return {}
    return {}


def _clean(triples):
    """Üçlüleri (özne, yüklem, değer) tekilliğe getirir; öznesizi atar.

    Qwen üçlüyü hem liste [[s,p,v]] hem sözlük [{"subject":..}] biçiminde
    verebilir (chat modellerde sözlük olası) — ikisi de kabul edilir, yoksa
    öğretilen olgu sessizce düşerdi.
    """
    out = []
    for t in triples or []:
        if isinstance(t, dict):
            subject = t.get("subject") or t.get("özne") or ""
            predicate = (t.get("predicate") or t.get("relation")
                         or t.get("yüklem") or t.get("ilişki") or "")
            value = t.get("value") or t.get("object") or t.get("değer") or ""
            parts = [str(subject).strip(), str(predicate).strip(),
                     str(value).strip()]
        elif isinstance(t, (list, tuple)):
            parts = [str(x).strip() for x in (list(t) + ["", "", ""])[:3]]
        else:
            continue
        subject, predicate, value = parts
        if subject:
            # _fold, .lower() DEĞİL: 'İçecek'.lower() → 'i̇çecek' (görünmez
            # U+0307) ayrı düğüm açıp zincirleri koparıyordu; _fold hem bunu
            # hem Qwen'in hazır ürettiği birleşen imleri temizler.
            out.append((_fold(subject), _fold(predicate), _fold(value)))
    return out


def _malformed(subject, value):
    """Özne cümleyi YUTMUŞ mu — "aslan bir memelidir" öznesi 'aslan bir
    memelidir' olursa düğüm etiketi cümle olur ve "aslan nedir" onu asla
    bulamaz (sessiz kayıp). Yapısal ölçüt: değerin sözcükleri öznenin İÇİNDE.
    Dil kuralı değil — küme bakışı."""
    if not (subject and value):
        return False
    sw = set(subject.split())
    vw = set(value.split())
    return bool(vw) and vw <= sw and len(sw) > len(vw)


def extract(message):
    """Mesajı oku → {'kind': WRITE|ASK|CHAT, 'triples': [(s,p,v)]}.
    Deterministik (temperature 0): aynı cümle aynı çıkarım.

    EMNİYET: özne-cümleyi-yutmuş üçlü önce reextract'la (farklı istem, daha
    sağlam) onarılmaya çalışılır; o da veremezse üçlü DÜŞER — çöp düğüm açıp
    "öğrendim" demektense dürüstçe öğrenememek (uydurma-0'ın yazma yüzü)."""
    raw = runtime.generate(message, system=prompts.EXTRACT_SYSTEM,
                           max_tokens=160, temperature=0.0)
    data = _json(raw)
    kind = str(data.get("kind", CHAT)).upper()
    if kind not in (WRITE, ASK, CHAT):
        kind = CHAT
    triples = _clean(data.get("triples"))
    if kind == WRITE and any(_malformed(s, v) for s, _p, v in triples):
        repaired = reextract(message)
        if repaired:
            triples = repaired
        else:
            triples = [t for t in triples if not _malformed(t[0], t[2])]
    return {"kind": kind, "triples": triples}


def reextract(sentence):
    """Doğrulama kapısı için: bir cümlenin taşıdığı OLGU iddialarını çıkarır.
    Olgu yoksa [] (selam, görüş, 'bilmiyorum')."""
    raw = runtime.generate(sentence, system=prompts.REEXTRACT_SYSTEM,
                           max_tokens=120, temperature=0.0)
    return _clean(_json(raw).get("triples"))
