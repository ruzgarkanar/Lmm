"""Çıkarım: mesaj → {kind, triples}. Qwen çıkarır ama YAZAMAZ — çıkan üçlüler
KAPIYA aday olarak girer (session/verify karar verir). reader.py'nin yerini
alır; sözleşme aynı kalır (kind + üçlüler) ki session mantığı bozulmasın.
"""
import json

from lmm import prompts, runtime

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
            out.append((subject.lower(), predicate.lower(), value.lower()))
    return out


def extract(message):
    """Mesajı oku → {'kind': WRITE|ASK|CHAT, 'triples': [(s,p,v)]}.
    Deterministik (temperature 0): aynı cümle aynı çıkarım."""
    raw = runtime.generate(message, system=prompts.EXTRACT_SYSTEM,
                           max_tokens=160, temperature=0.0)
    data = _json(raw)
    kind = str(data.get("kind", CHAT)).upper()
    if kind not in (WRITE, ASK, CHAT):
        kind = CHAT
    return {"kind": kind, "triples": _clean(data.get("triples"))}


def reextract(sentence):
    """Doğrulama kapısı için: bir cümlenin taşıdığı OLGU iddialarını çıkarır.
    Olgu yoksa [] (selam, görüş, 'bilmiyorum')."""
    raw = runtime.generate(sentence, system=prompts.REEXTRACT_SYSTEM,
                           max_tokens=120, temperature=0.0)
    return _clean(_json(raw).get("triples"))
