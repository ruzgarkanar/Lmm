"""Agentic araştırma — LMM bilmediğini, KULLANICI ONAYIYLA internetten öğrenir.

Belkemiği kural burada da tutar: web = GÜVENİLMEZ kaynak. Çekilen bilgi grafa
"biliyorum" diye DEĞİL, `#web:<url>` damgası + DÜŞÜK güvenle girer; konuşulurken
kaynak-etiketiyle işaretlenir (session._hedge). Böylece condition-3 (retrain'siz
büyüme) sağlanır ve condition-4 (uydurma yok) bozulmaz — icat değil, kaynaklı.

Kaynak: Wikipedia REST özeti (anahtarsız, saygın, salt-okunur). Sorunun dili
bilinmediğinden birkaç dil denenir; ilk özet bulunan kullanılır.
"""
import json
import urllib.parse
import urllib.request

# Denenecek Wikipedia dilleri — dil algılamayı Wikipedia'nın kendi yönlendirmesi
# yapar; hangi dilde madde bulunursa o kullanılır (elle dil kuralı değil).
LANGS = ("tr", "en", "de", "fr", "es")


def wiki_summary(title, langs=LANGS, timeout=10):
    """Wikipedia'dan `title` için (özet_metin, sayfa_url) döndürür; bulunamazsa
    ("", ""). REST summary uç noktası ilk cümlede genelde 'X, bir Y'dir' verir —
    extract için ideal."""
    for lang in langs:
        url = (f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
               + urllib.parse.quote(title.strip().replace(" ", "_")))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LMM/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
        except Exception:                                   # noqa: BLE001
            continue
        if data.get("type", "").endswith("disambiguation"):
            continue
        extract = (data.get("extract") or "").strip()
        if extract:
            page = (data.get("content_urls", {}).get("desktop", {})
                    .get("page") or url)
            return extract, page
    return "", ""
