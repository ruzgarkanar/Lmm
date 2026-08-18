"""Agentic research — LMM learns what it doesn't know from the internet, WITH
USER APPROVAL.

The backbone rule holds here too: the web = an UNTRUSTED source. Fetched
information enters the graph NOT as "I know this" but with a `#web:<url>` stamp
+ LOW trust; when spoken it is marked with the source label (session._hedge).
Thus condition-3 (growth without retraining) is met and condition-4 (no
fabrication) is not broken — not invention, but sourced.

Source: the Wikipedia REST summary (keyless, reputable, read-only). Since the
question's language is unknown, several languages are tried; the first summary
found is used.
"""
import json
import urllib.parse
import urllib.request

# Wikipedia languages to try — language detection is done by Wikipedia's own
# redirects; whichever language has the article is used (not a hand-written
# language rule).
LANGS = ("tr", "en", "de", "fr", "es")


def wiki_summary(title, langs=LANGS, timeout=10):
    """Returns (summary_text, page_url) for `title` from Wikipedia; if not
    found, ("", ""). The REST summary endpoint usually gives 'X is a Y' in the
    first sentence — ideal for extract."""
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
