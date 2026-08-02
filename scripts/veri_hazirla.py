"""Wikipedia dökümünden iki şey çıkarır: dil ve bilgi.

Aynı indirme iki organı birden besler ve bu bir tasarım tercihi:

    tüm metin        ->  dil çekirdeğinin eğitim derlemi (akıcılık)
    ilk cümleler     ->  LMM grafiğine olgu adayı (bilgi)

Wikipedia maddelerinin açılış cümlesi neredeyse her zaman bir tanımdır —
"Kartal, ... bir kuş türüdür" — yani zaten bizim kalıbımızın şeklinde. DBpedia
milyonlarca olguyu tam olarak bu düzenlilikten çıkardı. Buradaki iş de o:
tanımları ayırmak, gerisini dil çekirdeğine bırakmak.

Döküm sıkıştırılmış halde akıtılarak okunur; hiçbir aşamada tamamı belleğe
alınmaz, çünkü açılmış hali birkaç GB.

Kullanım: python3 scripts/veri_hazirla.py [--sinir N]
"""
import bz2
import html
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP = os.path.join(ROOT, "data", "trwiki-latest-pages-articles.xml.bz2")
TEXT = os.path.join(ROOT, "data", "tr-metin.txt")
DEFINITIONS = os.path.join(ROOT, "data", "tr-tanimlar.txt")

# Wikipedia biçimlendirmesi. Sırayla uygulanır; sıra önemli.
MARKUP = [
    (re.compile(r"<ref[^>]*/>"), " "),
    (re.compile(r"<ref[^>]*>.*?</ref>", re.S), " "),
    (re.compile(r"<!--.*?-->", re.S), " "),
    # Tablolar. Eğitilmiş çekirdek "{| class=\"wikitable\" width=200" üretti:
    # biçimlendirme çok öngörülebilir olduğu için model onu kolay lokma olarak
    # öğreniyor. Derlemin yalnızca %0,5'i ama çıktıda payından fazla görünüyor.
    (re.compile(r"\{\|.*?\|\}", re.S), " "),
    (re.compile(r"^\{\|.*$", re.M), " "),
    (re.compile(r"\b(?:Dosya|Resim|File|Image):[^\s|\]]*"), " "),
    (re.compile(r"<[^>]+>"), " "),
    (re.compile(r"\{\{[^{}]*\}\}"), " "),          # şablonlar, içten dışa
    (re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]"), r"\1"),   # [[hedef|görünen]]
    (re.compile(r"\[\[([^\]]*)\]\]"), r"\1"),
    (re.compile(r"\[https?://\S+\s([^\]]*)\]"), r"\1"),
    (re.compile(r"https?://\S+"), " "),
    (re.compile(r"'{2,}"), ""),                    # ''vurgu'' ve '''kalın'''
    (re.compile(r"^[*#:;=|!].*$", re.M), " "),     # liste, başlık, tablo satırı
    (re.compile(r"[ \t]+"), " "),
]

SKIP_TITLE = re.compile(r"^(Kategori|Şablon|Dosya|Vikipedi|Yardım|MediaWiki|"
                        r"Modül|Portal|Tartışma):")
# "Kartal, ... bir kuştur." — tanım cümlesinin imzası.
DEFINITION = re.compile(r"^(.{2,40}?),\s+(.{10,240}?\b(?:bir|birer)\s+.{2,60}?"
                        r"(?:dir|dır|dur|dür|tir|tır|tur|tür))\b")


# Ölçüldü: derlemin %10,2'si "Kategori:" satırı, %2,3'ü bilgi kutusu
# parametresi, %5'i boru işareti taşıyor. Eğitim sinyalinin sekizde biri
# Türkçe değil biçimlendirmeydi ve çekirdek onu öğrenip çıktıda kusuyordu.
#
# Bunları düzenli ifadeyle satır satır elemek denendi ve süreç 20 dakikada
# yarısını bile bitiremeden öldü: `^.*\|.*$` gibi bir kalıp, re.M ile uzun
# metinde katlanarak pahalılaşıyor. Aynı iş düz Python'da kelime araması ile
# yapılıyor — okunması da daha kolay.
DROP_LINE = ("|", "wikitable", "class=", "style=", "thumb", "px|")
DROP_PREFIX = ("Kategori:", "Category:", "!", "*", "#", ";", ":", "=")


def keep(line):
    """Bu satır Türkçe nesir mi, yoksa biçimlendirme mi?"""
    stripped = line.strip()
    if not stripped or stripped.startswith(DROP_PREFIX):
        return False
    if any(mark in stripped for mark in DROP_LINE):
        return False
    # "ad = değer" biçimi: eşittir var ama cümle noktalaması yok.
    if "=" in stripped and not any(end in stripped for end in ".!?"):
        return False
    return True


def clean(text):
    # Kaçırılmış işaretler önce çözülür. Tersi yapılınca "&lt;ref&gt;" önce
    # varlık kuralına takılıyor ve geriye çıplak "ref /ref" kelimeleri kalıyordu
    # — metnin içinde, cümlenin ortasında, dil çekirdeğinin öğreneceği bir
    # düzenlilik olarak. Sıra önemliydi.
    text = html.unescape(text)
    for pattern, replacement in MARKUP:
        text = pattern.sub(replacement, text)
    for _ in range(3):                  # iç içe şablonlar için birkaç geçiş
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    return "\n".join(line.strip() for line in text.split("\n") if keep(line))


def first_sentence(text):
    """Maddenin ilk düzgün cümlesi — tanım oradadır."""
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 30 or line.startswith(("|", "!", "{", "}")):
            continue
        end = line.find(". ")
        sentence = line if end < 0 else line[:end + 1]
        if 30 <= len(sentence) <= 300:
            return sentence
    return None


def articles(path):
    """Dökümü akıtarak (başlık, metin) üretir. Bellekte tek madde durur."""
    title, buffer, inside = None, [], False
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if "<title>" in line:
                title = re.sub(r".*<title>(.*?)</title>.*", r"\1", line.strip())
            elif "<text" in line:
                inside = True
                buffer = [line.split(">", 1)[-1]]
            elif inside:
                if "</text>" in line:
                    buffer.append(line.split("</text>")[0])
                    inside = False
                    if title and not SKIP_TITLE.match(title):
                        yield title, "".join(buffer)
                    buffer = []
                else:
                    buffer.append(line)


def main(argv):
    limit = 0
    if "--sinir" in argv:
        limit = int(argv[argv.index("--sinir") + 1])
    if not os.path.exists(DUMP):
        print(f"döküm bulunamadı: {DUMP}")
        return 1

    started = time.time()
    words = kept = definitions = 0
    with open(TEXT, "w", encoding="utf-8") as text_out, \
            open(DEFINITIONS, "w", encoding="utf-8") as definition_out:
        for index, (title, raw) in enumerate(articles(DUMP), 1):
            body = clean(raw)
            if len(body) < 200:
                continue                # taslak ve yönlendirmeler
            kept += 1
            words += body.count(" ") + 1
            text_out.write(body + "\n")
            opening = first_sentence(body)
            if opening and DEFINITION.match(opening):
                definition_out.write(opening + "\n")
                definitions += 1
            if kept % 2000 == 0:
                elapsed = time.time() - started
                print(f"  {kept:>7} madde | {words/1e6:>6.1f}M kelime | "
                      f"{definitions:>6} tanım | {elapsed:>5.0f} sn | "
                      f"{words/max(elapsed,1)/1000:>5.0f}K kelime/sn", flush=True)
            if limit and kept >= limit:
                break

    print(f"\nBİTTİ  {kept} madde, {words/1e6:.1f}M kelime, "
          f"{definitions} tanım cümlesi, {time.time()-started:.0f} sn")
    for path in (TEXT, DEFINITIONS):
        print(f"  {path}  {os.path.getsize(path)/2**20:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
