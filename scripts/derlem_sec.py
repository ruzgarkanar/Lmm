"""Ham web derleminden okumaya DEĞER cümleleri ayıklar.

`data/tr-web.txt` 31 GB ve hepsini okutmak hem pahalı hem gereksiz. Vikipedi
tanımları bir şeyin NE OLDUĞUNU söylüyordu; web metni NASIL olduğunu da
söylüyor ve bizim eksiğimiz tam orası:

    "Elma sirkesi güneş lekesi tedavisinde kullanılabilen bitkisel bir
     şifa kaynağıdır."

Ama aynı derlemde gezinme çubukları, ürün listeleri, haber spotları ve
yorum kırıntıları da var. Seçim, okuyucuya para harcanmadan önce yapılmalı.

Beş süzgeç, hepsi sayıya dayanıyor — hiçbiri kelime listesi değil:

    ŞEKİL       cümle uzunluğu, harf oranı, noktalama. Tablo ve liste eleniyor.
    ÖZNE        cümle bir KAVRAMLA başlamalı ve o kavram grafın tanıdığı ya da
                derlemde ad gibi davranan bir şey olmalı.
    YÜKLEM      koşaç ya da geniş zaman taşımalı — "X şudur", "X şunu yapar".
                Anlatı geçmiş zamanla konuşur ("gasp etti") ve olgu vermez.
    TEKRAR      aynı kalıbın yüzlerce kopyası (ürün sayfaları) elenir.
    DEĞER       grafın hakkında az şey bildiği kavramlar öne alınır — okuma
                bütçesi en çok orayı büyütür.

Kullanım:
    python3 scripts/derlem_sec.py [--veri data/tr-web.txt] [--tara 400000]
                                  [--sayi 20000] [--yaz secilen.txt]
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm import frequency                                   # noqa: E402
from lmm.intuition import tokenize                          # noqa: E402
from lmm.memory import Memory                               # noqa: E402
from lmm.turkish import TurkishMorphology                   # noqa: E402
from lmm.verbs import is_structural                         # noqa: E402

# Derlem satır başına bir BELGE tutuyor, cümle değil: ortanca satır 1.766
# karakter, en uzunu 618.679. İlk yazışta satırları cümle sandım ve 300.000
# satırdan SIFIR cümle seçildi — süzgeçler doğruydu, girdi yanlıştı.
SENTENCE = None

SHORTEST, LONGEST = 35, 170
LETTERS = 0.88                  # tablo ve kod bu oranı geçemez
MOST_WORDS = 14

# Cümlenin ÖZNEYLE başladığını gösteren şekil: büyük harfle başlayan bir ya da
# iki kelime, ardından cümlenin gerisi. Başlık değil özne arıyoruz, o yüzden
# nokta ya da virgül şart değil.
OPENING = re.compile(r"^([A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,20}(?: [a-zçğıöşü]{2,18}){0,2})\s")


def _predicative(word, morphology, verbs):
    """Bu kelime bir yüklem mi — koşaç ya da geniş zaman.

    Anlatı geçmiş zamanla konuşur ("gasp etti", "kuruldu") ve olgu vermez:
    bir olayı anlatır. Olgu şimdiki hâli bildirir — "X şudur", "X şunu yapar".
    Ayrım zamanda ve zaman ekten okunuyor.
    """
    if morphology.has_copula(word):
        return True
    return word in verbs


BREAKS = re.compile(r"(?<=[.!?])\s+")


def _sentences(document):
    """Belgeyi cümlelere böler — nokta, ünlem, soru işaretinden."""
    return [piece.strip() for piece in BREAKS.split(document) if piece.strip()]


def shaped(line):
    """Cümle şeklinde mi — tablo, liste, gezinme çubuğu değil mi."""
    if not SHORTEST <= len(line) <= LONGEST:
        return False
    letters = sum(1 for ch in line if ch.isalpha() or ch == " ")
    if letters / len(line) < LETTERS:
        return False
    if line.count(",") > 3 or "|" in line or "…" in line:
        return False
    # İki nokta bir BAŞLIK işareti: "Yüz Masajının Faydaları: Paket İçeriği:".
    # Ürün sayfaları ve haber spotları böyle yazılıyor ve olgu taşımıyorlar.
    if ":" in line or line.count("(") > 1:
        return False
    # Rakam taşıyan cümle çoğunlukla fiyat, tarih ya da teknik döküm; ölçüldü,
    # okuyucudan çıkan olguların en çöp olanları buradan geliyordu
    # ("ilaç --property--> akşam", "reçete --has--> yaş").
    if any(ch.isdigit() for ch in line):
        return False
    return len(line.split()) <= MOST_WORDS


def pick(path, scan, want, memory, morphology):
    """Okumaya değer cümleler, değerlisi önce."""
    counts = frequency.counts()
    verbs = frequency.verbs()
    known = set(memory.concepts())
    thin = {name for name in known
            if not any(edge.relation in ("property", "can", "has")
                       for edge in memory.query(name))}
    shapes = collections.Counter()       # aynı kalıbın kopyalarını elemek için
    rich, plain = [], []
    seen = 0
    for document in open(path, encoding="utf-8", errors="ignore"):
        seen += 1
        if seen > scan:
            break
        for line in _sentences(document):
            if not shaped(line):
                continue
            opening = OPENING.match(line)
            if not opening:
                continue
            subject = opening.group(1).lower()
            head = subject.split()[-1]
            if is_structural(head, counts) or head in verbs:
                continue
            words = tokenize(line)
            if not any(_predicative(word, morphology, verbs)
                       for word in words[1:]):
                continue
            # Aynı şekli yüzlerce kez almak derlemi değil şablonu öğrenmektir.
            shape = " ".join(words[:2] + words[-2:])
            shapes[shape] += 1
            if shapes[shape] > 3:
                continue
            if subject in thin or head in thin or subject in known \
                    or head in known:
                rich.append(line)       # grafın bildiği ya da az bildiği
            else:
                plain.append(line)
        if len(rich) >= want:
            break
    return rich[:want], plain[:max(0, want - len(rich))], seen


def main(argv):
    path = (argv[argv.index("--veri") + 1] if "--veri" in argv
            else "data/tr-web.txt")
    scan = int(argv[argv.index("--tara") + 1]) if "--tara" in argv else 400000
    want = int(argv[argv.index("--sayi") + 1]) if "--sayi" in argv else 20000
    out = argv[argv.index("--yaz") + 1] if "--yaz" in argv else None
    if not os.path.exists(path):
        print(f"yok: {path}")
        return 1

    memory = Memory.load("models/graph/birlesik.lmm")
    morphology = TurkishMorphology()
    rich, plain, seen = pick(path, scan, want, memory, morphology)
    print(f"{path} — {seen:,} satır tarandı")
    print(f"  bilinen kavram hakkında : {len(rich):,}")
    print(f"  yeni kavram hakkında    : {len(plain):,}")
    print(f"  seçim oranı             : %{(len(rich) + len(plain)) / max(seen, 1) * 100:.2f}\n")
    for line in rich[:6]:
        print(f"    · {line[:100]}")

    if out:
        with open(out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(rich + plain))
        print(f"\n  -> {out}  ({len(rich) + len(plain):,} cümle)")
    else:
        print("\n  (yazmak için --yaz ekle)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
