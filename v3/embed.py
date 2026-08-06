"""Gömme kurucu: kelimelere geometri — sayarak, gradyansız, GPU'suz.

Geometri katmanı (`v3/geometry.py`) vektör bekler ve bu dosya olmadan kördü:
`Identity.vector` alanı vardı, dolduranı yoktu. Doldurulmadığında `resolve`
"en çok erişilen"e düşüyor — yani dizgi eşleşmesi + popülerlik. Çokanlamlılığı
çözmek için kurulan katman, dolduranı olmadığı için hiç çalışmamış olurdu.
Denetim bunu satır satır kıyasta yakaladı.

Yöntem sayımdır ve dayanağı ölçülüdür: word2vec'in kaydırılmış bir PMI
matrisini örtük çarpanlara ayırdığı gösterildi (Levy & Goldberg 2014) —
gradyan, anlamın geldiği yer değil, ona varmanın bir yoluydu. Burada aynı
geometriye doğrudan gidilir:

    1. SAYIM     hangi kelime hangi kelimeyle geçiyor (pencere)
    2. PPMI      "herkesin arkadaşı" kelimelerin şişirmesi bölünür
    3. İZDÜŞÜM   sabit tohumlu rastgele izdüşümle boyut indirilir —
                 deterministik: aynı derlem her zaman aynı vektörler

Dile ait hiçbir şey yoktur: kelime, boşlukla ayrılan şeydir ve hangi dilde
olduğu sorulmaz. Aynı kurucu yarın İngilizce derlemle İngilizce geometri
kurar.

SINIR — geometri yalnız BULUR. Vektör hiçbir zaman kayıt yazmaz; zıtlar aynı
çevrede geçer ve dağılım onları ayıramaz (bu projede ölçüldü). Karar kapınındır.

Kullanım:
    python3 -m v3.embed data/raw/tr-metin.txt --out models/v3/vectors.json
"""
import json
import math
import os
import sys

# Pencere: bir kelimenin "çevresi" sayılan komşuluk. Dördün ötesi, cümle
# sınırını aşıp ilgisiz kelimeleri çevre saymaya başlıyor.
WINDOW = 4

# Bir kelimenin sayılması için en az kaç kez görülmesi gerekir. Bir kez
# görülenin dağılımı yoktur; vektörü gürültü olur.
LEAST = 5

DIMENSIONS = 128


def sentences(paths, most=None):
    """Derlem satırları — kelimelere bölünmüş. Bölme yalnız boşluktan."""
    count = 0
    for path in paths:
        opener = _open(path)
        for line in opener:
            words = []
            for one in line.split():
                start, stop = 0, len(one)
                while start < stop and not one[start].isalnum():
                    start += 1
                while stop > start and not one[stop - 1].isalnum():
                    stop -= 1
                piece = one[start:stop]
                if piece:
                    words.append("".join(ch.casefold()[0] for ch in piece))
            if len(words) < 2:
                continue
            yield words
            count += 1
            if most and count >= most:
                return


def _open(path):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def counts(rows):
    """Kelime ve pairs sayımları. Tek geçişte, bellek dostu."""
    seen, pairs = {}, {}
    for words in rows:
        for at, word in enumerate(words):
            seen[word] = seen.get(word, 0) + 1
        for at, word in enumerate(words):
            low = max(0, at - WINDOW)
            high = min(len(words), at + WINDOW + 1)
            for other in range(low, high):
                if other == at:
                    continue
                pair = (word, words[other])
                pairs[pair] = pairs.get(pair, 0) + 1
    kept = {word for word, count in seen.items() if count >= LEAST}
    pairs = {pair: count for pair, count in pairs.items()
             if pair[0] in kept and pair[1] in kept and count > 1}
    return kept, seen, pairs


def vectors_of(kept, pairs, dimensions=DIMENSIONS, rounds=10):
    """PPMI matrisinin baş yönleri — altuzay yinelemesiyle.

    İlk yazış seyrek rastgele izdüşümdü ve ölçüm çürüttü: "kuş" kelimesinin
    komşuları "karıştırıp, saklamak, luhansk" çıktı — gürültü. Kelime başına
    ~48 bağlam var; o seyreklikte izdüşüm gürültüsü sinyali yutuyor.

    Altuzay yinelemesi aynı geometriye emin yoldan gider: M·V çarpımı tekrar
    tekrar uygulanır ve V, matrisin en güçlü yönlerine yakınsar — word2vec'in
    örtük yaptığı çarpanlara ayırmanın açık hâli (Levy & Goldberg 2014).
    Başlangıç SABİT tohumlu: aynı derlem, her zaman aynı vektörler.

    torch yalnız hız için (seyrek çarpım + QR); eğitim tarafının aracı zaten.
    Çalışma anındaki geometri (`v3/geometry.py`) torch'suz kalır.
    """
    import torch

    words = sorted(kept)
    index = {word: at for at, word in enumerate(words)}
    total = sum(pairs.values()) or 1
    word_total = {}
    for (one, _), count in pairs.items():
        word_total[one] = word_total.get(one, 0) + count

    rows, cols, vals = [], [], []
    for (one, other), count in pairs.items():
        joint = count / total
        left = word_total.get(one, 1) / total
        right = word_total.get(other, 1) / total
        pmi = math.log(joint / (left * right))
        if pmi > 0:
            rows.append(index[one])
            cols.append(index[other])
            vals.append(pmi)
    matrix = torch.sparse_coo_tensor(
        torch.tensor([rows, cols]), torch.tensor(vals, dtype=torch.float32),
        (len(words), len(words))).coalesce()

    generator = torch.Generator().manual_seed(
        int.from_bytes(__import__("hashlib").md5(
            " ".join(words[:100]).encode()).digest()[:4], "big"))
    basis = torch.randn(len(words), dimensions, generator=generator)
    for _ in range(rounds):
        basis = torch.sparse.mm(matrix, basis)
        basis, _ = torch.linalg.qr(basis)
    projected = torch.sparse.mm(matrix, basis)
    projected = torch.nn.functional.normalize(projected, dim=1)

    return {word: [round(float(one), 5) for one in projected[at]]
            for word, at in index.items()}


def attach(memory, table):
    """Vektörleri kimliklere bağlar: kimliğin vektörü, etiketlerinin ortalaması."""
    bound = 0
    for identity in memory.identities.values():
        vectors = [table[label] for label in identity.labels
                   if label in table]
        if not vectors:
            continue
        width = len(vectors[0])
        identity.vector = [sum(one[at] for one in vectors) / len(vectors)
                           for at in range(width)]
        bound += 1
    return bound


def main(argv):
    paths = [one for one in argv[1:] if not one.startswith("--")]
    out = (argv[argv.index("--out") + 1] if "--out" in argv
           else "models/v3/vectors.json")
    most = int(argv[argv.index("--most") + 1]) if "--most" in argv else 300000
    if not paths:
        print(__doc__.strip().splitlines()[-1])
        return 1
    kept, seen, pairs = counts(sentences(paths, most))
    print(f"  {len(kept):,} words · {len(pairs):,} pairs")
    held = vectors_of(kept, pairs)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(held, handle, ensure_ascii=False)
    print(f"  -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
