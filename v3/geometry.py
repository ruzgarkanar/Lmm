"""Sürekli katman: BULMAK için geometri, doğrulamak için değil.

Projenin çekirdek fikri burada yarısını buluyor:

    Bir dil modelinde her şey süreklidir — bulunur ama doğrulanamaz.
    Klasik bir bilgi grafında her şey ayrıktır — doğrulanır ama bulunamaz.
    Burada ikisi aynı kayıtların iki katmanıdır: SÜREKLİ BULUR, AYRIK DOĞRULAR.

Ölçüldü ve eksik olan tam buydu: eski bellekte gerçek soruların %19'unda
"kavram grafta yok" deniyordu ve kavram GRAFTAYDI — dizgi tutmuyordu. Erişim
harf harf eşleşmeye dayandığı sürece, bilinen bir şey bilinmiyor görünür.

Burada erişim iki adımdır:

    1. YAKINLIK   sorunun vektörüne en yakın kimlikler (sürekli katman)
    2. YAYILIM    o kimliklerden başlayıp BAĞLAR üzerinden yürümek

İkincisi çağrışımdır: bir örüntünün parçası verilince bütünü geri gelir
(Hopfield sezgisi). Dizgi araması bunu yapamaz; bir kayıt ancak adı tam
bilinirse bulunur. Yayılım, adı bilinmeyeni komşusundan getirir.

Vektörler bu dosyada ÜRETİLMEZ — sayımla kurulur ve dışarıdan verilir. Burada
yalnız kullanılırlar. Dile ait hiçbir şey yoktur: ne kelime, ne ek, ne kural.
Kosinüs ve graf yürüyüşü, o kadar.

SINIR — MUTLAK. Geometri hiçbir zaman bir kayıt YAZMAZ. Zıtlar aynı çevrede
geçer ve dağılımsal benzerlik onları ayıramaz; bu, bu projede bir kez ölçüldü.
Geometri aday bulur, kapı karar verir.
"""
import math

# Yakınlığın anlamlı sayılması için en az bu kadar olmalı. Eşiğin altı gürültü:
# yüksek boyutlu bir uzayda her şey her şeye biraz benzer.
NEAR = 0.35

# Yayılımda her adımda benzerliğin ne kadarı sönümlenir. Bire yakın olursa
# uzak kayıtlar yakınlar kadar bağırır ve çağrışım anlamını kaybeder.
DECAY = 0.6

# Yayılımın kaç adım süreceği. Üçten sonrası pratikte bütün grafa yayılıyor.
STEPS = 3


def cosine(one, other):
    """İki vektör arasındaki yön benzerliği."""
    if not one or not other:
        return 0.0
    total = sum(a * b for a, b in zip(one, other))
    left = math.sqrt(sum(a * a for a in one))
    right = math.sqrt(sum(b * b for b in other))
    if not left or not right:
        return 0.0
    return total / (left * right)


def mean(vectors):
    """Vektörlerin ortalaması — bir cümlenin kaba temsili.

    Sözcük sırasını kaybeder ve bunu biliyoruz. Yerini alacak şey okuyucu
    ağıdır; burası yalnız ilk elemeyi yapan ucuz bir el.
    """
    held = [one for one in vectors if one]
    if not held:
        return None
    width = len(held[0])
    return [sum(one[at] for one in held) / len(held) for at in range(width)]


def nearest(memory, vector, count=8, least=NEAR):
    """Vektöre en yakın kimlikler — [(kimlik, yakınlık)], yakından uzağa."""
    if vector is None:
        return []
    scored = []
    for key, held in memory.identities.items():
        if held.vector is None:
            continue
        score = cosine(vector, held.vector)
        if score >= least:
            scored.append((score, key))
    scored.sort(reverse=True)
    return [(key, score) for score, key in scored[:count]]


def resolve(memory, label, vector=None):
    """Bir etiketin HANGİ kimliği olduğunu bağlamdan seçer.

    Aynı yazılış birden çok kavram olabilir ve hangisi olduğunu bellek
    bilmez — bilmesi de gerekmez. Karar burada, bağlamın vektörüne en yakın
    kimliği seçerek verilir.

    Bağlam yoksa ya da hiçbiri yeterince yakın değilse en çok erişilmiş
    kimlik döner: yokluğunda en tanıdık olanı seçmek, hiç seçmemekten iyidir
    ve seçimin zayıf olduğu güvene yansır.
    """
    held = memory.candidates(label)
    if not held:
        return None, 0.0
    if len(held) == 1:
        return held[0], 1.0
    if vector is not None:
        scored = []
        for key in held:
            other = memory.identities[key].vector
            if other is not None:
                scored.append((cosine(vector, other), key))
        if scored:
            scored.sort(reverse=True)
            score, key = scored[0]
            if score >= NEAR:
                return key, score
    best = max(held, key=lambda key: memory.identities[key].seen)
    return best, 0.0


def spread(memory, seeds, steps=STEPS, decay=DECAY, most=40):
    """Çağrışım: tohum kimliklerden başlayıp bağlar üzerinden yürür.

    Dönen: {kayıt anahtarı: ağırlık}. Ağırlık, tohuma olan uzaklıkla söner —
    yakından gelen kayıt daha yüksek sesle konuşur.

    Yürüyüş iki tür bağı da kullanır: bir kimliğin kayıtları (özne bağı) ve
    kayıtların birbirine bağları (neden, sonra, koşul). İkincisi olmadan bu
    yalnız komşu listelemek olurdu; onunla birlikte bir zincir izlemek oluyor.
    """
    reached = {}
    frontier = {key: weight for key, weight in seeds}
    for step in range(steps):
        following = {}
        for key, weight in frontier.items():
            for record in memory.about(key):
                held = reached.get(record.key, 0.0)
                if weight > held:
                    reached[record.key] = weight
                # Kaydın değeri bir kimlikse oradan devam et.
                if isinstance(record.value, int) \
                        and record.value in memory.identities:
                    onward = weight * decay
                    if onward > following.get(record.value, 0.0):
                        following[record.value] = onward
                # Kayıttan kayda bağlar — asıl çağrışım burada.
                for _, other in record.links:
                    onward = weight * decay
                    if onward > reached.get(other, 0.0):
                        reached[other] = onward
        if not following:
            break
        frontier = following
    return dict(sorted(reached.items(), key=lambda pair: -pair[1])[:most])


def recall(memory, vector=None, labels=(), most=40):
    """Bir soru için ilgili kayıtlar — geometri ile bulur, yayılım ile toplar.

    Erişimin tek kapısı: önce en yakın kimlikler, sonra onlardan yayılım.
    Hiçbir yerde dizgi eşleşmesi zorunlu değil; etiket verilirse yalnız
    tohumu güçlendirir.
    """
    seeds = []
    for label in labels:
        key, score = resolve(memory, label, vector)
        if key is not None:
            seeds.append((key, max(score, 0.5)))
    if vector is not None:
        seeds.extend(nearest(memory, vector, count=6))
    if not seeds:
        return {}
    return spread(memory, seeds, most=most)
