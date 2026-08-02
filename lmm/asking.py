"""Bir cümlenin soru olduğunu nereden anlarız — çekimiyle birlikte.

Soru eki Türkçe'de çıplak durmaz. İnsanlar "uçar mı" kadar sık "bahseder
misin", "anlatır mısınız", "gider miyim" diyor: ek kişi ve zaman eki alıyor.
Ayrıştırıcı bunları hiç tanımıyordu ve bedeli ölçüldü — "bana penguenlerden
bahseder misin" cümlesi dört kavram olarak etiketleniyordu, yani soru olduğu
tamamen görünmezdi.

Çekimli hâller elle yazılmadı, **derlemden keşfedildi**
(`scripts/soru_kesfet.py`): bilinen bir ekle başlayacak ve çekimli fiilden
sonra gelmeye meyilli olacak. Tek başına hiçbiri yetmiyor — önek "milyon"u da
yakalıyor, fiil-sonrası dağılım "çünkü"yü de. İkisi birlikte temiz ayırıyor.

Tablo yoksa çıplak ek ve koşaçlı hâli yine tanınır: keşif bir kolaylık, bir
bağımlılık değil.
"""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(HERE, "data", "tr-soru.txt")

_loaded = None


def load(path=TABLE):
    """{çekimli hâl} — keşfedilmiş soru eki paradigması."""
    found = set()
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            word = line.split("\t", 1)[0].strip()
            if word:
                found.add(word)
    return found


def inflected(path=TABLE):
    """Bir kez yüklenir, sonra paylaşılır."""
    global _loaded
    if _loaded is None:
        _loaded = load(path)
    return _loaded


def particle_of(token, morphology):
    """Bu kelime bir soru eki mi — öyleyse çıplak hâli, değilse None.

    Üç biçim tanınır ve üçü de aynı şeyi sorar:

        çıplak      "uçar mı"
        koşaçlı     "mutlu mudur"
        çekimli     "bahseder misin"
    """
    particles = morphology.question_particles
    if token in particles:
        return token
    if morphology.has_copula(token):
        bare = morphology.strip_copula(token)
        if bare in particles:
            return bare
    if token in inflected():
        for seed in particles:
            if token.startswith(seed):
                return seed
    return None


def interrogative_of(token, morphology):
    """Soru sözcüğü mü — koşaç almış olsa da. "neler" ile "nelerdir" aynı."""
    words = morphology.interrogatives
    if token in words:
        return token
    if morphology.has_copula(token):
        bare = morphology.strip_copula(token)
        if bare in words:
            return bare
    return None


def asks(token, morphology):
    """Bu kelime, hangi çekimde olursa olsun, soru soruyor mu?"""
    return (particle_of(token, morphology) is not None
            or interrogative_of(token, morphology) is not None)
