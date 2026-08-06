"""Belleğin yaşayan kısmı: pekişme, solma, damıtma, hakemlik.

Bir veritabanı yazdığını olduğu gibi tutar. Bellek tutmaz — kullanılan
güçlenir, kullanılmayan söner, tekrarlanan kalıcılaşır, çelişen hakemlenir.
Bu dosya o dört hareketi kurar ve "yaşayan bellek" adının hak edildiği yer
burasıdır.

Dayanağı beynin ikili yapısı (tamamlayıcı öğrenme sistemleri):

    HIZLI   hipokampus   tek seferde yazar · EPİSODİK · çabuk değişir
    YAVAŞ   korteks      tekrarla damıtır · SEMANTİK · kalıcı

Uykuda hipokampus gündüz yazdığını kortekse tekrar oynatır; işe yarayan
kalıcılaşır, gerisi silinir. `sleep()` bunun karşılığıdır ve gradyan
istemez, GPU istemez — sayım ve eşiktir.

Bir dil modelinde bu döngünün karşılığı yoktur. Orada bir bilgi ya eğitimde
ağırlıklara girmiştir ya hiç yoktur; ne pekişir, ne solar, ne damıtılır.

Dile ait hiçbir şey yoktur. Buradaki her şey sayaç ve eşiktir.
"""
import time

# Bir episodik kaydın semantiğe yükselmesi için kaç bağımsız tanık gerekir.
# İki, çünkü tek tanık bir olaydır; ikinci tanık onu bir düzenlilik yapar.
WITNESSES_TO_SETTLE = 2

# Semantiğe yükselirken güvenin ulaşacağı taban. Yükselmek bir terfidir ve
# terfi eden kayıt artık tek bir konuşmanın rehinesi değildir.
SETTLED_TRUST = 0.65

# Solma hızı: erişilmeyen bir kaydın güveninin her turda kaybettiği pay.
# Küçük tutuluyor çünkü unutmak ucuz olmamalı — bilgi kaybı geri alınamaz.
FADE = 0.03

# Bu güvenin altına düşen ve hiç erişilmeyen kayıt artık konuşmaz. Silinmez:
# kaynağı vardır ve bir gün ikinci bir tanık gelebilir.
FLOOR = 0.1

# Bir turda solmaktan muaf tutulacak tazelik penceresi (saniye). Yeni yazılan
# bir kayıt daha erişilmeye fırsat bulamadan sönmemeli.
FRESH = 3600.0


def reinforce(memory, record, source, level=None):
    """Aynı olguyu başka bir kaynak da söyledi — kuşkunun bir payı kapanır.

    Toplamsal değil: eski bellekte toplamsalken dördüncü belge tavanı
    deliyordu. Her tanık kalan kuşkunun sabit bir payını kapatır, tavana
    yaklaşılır ama ulaşılmaz.
    """
    return record.strengthen(source, level)   # koruma ve hesap tek yerde


def settle(memory, record):
    """Episodik kaydı semantiğe yükseltir — yeterince tanık varsa.

    "Ali dedi ki" ile "bilinen bir şey" arasındaki fark budur ve zamanla
    kendiliğinden oluşur: bir olay, tekrarlana tekrarlana bilgi olur.
    """
    if not record.episodic:
        return False
    if record.witnesses < WITNESSES_TO_SETTLE:
        return False
    record.episodic = False
    record.trust = max(record.trust, SETTLED_TRUST)
    return True


def fade(memory, record, now=None):
    """Erişilmeyen kayıt söner. Silinmez — sesi kısılır.

    Tazelik penceresindeki kayıtlara dokunulmaz: yeni yazılan bir şey,
    erişilmeye fırsat bulamadan sönmemeli.
    """
    now = time.time() if now is None else now
    if now - record.last_seen < FRESH:
        return False
    if record.witnesses > 1 or not record.episodic:
        return False        # tanıklı ya da yerleşmiş kayıt solmaz
    from v3.memory import OPERATOR
    if record.level >= OPERATOR:
        return False        # işletmecinin öğrettiği sessizce çürümez
    before = record.trust
    record.trust = max(FLOOR, record.trust - FADE)
    return record.trust < before


def arbitrate(memory, record):
    """Çelişen kayıtlar arasında hakemlik — kim daha ağır basıyor.

    Karar iki ölçüte dayanır ve ikisi de kayıtlıdır: kaynağın basamağı ve
    tanık sayısı. Kaybeden SİLİNMEZ, yalnız konuşmaz — çünkü bir gün
    kazanabilir ve o gün geldiğinde geçmişi durmalıdır.

    Dönen: kazanan kayıt (kendisi de olabilir).
    """
    from v3.memory import CONTRA
    rivals = [memory.records[key] for kind, key in record.links
              if kind == CONTRA and key in memory.records]
    if not rivals:
        return record
    field = [record] + rivals
    field.sort(key=lambda one: (-one.level, -one.witnesses, -one.trust))
    winner = field[0]
    for other in field[1:]:
        if other.trust >= winner.trust:
            other.trust = winner.trust * 0.5
    return winner


# Uyku turunda tutulacak en çok yaşantı. Şaşırtan kalır, sıradan gider —
# 3. tur denetimi ölçtü: 1000 cevap 1000 yaşantıydı ve hiçbir bakım
# dokunmuyordu; her "hmm" sonsuza dek dosyada büyüyordu.
MOST_EXPERIENCES = 2000


def sleep(memory, now=None):
    """Uyku turu: damıt, sönümle, hakemle.

    Hipokampal tekrarın karşılığı ve tek turda üç iş yapar. Gradyan yok,
    GPU yok — bir tarama ve eşiklerdir.

    Dönen: sayım sözlüğü — kaç kayıt yerleşti, kaç sönümlendi, kaç hakemlendi.
    Rapor DİL DEĞİL sayıdır; anahtarlar bu dosyanın kendi kimlikleri.
    """
    now = time.time() if now is None else now
    counted = {"settled": 0, "faded": 0, "judged": 0}
    from v3.memory import CONTRA
    for record in list(memory.records.values()):
        if settle(memory, record):
            counted["settled"] += 1
        if fade(memory, record, now):
            counted["faded"] += 1
        if any(kind == CONTRA for kind, _ in record.links):
            arbitrate(memory, record)
            counted["judged"] += 1
    # Yaşantı budaması: sıradan (sonuçsuz, şaşırtmamış) eskiler gider.
    if len(memory.experiences) > MOST_EXPERIENCES:
        ranked = sorted(memory.experiences.values(),
                        key=lambda one: (abs(one.outcome) + one.surprise,
                                         one.at))
        for stale in ranked[:len(ranked) - MOST_EXPERIENCES]:
            del memory.experiences[stale.key]
            counted["pruned"] = counted.get("pruned", 0) + 1
    return counted


def pressure(memory):
    """Çelişki basıncı: sistem neye rahatsız olmalı.

    Değerlendirme sinyallerinden biri. Yüksek basınç, gidip doğrulama
    yapmayı hak eden yerleri gösterir — merakın nereye bakacağını bu söyler.

    Dönen: [(kayıt anahtarı, basınç)], yüksekten alçağa.
    """
    from v3.memory import CONTRA
    found = []
    for record in memory.records.values():
        rivals = [memory.records[key] for kind, key in record.links
                  if kind == CONTRA and key in memory.records]
        if not rivals:
            continue
        # İki taraf da güçlüyse basınç yüksek: zayıf bir itiraz rahatsız
        # etmez, denk iki iddia eder.
        best = max(one.trust for one in rivals)
        found.append((record.key, min(record.trust, best)))
    found.sort(key=lambda pair: -pair[1])
    return found


def gaps(memory, most=20):
    """Merak: belleğin kendi boşlukları.

    Bir kimlik hakkında hiç kaydı yoksa ya da yalnız episodik kaydı varsa,
    orası öğrenilecek yerdir. Sistem neyi bilmediğini bilir ve nereye
    bakacağını kendisi söyler.

    Dönen: [(kimlik anahtarı, boşluk büyüklüğü)], büyükten küçüğe.
    """
    found = []
    for key, identity in memory.identities.items():
        records = memory.by_subject.get(key, ())
        if not records:
            found.append((key, 1.0 + identity.seen))
            continue
        settled = sum(1 for one in records
                      if not memory.records[one].episodic)
        if not settled:
            found.append((key, 0.5 + identity.seen))
    found.sort(key=lambda pair: -pair[1])
    return found[:most]
