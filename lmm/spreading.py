"""Grafta yayılım: bir kavramdan yola çıkıp neyin ilgili olduğunu bulmak.

Şimdiye kadar graf düz bir aramaydı: "kartal" sorulunca kartalın kayıtlarına
bakılıyor, orada bitiyordu. Oysa bilgi bir ağ ve ağın değeri komşuluklarda —
kartal bir kuştur, kuşlar uçar, uçan başka neler var, onlar da tüylü mü.

Beyinde bunun karşılığı hipokampüsün **örüntü tamamlaması**: bir parçadan
bütünü çağırmak. HippoRAG (NeurIPS 2024) tam bu benzetmeyi kurdu — düzenlenebilir
graf hipokampal indeks, üzerinde yürütülen kişiselleştirilmiş PageRank örüntü
tamamlaması — ve çok adımlı sorularda %20 kazanç, 10-20 kat ucuz maliyetle.

Matematiği de aynı yere çıkıyor: kişiselleştirilmiş PageRank ile hipokampüsün
"ardıl temsili" aynı operatördür, (I - αT)⁻¹.

Burada üç şey bilinçli:

    eğitim yok      yayılım bir hesap, bir model değil; graf değişince anında
                    yansır ve yeniden eğitim gerekmez
    yön yok         ilişkiler iki yönlü gezilir: "kuş uçar" bilgisi hem kuştan
                    uçmaya hem uçmadan kuşa götürür
    sönüm var       her adımda ilgi azalır, yoksa yayılım grafın tamamına
                    dağılır ve hiçbir şey söylemez

Ve bir uyarı, ölçülmüş: graf yapısı çağrışımı artırırken düz olgu erişimini
bozabiliyor. Bu yüzden yayılım, doğrudan aramanın YERİNE geçmiyor — yanına
konuyor. Bir soru doğrudan cevaplanabiliyorsa öyle cevaplanır.
"""
import collections

DAMPING = 0.5           # her adımda ilginin ne kadarı komşuya geçer
ROUNDS = 3              # kaç adım uzağa bakılır — üçten sonrası dağılıyor
SMALLEST = 0.01         # bunun altındaki ilgi gürültüdür


def neighbours(memory):
    """Kavramdan kavrama, iki yönlü. Bir kez kurulur, tekrar kullanılır."""
    links = collections.defaultdict(set)
    for edge in memory.edges:
        for other in (edge.target, edge.object):
            if not other or other == edge.concept:
                continue
            links[edge.concept].add(other)
            links[other].add(edge.concept)
    return links


def spread(memory, seeds, links=None, rounds=ROUNDS, damping=DAMPING):
    """Tohum kavramlardan yayılan ilgi. {kavram: ilgi} döner, sıralı değil.

    Tohumlar ilgiyi 1,0 ile başlatır ve her turda komşularına dağıtırlar. Bir
    kavram birden çok yoldan besleniyorsa ilgisi toplanır — "hem kuş hem uçan"
    olan bir şey, yalnız birinden gelenden daha ilgilidir. Bu, çok adımlı
    çağrışımın tamamı.
    """
    links = links if links is not None else neighbours(memory)
    seeds = [s for s in seeds if s in links or memory.query(s)]
    if not seeds:
        return {}
    interest = {seed: 1.0 for seed in seeds}
    for _ in range(rounds):
        passed = collections.defaultdict(float)
        for concept, value in interest.items():
            around = links.get(concept)
            if not around:
                continue
            share = value * damping / len(around)
            for other in around:
                passed[other] += share
        if not passed:
            break
        for concept, value in passed.items():
            interest[concept] = interest.get(concept, 0.0) + value
    for seed in seeds:
        interest.pop(seed, None)        # tohumun kendisi cevap değil
    return {c: v for c, v in interest.items() if v >= SMALLEST}


def related(memory, seeds, count=8, links=None):
    """En ilgili kavramlar, ilgiden sıraya. Beraberlikte ad sırası — kararlılık
    için: aynı graf her zaman aynı cevabı vermeli."""
    found = spread(memory, seeds, links)
    return [concept for _, concept in
            sorted(((-value, c) for c, value in found.items()))][:count]


def bridge(memory, first, second, links=None, rounds=ROUNDS):
    """İki kavram arasındaki bağlantı: ikisinden de beslenen kavramlar.

    "kartal ile penguen arasındaki bağ nedir" sorusunun cevabı, ikisinin de
    ilgisini alan şeydir — bu örnekte "kuş". Düz arama bunu bulamaz, çünkü
    hiçbir kayıtta ikisi birlikte geçmiyor.
    """
    links = links if links is not None else neighbours(memory)
    left = spread(memory, [first], links, rounds)
    right = spread(memory, [second], links, rounds)
    shared = set(left) & set(right)
    return [concept for _, concept in
            sorted(((-(left[c] * right[c]), c) for c in shared))]
