"""Okumak = en ucuz açıklamayı bulmak.

Bugüne kadar ayrıştırıcı ilk uyan kalıbı alıyordu — `match()` işlevinin kendi
açıklaması bunu söylüyor: "First pattern that fits". Rakip okumalar üretiliyor
ve sessizce atılıyordu. Hangi okumanın doğru olduğuna karar veren bir mekanizma
hiç yoktu.

Ölçtük: küçük bir grafla okuma %26 iyileşiyor, 35 bin kavramlı grafla geri
düşüyor. Yani bilgi elimizde ama *belirsizliği azaltmak için* kullanılmıyor;
yalnızca "bu kelime tanıdık mı" diye soruluyor ve 35 bin kavramla her şey
tanıdık görünüyor.

Eksik olan fikrin adı var: Hobbs, Stickel, Appelt ve Martin'in 1988'de ortaya
attığı **yorumlama = kaçınım** (interpretation as abduction). İddia şu: bir
metnin yorumu, o metnin neden doğru olacağının en ucuz açıklamasıdır. Bildiğinden
kanıtlayabildiğin bedavadır; kanıtlayamadığını varsayarsın ve her varsayımın bir
bedeli vardır. En az varsayım gerektiren okuma, doğru okumadır.

O yaklaşım iki sebeple terk edilmişti: kanıt araması 1990'da çözülemiyordu, ve
**her aksiyomu elle yazmak gerekiyordu**. Birincisini dünya çözdü. İkincisini
biz çözdük — çünkü önce grafı kurduk. 37.039 olgu, tam da Hobbs'un elinde
olmayan şey.

Buradaki uygulama kasten en yalın hâli: maliyet, okumanın gerektirdiği yeni
varsayım sayısı. Grafta zaten duran her kavram bedava; her yeni kavram bir
varsayım. Tam kanıt araması yok, tamsayılı programlama yok — önce fikrin
ölçülebilir en küçük hâli.
"""
from lmm.relations import IS_A

# Bilinen bir kavramı kullanmak bedava; yenisini varsaymak bir birim. Kökü
# grafta olan bir eylem yarım varsayım sayılır: "uçmak" biliniyorsa "uçabilmek"
# tamamen yabancı değildir.
NEW_CONCEPT = 1.0
NEW_TARGET = 1.0
FAMILIAR = 0.4
CONTRADICTION = 3.0     # bildiğinin tersini söyleyen okuma pahalıdır


class Reading:
    """Bir cümlenin olası okumalarından biri, maliyetiyle birlikte."""

    def __init__(self, pattern, captured, parts, cost, assumptions):
        self.pattern = pattern
        self.captured = captured
        self.parts = parts          # (kind, relation, concept, target, ...)
        self.cost = cost
        self.assumptions = assumptions      # neyin varsayıldığı, okunabilir

    def __repr__(self):
        kind, relation, concept, target = self.parts[:4]
        return f"{concept} {relation} {target} (maliyet {self.cost:.1f})"


def cost_of(parts, memory):
    """Bu okuma, bildiklerimize göre kaç varsayım gerektiriyor?

    Ucuz olan, açıklanabilen okumadır. Bir kavramı zaten tanıyorsak onu
    kullanmak bedavadır; hiç duymadığımız bir şeyi kavram saymak bir bedeldir.
    """
    _, relation, concept, target = parts[:4]
    known = memory.concepts()
    actions, properties = memory.actions(), memory.properties()
    cost, assumptions = 0.0, []

    if concept:
        if concept in known:
            pass
        elif any(concept.startswith(other[:4]) for other in known if len(other) > 3):
            cost += FAMILIAR
            assumptions.append(f"{concept} tanıdık ama tam bilinmiyor")
        else:
            cost += NEW_CONCEPT
            assumptions.append(f"{concept} yeni bir kavram")

    if target:
        if target in known or target in actions or target in properties:
            pass
        else:
            cost += NEW_TARGET
            assumptions.append(f"{target} yeni")

    # Bildiğinin tersini iddia eden okuma, aynı şeyi söyleyenden pahalıdır.
    # Bu, çelişkiyi yasaklamak değil — çelişki gerçekten varsa yine öğrenilir,
    # ama iki okuma arasında seçim yapılırken uyumlu olan tercih edilir.
    if concept and target and relation:
        opposite = memory.kinds.negation(relation)
        if opposite and memory.direct(concept, opposite, target) is not None:
            cost += CONTRADICTION
            assumptions.append(f"bildiğimin tersi: {concept} {opposite} {target}")

    return cost, assumptions


def readings(grammar, tokens, lexicon, memory, most=6):
    """Uyan her kalıbı bir okuma olarak döndürür, ucuzdan pahalıya.

    `most` bir güvenlik sınırı: bir cümlede onlarca kalıp uyabilir ve hepsini
    puanlamak, kazandırdığından fazlasını götürür.
    """
    found = []
    for pattern in grammar.patterns:
        captured = grammar._fit(pattern, tokens, lexicon)
        if captured is None:
            continue
        parts = grammar.read(pattern, captured)
        cost, assumptions = cost_of(parts, memory)
        found.append(Reading(pattern, captured, parts, cost, assumptions))
        if len(found) >= most:
            break
    found.sort(key=lambda reading: reading.cost)
    return found


def best(grammar, tokens, lexicon, memory):
    """En ucuz okuma, ya da hiçbiri uymuyorsa None."""
    found = readings(grammar, tokens, lexicon, memory)
    return found[0] if found else None
