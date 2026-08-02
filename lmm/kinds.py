"""What a relation *does*, as data rather than as code.

Relations used to be constants the reasoning branched on, so every new kind of
fact — possession, quantity, condition, sequence — meant editing the engine. The
evidence says that is the wrong shape, and also that the opposite extreme does
not work: relation types cannot be discovered from text. Open relation
extraction produces clusters that do not align with semantic classes and cannot
be named, and distributional similarity famously cannot separate opposites —
"caused" and "prevented" occur in the same contexts — which is precisely the
distinction a reasoner needs most.

What does work is a closed inventory: WordNet has about ten relations, ATOMIC
twenty-three, ConceptNet thirty-six, Universal Dependencies thirty-seven. So a
core ships here and anything else is loaded as data.

A reasoner needs to know very little about a relation. A survey of some 1300
ontologies found most of them never exceed the simplest expressive class: what
carries weight is inheritance, inversion, transitivity, and occasionally being
single-valued. Reflexivity, property chains and cardinality arithmetic cost more
than they return, so they are not here.

Two fields, not one. Negation and inversion are different things — "not the
parent of" is not "the child of" — and folding them together is the modelling
mistake that produces confident nonsense later.
"""
from lmm.relations import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,
                           LACKS_PROPERTY, HAS_PART, LACKS_PART, MUST,
                           MUST_NOT, SAME_AS, REQUIRES,
                           LACKS_REQUIREMENT)


class Kind:
    def __init__(self, name, inherits=True, transitive=False, functional=False,
                 negation_of=None, inverse_of=None, hierarchical=False,
                 label="", denies=None):
        self.name = name
        self.inherits = inherits        # does it carry down the type hierarchy
        self.transitive = transitive    # a is R b, b is R c, therefore a is R c
        self.functional = functional    # at most one value — a second is a clash
        self.negation_of = negation_of  # the relation this denies
        self.inverse_of = inverse_of    # the same fact seen from the other end
        self.hierarchical = hierarchical    # this is what ancestors walk
        self.label = label or name
        # Çiftin hangi ucu olduğu: True yadsıyan uç, False olumlayan uç.
        # None "bildirilmedi" demektir ve False'tan farklıdır — eski kayıtlardan
        # yüklenen bir ilişkinin kutbunu, künyesinde yazmıyor diye "olumlu"
        # saymak, tam da düzeltilen hatayı geri getirirdi.
        self.denies = denies

    def to_dict(self):
        return {"name": self.name, "inherits": self.inherits,
                "transitive": self.transitive, "functional": self.functional,
                "negation_of": self.negation_of, "inverse_of": self.inverse_of,
                "hierarchical": self.hierarchical, "label": self.label,
                "denies": self.denies}

    @staticmethod
    def from_dict(data):
        return Kind(**data)


# Her çiftin İKİ ucu da kutbunu bildirir. Yalnız yadsıyan ucu işaretlemek de
# yeterdi, ama bir ilişki tek başına veri olarak eklendiğinde eşi henüz
# yüklenmemiş olabilir; künyesi kendi başına okunabilsin diye ikisi de yazılı.
CORE = [
    Kind(IS_A, inherits=False, transitive=True, hierarchical=True,
         negation_of=NOT_A, denies=False, label="türü"),
    Kind(NOT_A, inherits=False, negation_of=IS_A, denies=True,
         label="türü değil"),
    Kind(CAN, negation_of=CANNOT, denies=False, label="yapabilir"),
    Kind(CANNOT, negation_of=CAN, denies=True, label="yapamaz"),
    Kind(HAS_PROPERTY, negation_of=LACKS_PROPERTY, denies=False,
         label="niteliği"),
    Kind(LACKS_PROPERTY, negation_of=HAS_PROPERTY, denies=True,
         label="niteliği değil"),
    Kind(HAS_PART, negation_of=LACKS_PART, denies=False, label="sahip"),
    # Gereklilik, bir belgenin en çok söylediği şey: "şifrelenmeli",
    # "kaydedilmelidir". İlişkiler veri olduğu için eklemek motoru
    # değiştirmiyor — bir satır.
    Kind(MUST, negation_of=MUST_NOT, denies=False, label="yapılmalı"),
    Kind(MUST_NOT, negation_of=MUST, denies=True, label="yapılmamalı"),
    Kind(LACKS_PART, negation_of=HAS_PART, denies=True, label="sahip değil"),
    # Eş anlamlılık geçişli ve kendi tersidir: "a=b" ise "b=a", ve "a=b, b=c"
    # ise "a=c". Kalıtmaz — bir kelimenin eşi olmak, o kelimenin türlerinin de
    # eşi olmak demek değil.
    Kind(SAME_AS, inherits=False, transitive=True, inverse_of=SAME_AS,
         label="aynı anlama gelir"),
    # Önkoşul kalıtılır: "kuş uçmak için kanat gerektirir" ise kartal için de
    # geçerlidir. Geçişli DEĞİL — A, B'yi; B, C'yi gerektiriyorsa A'nın C'yi
    # gerektirdiği söylenemez; aradaki koşul kopabilir.
    Kind(REQUIRES, negation_of=LACKS_REQUIREMENT, denies=False,
         label="gerektirir"),
    Kind(LACKS_REQUIREMENT, negation_of=REQUIRES, denies=True,
         label="gerektirmez"),
]


class Kinds:
    """The relations a memory knows how to reason about."""

    def __init__(self, kinds=None):
        self.by_name = {}
        for kind in (kinds if kinds is not None else CORE):
            self.add(kind)

    def add(self, kind):
        # Kutbunu bildirmeyen bir künye, bilinen kutbu SİLMEMELİ. Kayıtlı
        # modeller bu alan yokken yazıldı (birlesik.lmm'deki on üç ilişkinin
        # hiçbirinde yok); veri olduğu gibi üzerine yazsaydı, dosyadan yüklenen
        # her oturumda "kanadı var"dan "kanadı yok" çelişkisi geri gelirdi.
        known = self.by_name.get(kind.name)
        if kind.denies is None and known is not None:
            kind.denies = known.denies
        self.by_name[kind.name] = kind
        return kind

    def get(self, name):
        return self.by_name.get(name) or Kind(name)

    def negation(self, name):
        return self.get(name).negation_of

    def pair(self, name):
        """(affirming, denying) for a relation, whichever end was asked about."""
        other = self.get(name).negation_of
        if other is None:
            return name, None
        return (other, name) if self._denies(name, other) else (name, other)

    def _denies(self, name, other):
        """Çiftin yadsıyan ucu bu mu — hangi uçtan sorulursa sorulsun aynı yanıt.

        Kutup, ilişkinin KENDİ künyesinden okunur. Eskiden adın "not_" ile
        başlamasına bakılıyordu ve envanterdeki `has_not` ile `must_not` bu
        kalıba uymuyordu: `pair()` bu iki ilişkide olumlu/olumsuz sırasını ters
        döndürüyordu. Ölçülen sonuç, projenin merkez iddiasının ihlaliydi —
        bellekte yalnızca ('kuş','has','kanadı') varken "kuşun kanadı yok"
        cümlesi üretiliyor ve "(kaynak: sen)" diye öğretmene atfediliyordu.
        Ada bakmak dilin içine kaçmaktı; ilişkiler veri olduğuna göre kutup da
        veridir, ve yeni bir ilişki veri olarak eklendiğinde kod değişmeden
        doğru kurulur.
        """
        kind, partner = self.get(name), self.get(other)
        if kind.denies is not None:
            return kind.denies
        if partner.denies is not None:
            return not partner.denies
        # Hiçbir uç bildirmemişse envanterdeki bildirim sırasına düşülür: önce
        # tanıtılan uç olumlu sayılır. Bu bir tahmin, ama iki uçtan da AYNI
        # cevabı veren kararlı bir tahmin — çelişki uydurmasının kaynağı
        # kutbun yanlış olması değil, iki yönde birbirini tutmamasıydı.
        order = list(self.by_name)
        if name in order and other in order:
            return order.index(name) > order.index(other)
        return False

    def inherits(self, name):
        return self.get(name).inherits

    def hierarchical(self):
        for kind in self.by_name.values():
            if kind.hierarchical:
                return kind.name
        return IS_A

    def known(self):
        return sorted(self.by_name)

    def to_list(self):
        return [kind.to_dict() for kind in self.by_name.values()]

    def load(self, entries):
        """Add relations declared as data. Returns the names that were new."""
        added = []
        for entry in entries:
            kind = Kind.from_dict(entry)
            if kind.name not in self.by_name:
                added.append(kind.name)
            self.add(kind)
        return added


DEFAULT = Kinds()
