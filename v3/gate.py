"""Epistemik kapı: belleğe girişin ve çıkışın TEK yolu.

İki yönlü bir kapıdır ve iki soruyu sorar:

    GİRİŞTE   bu iddia yazılmalı mı — kaynağı ne, çelişiyor mu, tanığı var mı
    ÇIKIŞTA   bu cümle söylenmeli mi — içindeki her iddianın kaydı var mı

İkincisi projenin dördüncü şartıdır: desteklenmeyen iddiaya giden yol olmasın.
Bir dil modelinde bu kapı yoktur ve olamaz — orada bilgi ağırlıklara gömülü
olduğu için "bu cümle neye dayanıyor" sorusunun cevabı hesaplanamaz. Burada
her cümlenin arkasında kayıt anahtarları durur.

Dile ait hiçbir şey yoktur: ne kelime, ne kalıp, ne ek. Kapı iddiaları
KAYIT olarak alır, cümle olarak değil. Cümleyi kayda çeviren okuyucu ağıdır;
kapı yalnız kayıtlara bakar.

TEK YÖNLÜ KURAL — mutlak: geometri kayıt yazamaz. Sürekli katman aday
üretir, ayrık katman karar verir. Bu kural bir kez gevşetildiğinde sistem
uydurmaya açılır ve o zaman elde LLM'in kötü bir kopyası kalır.
"""
from v3.memory import CONTRA, INFERRED, STRANGER

# Bir iddianın söylenebilmesi için gereken en az güven. Altındakiler bellekte
# durur — atılmazlar, çünkü kaynağı vardır — ama konuşurken sayılmazlar.
SPEAK = 0.4

# Bir yabancının, hiç kimsenin doğrulamadığı sözü. Bellekte durur, cevapta
# işaretlenir. Eski bellekte ölçüldü: işaretsizken tek bir yabancı, tek bir
# cümleyle grafın sesini değiştirebiliyordu.
UNVERIFIED = 1


class Gate:
    """Belleğin önünde duran denetim. Bellek dışarıdan doğrudan yazılamaz."""

    def __init__(self, memory):
        self.memory = memory
        # Opsiyonel anlamsal RAKİP kontrolü: (eski_değer_key, yeni_değer_key)->bool.
        # LMM'de session bunu Qwen'e bağlar. Yüklem bilinmediğinde (predicate=None)
        # iki farklı değer ancak RAKİP (aynı yuva, birbirini dışlayan) ise çelişki
        # sayılır — yoksa "kartal→kuş, kartal→yırtıcı" gibi bir arada var olan
        # olgular yanlışlıkla çelişki sanılıp arbitrate ile bozuluyordu.
        self.rival = None

    # --- giriş ----------------------------------------------------------

    def admit(self, subject, predicate, value, source, level=None,
              episodic=True):
        """Bir iddiayı belleğe koymayı dener.

        Dönen: (kayıt | None, sebep). Sebep bir SAYIDIR — dil değil:
            0  yazıldı ya da pekişti
            1  aynı yuvada karşıt kayıt var, güveni bundan yüksek
            2  kaynak tanınmıyor
        """
        if source is None:
            return None, 2
        from v3.memory import DOCUMENT
        level = DOCUMENT if level is None else level
        clash = self._contradiction(subject, predicate, value)
        if clash is not None:
            trust = self.memory.trust_of(level)
            # Çelişki bağı HER DURUMDA kurulur. Denetim yakaladı: yeni iddia
            # güçlüyken bağ kurulmuyordu ve `arbitrate`/`pressure` o
            # çelişkiyi hiç göremiyordu — sistemin rahatsız olması gereken
            # yer, kayıtsız kalıyordu.
            weaker = clash.trust >= trust
            held = self.memory.write(subject, predicate, value, source,
                                     level,
                                     trust=trust * 0.5 if weaker else None,
                                     episodic=True)
            self.memory.link(held.key, clash.key, CONTRA)
            self.memory.link(clash.key, held.key, CONTRA)
            return held, (1 if weaker else 0)
        return self.memory.write(subject, predicate, value, source, level,
                                 episodic=episodic), 0

    def _contradiction(self, subject, predicate, value):
        """Aynı özne + aynı yüklemde BAŞKA değer taşıyan RAKİP kayıt.

        İki farklı değer ÇELİŞKİ mi — YÜKLEMDEN BAĞIMSIZ, değerlerin RAKİP olup
        olmadığına bakılır. `rival` bağlıysa (LMM): hiyerarşik bağlı değerler
        (kedigil/memel — kedigil bir memelidir) rakip DEĞİL → çelişki değil, bir
        arada var olurlar; bağsız/dışlayan değerler (paris/berlin, kuş/balık) →
        çelişki. `rival` bağlı değilse (v3) yüklem eşleşmesi çelişki sayılır (eski).

        DÜZELTME (test-kanıtı): rakip kontrolü eskiden yalnız predicate=None
        dalındaydı; açık yüklem ("tür", is-a) o kontrolü atlayıp aslan
        kedigil/memel'i sahte çelişki sayıyordu. Artık tüm yüklemlerde uygulanır."""
        for held in self.memory.about(subject, touch=False):
            if held.predicate != predicate or held.value == value:
                continue
            if self.rival is not None and not self.rival(held.value, value):
                continue          # bağlı/hiyerarşik değerler → çelişki değil
            return held
        return None

    # --- çıkış ----------------------------------------------------------

    def supported(self, claims):
        """Söylenmek istenen iddiaların hangileri kayıtla destekli.

        `claims`: [(özne, yüklem, değer)] — üretici ağın söylemek istediği.
        Dönen: (geçenler, düşenler). Düşenler SÖYLENMEZ.

        Bu, üretici ağın önündeki son duvardır. Ağ akıcı olabilir, hatta
        kendinden emin olabilir; kaydı yoksa cümle düşer.
        """
        passed, dropped = [], []
        for claim in claims:
            record = self.behind(*claim)
            if record is not None and record.trust >= SPEAK:
                passed.append((claim, record))
            else:
                dropped.append(claim)
        return passed, dropped

    def behind(self, subject, predicate, value):
        """Bu iddianın arkasındaki kayıt — yoksa None.

        "Bunu neden söyledin" sorusunun cevabı budur ve hesaplanabilir
        olması bu mimarinin ayırt edici yanıdır.
        """
        for held in self.memory.about(subject, touch=False):
            # Yüklem None ise JOKER: değer eşleşmesi yeter. Sebep ölçüldü —
            # okuyucu yüklemi cümlelerin ~%0,5'inde çıkarıyor; yüklem eşleşmesi
            # zorunlu tutulunca özneli+değerli ama yüklemsiz doğal cümleler
            # (kayıtları da yüklemsiz saklanır) desteksiz sayılıp düşüyordu.
            # Özne+değer eşleşmesi uydurma engeli için zaten yeterli.
            if held.value == value and (predicate is None
                                        or held.predicate == predicate):
                return held
        return None

    def doubtful(self, record):
        """Bu kayıt yalnız tek bir düşük basamaklı sözcüğe mi dayanıyor.

        Söylenirken işaretlenmeli: bilgi atılmıyor ama aynı sesle
        konuşmuyor.
        """
        # Kaynağın BASAMAĞINA bakılır, güven DEĞERİNE değil. İlk yazışta
        # `trust_of(source) <= STRANGER` yazmıştım: 0,6 <= 1 her zaman doğru
        # çıkıyor ve belge kayıtları da kuşkulu görünüyordu. Basamak ile
        # değer aynı ölçekte değil; karşılaştırılmaları da bir hataydı.
        return record.witnesses <= UNVERIFIED and record.level == STRANGER

    def inferred(self, subject, predicate, value, because):
        """Çıkarımla türetilen kayıt — düşük güvenle, gerekçesine bağlı.

        Türetme motorunun belleğe dokunduğu tek nokta burasıdır ve buradan
        geçen her kayıt kaynağını "çıkarım" olarak taşır: sonradan hangi
        bilginin gözlemden hangisinin akıl yürütmeden geldiği ayrılabilir.
        """
        held = self.memory.write(subject, predicate, value, "#inference",
                                 INFERRED, episodic=False)
        for key in because:
            self.memory.link(held.key, key, 3)      # koşul bağı
        return held

    # --- yaşantı --------------------------------------------------------

    def weigh(self, expected, actual):
        """Şaşkınlık: tahmin ile gerçek arasındaki fark.

        Öngörücü kodlamanın karşılığı. Tahmin tuttuysa yaşantı ucuzdur ve
        belleği meşgul etmez; tutmadıysa pahalıdır ve kalıcı olur. Sobaya
        bir kez dokunmak bin tekrardan bu yüzden güçlüdür.
        """
        return min(1.0, abs(float(expected) - float(actual)))
