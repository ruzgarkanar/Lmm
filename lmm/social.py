"""Sohbetin bilgi taşımayan kısmı: selamlaşma, teşekkür, kendini tanıtma.

Sekiz turluk gerçek bir sohbet denemesinde beş tur "anlamadım" ile geçti ve
üçü buydu: "selam", "nasılsın", "teşekkürler". Sistem bunları bilgi sorusu
sanıp grafta arıyor, bulamıyor ve reddediyordu.

Oysa bunlar bilgi sorusu değil. Bir insanın "nasılsın" sorusuna cevap vermesi
için hiçbir şey *bilmesi* gerekmez; sosyal bir alışveriş yapması gerekir.

Bu, LMM'in epistemik kapısını ihlal etmiyor — çünkü buradaki cevaplar bir olgu
iddiası taşımıyor. "Merhaba" doğru ya da yanlış olamaz. Sistemin kendisi
hakkında söylediği şeyler ise (ne olduğu, ne yapabildiği) kodun kendi
gerçekleridir, uydurma değil.

Kapalı bir sınıf: sayıca sabit, dile özel, ve bir dil için bir kez yazılır.
Bu gecenin ölçülmüş dersi — derleme ancak kapalı bir sınıf sorulabilir — burada
da geçerli.
"""
from lmm.intuition import lower, tokenize
from lmm import phrasing

# Alışverişin KİMLİĞİ, adı değil: bu dizgiler kullanıcıya hiç görünmüyor,
# yalnızca `EXCHANGES` ile `REPLIES` arasında eşleşiyorlar. Türkçe yazılıydı ve
# bu bir dil sızıntısı gibi duruyordu; değil — ama ilişki adları (`lmm/
# relations.py`) ve niyet türleri (`lmm/intuition.py`) İngilizce, dolayısıyla
# Türkçe olmaları tutarsızlıktı. Nötr hâle getirildi: ikinci bir dil eklendiğinde
# bu satırların hiçbiri değişmeyecek, yalnızca aşağıdaki sözcük listesi ve
# `lmm/phrasing.py`'deki cevaplar değişecek. Grafa yazılmıyorlar, kaydedilmiş
# hiçbir dosyada geçmiyorlar, yani değiştirmek bir şeyi bozmuyor.
GREETING = "greeting"
FAREWELL = "farewell"
THANKS = "thanks"
WELLBEING = "wellbeing"
IDENTITY = "identity"
ABILITY = "ability"

# Her biri, o niyeti taşıyan sözler. Kelime kelime değil, cümlenin tamamı ya da
# içinde geçen anahtar aranıyor — "selam", "selam nasılsın", "merhaba dostum".
EXCHANGES = {
    GREETING: ("selam", "merhaba", "günaydın", "iyi akşamlar", "iyi günler",
               "selamlar", "hey", "alo"),
    FAREWELL: ("görüşürüz", "hoşça kal", "hoşçakal", "bay", "iyi geceler",
               "kendine iyi bak", "güle güle"),
    THANKS: ("teşekkür", "teşekkürler", "sağ ol", "sağol", "eyvallah",
             "minnettarım"),
    WELLBEING: ("nasılsın", "naber", "ne haber", "nasıl gidiyor",
                "iyi misin", "keyifler nasıl"),
    IDENTITY: ("kimsin", "sen kimsin", "adın ne", "nesin", "kendini tanıt",
               "kendini tanıtır mısın", "sen nesin"),
    ABILITY: ("ne yapabilirsin", "neler yapabilirsin", "ne işe yarıyorsun",
              "nasıl kullanılır", "yardım"),
}

# Hangi alışverişe hangi söyleyiş karşılık geliyor. Cümlelerin kendisi burada
# DEĞİL: sistemin kendisi hakkında söyledikleri de Türkçe ve Türkçenin tamamı
# `lmm/phrasing.py`'de yaşıyor. Bu dosya "ne soruldu"yu bilir, "ne denir"i
# değil — ikinci bir dil bu tabloyu değiştirmeden ekleniyor.
REPLIES = {
    GREETING: phrasing.greeted,
    FAREWELL: phrasing.farewelled,
    THANKS: phrasing.thanked,
    WELLBEING: phrasing.wellbeing,
    IDENTITY: phrasing.introduced,
    ABILITY: phrasing.what_i_can_do,
}


def recognise(sentence):
    """Bu cümle bir sosyal alışveriş mi? Değilse None.

    Tam eşleşme önce denenir: "nasılsın" bir hatır sorusudur. Sonra içerik
    aranır, çünkü "selam, nasılsın" iki şeyi birden taşır ve ikincisi daha
    çok bilgi ister.
    """
    text = lower(sentence).strip(" .,!?")
    words = tokenize(sentence)
    if not words or len(words) > 6:
        return None                     # uzun cümle sosyal alışveriş değil
    for kind, forms in EXCHANGES.items():
        if text in forms:
            return kind
    # Hatır ve kimlik, selamlamadan önce bakılır: "selam nasılsın" cümlesinde
    # sorulan şey hatırdır, selamlama yalnızca girişi.
    #
    # Arama KELİME sınırında yapılır, düz alt-dizede değil. Alt-dize aranırken
    # "alo" selamlaması "balon nedir" ve "salon nedir" sorularının içinde
    # bulunuyordu: gerçek soru grafa hiç ulaşmadan "merhaba" deniyordu. Sosyal
    # kapı ayrıştırmadan ÖNCE çalıştığı için bu, bilinen bir kavramı bilinmez
    # kılan sessiz bir tıkaçtı — ve hangi kavramların tıkandığı sözcük
    # listesine bakılarak kestirilemezdi.
    for kind in (WELLBEING, IDENTITY, ABILITY, THANKS, FAREWELL, GREETING):
        for form in EXCHANGES[kind]:
            spoken = form.split()
            for start in range(len(words) - len(spoken) + 1):
                if words[start:start + len(spoken)] == spoken:
                    return kind
    return None


def reply(kind, memory=None):
    """Bu alışverişin cevabı. Sayılar bellekten okunur, uydurulmaz."""
    said = REPLIES.get(kind)
    if said is None:
        return None
    if memory is None:
        return said()       # bellek yoksa sayı da yok; söyleyiş onu biliyor
    return said(len(memory.edges), len(memory.concepts()))
