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

GREETING = "selamlama"
FAREWELL = "vedalaşma"
THANKS = "teşekkür"
WELLBEING = "hatır"
IDENTITY = "kimlik"
ABILITY = "yetenek"

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

# Cevaplar. Sistemin kendisi hakkında söylediği her şey doğrudur — bunlar
# uydurma değil, kodun kendi gerçekleri.
REPLIES = {
    GREETING: "merhaba. bildiğim şeyleri sorabilirsin.",
    FAREWELL: "görüşmek üzere. öğrendiklerim kayıtlı kalıyor.",
    THANKS: "rica ederim.",
    WELLBEING: "iyiyim. {facts} bilgi ve {concepts} kavram tutuyorum.",
    IDENTITY: ("ben LMM'im — yaşayan bellek modeli. bildiklerim ağırlıklarda "
               "değil, okunabilir bir bellekte duruyor: şu an {facts} bilgi, "
               "{concepts} kavram. bilmediğimi uyduramam."),
    ABILITY: ("bildiğim şeyleri sorabilirsin, bana yeni bilgi öğretebilirsin, "
              "yanlışımı tek cümleyle düzeltebilirsin. her cevabımda kaynağımı "
              "söylerim, bilmediğimde de bilmediğimi."),
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
    for kind in (WELLBEING, IDENTITY, ABILITY, THANKS, FAREWELL, GREETING):
        for form in EXCHANGES[kind]:
            if form in text:
                return kind
    return None


def reply(kind, memory=None):
    """Bu alışverişin cevabı. Sayılar bellekten okunur, uydurulmaz."""
    template = REPLIES.get(kind)
    if template is None:
        return None
    if memory is None:
        return template.replace("{facts}", "birçok").replace("{concepts}", "birçok")
    return template.format(facts=len(memory.edges),
                           concepts=len(memory.concepts()))
