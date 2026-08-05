"""Turkish, described in one place.

The suffixes, the function words and the sentence patterns of one language. The
parser, the grammar and the reasoning know nothing about any of it — they ask a
morphology object questions like "does this word carry a copula" and match slots
against a pattern list.

A second language means another module shaped like this one, not another parser.
"""
import re

from lmm.grammar import (Pattern, Grammar, KAVRAM, TUR, NITELIK, SOZ, FIIL,
                         SORU, KIM, ROL, NICEL, SAHIP, NESNEL, FROM_VERB)
# Durum ve zaman ETİKETLERİ. Türkçe sözcüklere benziyorlar ama dil değiller:
# ölçüldü — kullanıcıya hiç görünmüyorlar, grafa hiç yazılmıyorlar, ve
# `lmm/frames.py` dışında kimse okumuyor. Yani kimliktirler, ve kimliğin yeri
# onları TANIMLAYAN organdır. Dil onlara yalnızca ek eşliyor; ikinci bir dil
# aynı etiketlere kendi eklerini eşler. Bu yön, `frames`'in modül düzeyinde
# hiçbir şey ithal etmemesi sayesinde döngü kurmuyor.
from lmm.frames import (ABILITY, ABLATIVE, ACCUSATIVE, AORIST, CONDITION,
                        COPULA, DATIVE, GENITIVE, INSTRUMENTAL, LOCATIVE,
                        OBLIGATION, PAST, PERFECT, PROGRESSIVE)
from lmm.relations import (REQUIRES, IS_A, NOT_A, CAN, HAS_PROPERTY, LACKS_PROPERTY,
                           HAS_PART, LACKS_PART, PLACE, SOURCE, ALL,
                           MOST, SOME, NO)

TEACH = "TEACH"
ASK = "ASK"
ASK_WHO = "ASK_WHO"
ASK_ABILITIES = "ASK_ABILITIES"
ASK_WHY = "ASK_WHY"
ASK_PROPERTIES = "ASK_PROPERTIES"
ASK_DESCRIBE = "ASK_DESCRIBE"
# İki kavramı karşılaştırma. Yayılım organı (`lmm/spreading.py`) kurulmuş ve
# test edilmişti ama hiçbir yerden çağrılmıyordu; cevaplayabildiği soru ise
# ölçümde başarısız olanlardan biriydi: "kartal ile penguen arasındaki fark ne".
ASK_COMPARE = "ASK_COMPARE"
# Sohbetin kendisi hakkında soru. Dünya hakkında bir soru değil, o yüzden
# cevabı graftan değil sohbet geçmişinden geliyor (`lmm/thread.py`).
ASK_THREAD = "ASK_THREAD"
# "hangisi daha hızlı, kartal mı penguen mi" — iki kavramı bir nitelikte
# sıralama. İlişki grafta zaten var (`kartal --hızlı--> [çıkış: serçe]`);
# eksik olan yalnızca soru biçimi ve sıralama işlemiydi.
ASK_WHICH_MORE = "ASK_WHICH_MORE"
# "bir kuşun uçabilmesi için ne gerekir" — önkoşul sorusu.
ASK_REQUIREMENT = "ASK_REQUIREMENT"
# "başka ne biliyorsun", "daha fazlasını söyle" — sohbetin kendisi hakkında,
# dünya hakkında değil. Cevabı graftan değil, o an konuşulan kavramın henüz
# söylenmemiş kısmından geliyor.
ASK_MORE = "ASK_MORE"
# "ne biliyorsun", "hafızanda ne var" — sistemin KENDİSİ hakkında soru.
# Yaşayan bir hafızanın ilk cevaplaması gereken soru bu ve cevaplayamıyordu:
# 486 kavram bilirken "bunu anlamadım" diyordu.
ASK_INVENTORY = "ASK_INVENTORY"
# Cevabın KENDİSİ hakkında sorular. Yeni bilgi gerektirmiyorlar: graf zaten
# güveni ve künyeyi tutuyor, yalnızca sorulmuyordu. Bir LLM bu soruları
# cevaplayamaz — kaynağı yoktur.
ASK_CERTAINTY = "ASK_CERTAINTY"      # "emin misin"
ASK_SOURCE = "ASK_SOURCE"            # "nereden biliyorsun"
# Görüş sorusu. Cevaplanmıyor ama ANLAŞILIYOR: "anlamadım" demekle "benim
# görüşüm yok" demek aynı şey değil.
ASK_OPINION = "ASK_OPINION"
ASK_HOW_MANY = "ASK_HOW_MANY"
ASK_WHERE = "ASK_WHERE"


class TurkishMorphology:
    question_particles = ("mı", "mi", "mu", "mü")
    # Soru ekinin kişi çekimi: "görmüyor MUSUN", "olur MUYUM", "biliyor
    # MUSUNUZ". Ek soru ekinin üstüne biniyor ve bildirilmediği için soru
    # görünmez oluyordu — cümle BİLDİRME sanılıp grafa yazılıyordu.
    particle_persons = ("sun", "sün", "sın", "sin", "yum", "yüm", "yım",
                        "yim", "sunuz", "sünüz", "sınız", "siniz", "yuz",
                        "yüz", "yız", "yiz")
    # Soruyu işaretleyen noktalama. Dilin bildirdiği bir şey: İspanyolca "¿",
    # Yunanca ";" kullanır. Kod hangi işaret olduğunu bilmemeli.
    question_marks = ("?",)
    # Soru sözcükleri kapalı bir sınıf: bir dilde birkaç tanedir ve hiçbir
    # sayım "kaç"ın soru sorduğunu göstermez — bu, dil hakkında bildirilmesi
    # gereken bilgi. Liste eksikti ve bedeli ölçüldü: "kartal kaç yaşında
    # yaşar" sorusu BİLDİRME sanılıp grafa `kartal kaç --can--> yaşamak`
    # diye yazılıyordu. Hafızayı kirletmek, bu mimarideki en pahalı hata.
    interrogatives = ("kim", "kimler", "ne", "neler", "kaç", "nasıl",
                      "hangi", "hangisi", "nere", "nerede", "nereye",
                      "nereden", "niye", "niçin", "neden", "kaçıncı")
    copula_suffixes = ("tur", "tır", "dur", "dır", "tür", "tir", "dür", "dir")
    plural_suffixes = ("lar", "ler")
    # Koşul kipi. Bir koşul cümlesi bir İDDİA değildir: "yağmur yağarsa
    # ıslanırsın" yağmur hakkında hiçbir şey söylemez, iki olay arasında bir
    # bağ kurar. Bildirilmediği için ayrıştırıcı onu bildirme sanıyor ve grafa
    # olmayan bir olgu yazıyordu:
    #
    #   > yağmur yağarsa ıslanırsın
    #   öğrendim: yağmur yağarsa ıslanır.
    #   kenar: yağmur --can--> ıslanmak (nesne: yağarsa)
    #
    # Delik eskiydi ama görünmüyordu: 6 fiillik tohum dağarcık `ıslanırsın`ı
    # tanımadığı için cümle KOŞUL olduğundan değil KELİME eksikliğinden
    # reddediliyordu. Dağarcık derleme bağlanınca (18 -> 1559 yüzey) delik
    # açığa çıktı — bir yeteneğin başka bir kusuru gizlemesi.
    conditional_suffixes = ("sa", "se")
    # Aşağıdakiler ÇIKTI değil GİRDİ: sistemin söyledikleri değil, tanıdıkları.
    # `lmm/phrasing.py` "ne denir"i bilir, burası "ne duyulur"u. İkisi de dil ama
    # yönleri ters, ve karıştırılınca ikinci bir dil eklemek iki dosya yerine
    # sekiz dosyaya dokunmak oluyordu. Ölçüldü: bu listeler `cli.py` (7 yerde),
    # `social.py` (40 kalıp) ve `coordination.py`'de duruyordu.
    #
    # Eksiltili sorunun açılışı: "PEKİ YA kartal". Bilgi taşımıyorlar, yalnız
    # cümlenin eksiltili olduğunu işaretliyorlar. `cli.py`'de ikinci ve UYUŞMAYAN
    # bir kopya vardı — "acaba", "hem", "işte" orada yoktu ve o üç eksiltili
    # takip sessizce anlaşılmıyordu.
    openers = ("peki", "ya", "yaa", "hem", "acaba", "bir", "de", "da", "işte",
               "e", "ee", "pekiya")
    # Belirtisiz tanımlık. Kuyruk iki türlü kurulabiliyor — "penguen kuş mu" ve
    # "penguen BİR kuş mu" — ve hangisinin kalıbı olduğunu önceden bilmiyoruz.
    indefinite = "bir"
    # Sohbeti bitiren, yardım isteyen ve durum soran sözler. ASCII eşdeğerleri
    # bilerek duruyor: Türkçe klavyesi olmayan da çıkabilmeli.
    # Söylem arası: cümlenin konusu değil, konu DEĞİŞTİĞİNİN işareti.
    # "bu arada kalp nedir" sorusunda ayrıştırıcı `bu`yu özne sanıyordu, `bu`
    # zamiri de bir önceki konuya çözülüyordu — farklı soruya aynı cevap.
    # Kapalı sınıf: openers gibi dilin verisi, kod değil.
    asides = ("bu arada", "her neyse", "neyse", "aklıma gelmişken")

    exit_words = ("çık", "cik", "exit")
    help_words = ("yardım", "yardim", "help", "?")
    status_words = ("durum", "istatistik")
    # Çoğul göndermede kuyruğa karışan ama içerik taşımayan sözler:
    # "İKİSİ DE kuş mu" -> kuyruk "kuş mu".
    echoes = ("ikisi", "ikiside", "hepsi", "her", "ikisininde", "de", "da")
    # Bilgi taşımayan alışverişler: karşılıkları `lmm/phrasing.py`'de.
    # Anahtarlar `lmm/social.py`'deki kimliklerle eşleşiyor ve orası dil
    # bilmiyor — bu tablo değişince ikinci bir dil ekleniyor, kod değişmeden.
    exchanges = {
        "greeting": ("selam", "merhaba", "günaydın", "iyi akşamlar",
                     "iyi günler", "selamlar", "hey", "alo"),
        "farewell": ("görüşürüz", "hoşça kal", "hoşçakal", "bay",
                     "iyi geceler", "kendine iyi bak", "güle güle"),
        "thanks": ("teşekkür", "teşekkürler", "sağ ol", "sağol", "eyvallah",
                   "minnettarım"),
        "wellbeing": ("nasılsın", "naber", "ne haber", "nasıl gidiyor",
                      "iyi misin", "keyifler nasıl"),
        "identity": ("kimsin", "sen kimsin", "adın ne", "nesin",
                     "kendini tanıt", "kendini tanıtır mısın", "sen nesin"),
        "ability": ("ne yapabilirsin", "neler yapabilirsin",
                    "ne işe yarıyorsun", "nasıl kullanılır", "yardım"),
        # Tepki: bilgi istemeyen, konuşmayı süren sözler. Soru sanılıp
        # "kelimeyi öğret" deniyordu — 12 turluk ölçümde iki tur böyle gitti.
        "backchannel": ("hmm", "hm", "ilginç", "vay", "vay be", "öyle mi",
                        "anladım", "iyiymiş", "güzelmiş", "peki", "tamam",
                        "hadi ya", "cidden mi", "aynen"),
        # Vazgeçme: konuyu kapat, yenisine geç.
        "dismissal": ("boşver", "boş ver", "geç bunu", "fark etmez",
                      "önemli değil", "kapat bu konuyu"),
        # İtiraz açılışı: içerik yok, yalnız "yanlışsın" var. Cevap, doğrusunu
        # İSTEMEK — düzeltme mekanizması zaten var, ona davet ediliyor.
        "challenge": ("yanlış biliyorsun", "yanlışın var", "yanlış bu",
                      "emin misin", "bence yanlış", "doğru mu bu",
                      "yanlış bence"),
    }
    # Yönergeler: davranışı değiştiren buyruklar. LLM'deki system prompt'un
    # buradaki karşılığı — fark şu: prompt her istekte yeniden gönderilir,
    # yönerge bir kez söylenir ve oturum hatırlar. Kapalı sınıf.
    directives = {
        "short": ("kısa cevap ver", "kısa konuş", "kısa cevapla",
                  "daha kısa", "kısa kes", "uzatma"),
        "long": ("uzun anlat", "ayrıntılı anlat", "detaylı anlat",
                 "uzun cevap ver", "ayrıntılı cevap ver"),
        # Cümlenin İÇİNDE geçen uzunluk isteği. Yönerge oturumu değiştirir,
        # bunlar yalnız o cevabı: "penguen hakkında bildiğin her şeyi uzun
        # uzun anlat" bir yönerge değil, uzun cevap isteyen bir SORU.
        "long_here": ("uzun uzun", "her şeyi", "hepsini", "ayrıntılı olarak",
                      "detaylıca", "uzunca"),
        "short_here": ("kısaca", "tek cümleyle", "özetle", "kısa kes"),
        # ANLATMA İSTEĞİ: soru işareti yok ama cevap bekleniyor. "kuş hakkında
        # her şeyi anlat" bir bildirme değil; bunu bildirme sayıp susmak,
        # sorulanı duymamaktı.
        "asking_here": ("anlat", "bahset", "söyle", "açıkla", "tanıt"),
        "plain": ("kaynak gösterme", "kaynaksız konuş", "kaynak yazma",
                  "kaynakları gizle"),
        "cited": ("kaynak göster", "kaynaklı konuş", "kaynakları göster"),
    }
    # Açılış sözcükleri: cümlenin başında durup hiçbir şey eklemeyenler.
    # "acaba penguen uçabilir mi" ile "penguen uçabilir mi" aynı soru.
    # Bir kavram öbeğinin İÇİNE giremeyecek kelimeler. Hepsi kapalı sınıf ve
    # hepsi aynı işi görüyor: kavramı nitelemiyorlar, cümlenin yapısını
    # kuruyorlar. Bu bilgi DİLE aittir ve burada durur — `grammar.py` hangi
    # dile baktığını bilmemeli.
    #
    # Liste `lmm/clauses.py`'den buraya taşındı ve yön tersine döndü: eskiden
    # dil, bölme organından ithal ediyordu. Bağlaç bir dil olgusudur, bölme
    # ise bir işlem; ikincisi birincisini tanımlayamaz.
    joiners = ("ve", "veya", "ya da", "ancak", "fakat", "ama", "çünkü",
               "ayrıca", "yani", "oysa", "hâlbuki", "halbuki", "lakin",
               "ne var ki")
    # Türkçe yan cümlesini bağlaçla değil EKLE kurar. Ölçüldü: cümlelerin
    # %74,4'ü bunlardan en az birini taşıyor. Her biri bir yan cümlenin SONUNU
    # işaretler: "kaydı alıp raporlar" iki yüklemdir.
    subordinator_suffixes = (
        ("ip", "ıp", "up", "üp"),               # gelip, alıp
        ("erek", "arak"),                       # koşarak
        ("ince", "ınca", "unca", "ünce"),       # gelince
        ("madan", "meden"),                     # görmeden
        ("dıkça", "dikçe", "dukça", "dükçe"),   # gördükçe
        ("ken",),                               # bakarken
    )
    # Yan cümle ekleri fiili çekimsiz bırakır ve bölerken çekimi geri kurmak
    # gerekir — "-ken" hariç: "bakarken" ekini yitirince zaten "bakar"dır.
    # Hangi ekin böyle olduğunu ve altındaki gövdenin neye benzemesi
    # gerektiğini dil söyler; bölme organı yalnızca uygular.
    finite_subordinators = {"ken": ("ar", "er", "ır", "ir")}
    # Ayrı yazılan pekiştirme parçacıkları. Ek hâlleri bitişik yazıldığı için
    # karışmıyor: "kartalda" bulunma, "kartal da" pekiştirme.
    # Görüş isteyen açılışlar. Kapalı sınıf: bir dilde birkaç tanedir ve
    # hepsi aynı şeyi yapar — cümleyi bilgi sorusundan kanaat sorusuna çevirir.
    opinion_marks = ("sence", "bence", "sizce", "bizce", "kanaatince")
    clitics = ("da", "de", "dahi", "bile")
    # Çoğul gönderme: "ikisi de", "onlar", "bu ikisi" — sohbette az önce
    # geçen kavramlara işaret eder. Tekil zamirler (`o`, `bu`) zaten
    # `pronouns` içinde; bunlar İKİ şeye birden gönderdikleri için ayrı.
    plural_pronouns = ("ikisi", "ikiside", "onlar", "bunlar", "şunlar",
                       "hepsi", "her", "üçü")
    # Muhatap zamirleri. Bir soruda "bana"/"bize" cümlenin KONUSU değil,
    # kime söylendiğidir — ve LMM her zaman muhataptır. "bunu bana anlatır
    # mısın" ile "bunu anlatır mısın" aynı şeyi soruyor.
    addressees = ("bana", "bize", "sen", "siz", "sana", "size", "senin",
                  "sizin", "benim", "bizim", "ben", "biz")
    # Aynı şeyi soran kapalı sınıf sözcükler. Kalıpta biri yazılıyor, üçü de
    # eşleşiyor — "neden" için kalıp yazıp "niçin" için yazmamak, aynı soruyu
    # iki kez yazmak demekti. Hangi sözcüğün hangisiyle aynı şeyi sorduğu
    # DİLE ait bilgidir ve sayımla bulunamaz.
    groups = (
        ("neden", "niçin", "niye", "niden"),
        ("nasıl", "ne şekilde"),
        ("kim", "kimler"),
        ("ne", "neler"),
    )
    # Tek başına söylenen bir soru sözcüğü hangi soruyu sürdürür. "penguen
    # nedir" dedikten sonra yalnızca "neden" demek, aynı kavram için bir SEBEP
    # sorusudur. Hangi sözcüğün hangi soruyu sorduğu dile ait bilgi: `groups`
    # zaten hangilerinin aynı şeyi sorduğunu bildiriyor, bu da onların NEYİ
    # sorduğunu. `lmm/cli.py`'de sabit duruyordu ve orası dilin yaşadığı yer
    # değil — `affirmations` ile aynı gerekçeyle buraya taşındı.
    bare_questions = {"neden": ASK_WHY, "niye": ASK_WHY, "niçin": ASK_WHY,
                      "nasıl": ASK_PROPERTIES, "kim": ASK_WHO,
                      "kimler": ASK_WHO}
    denials = ("değil", "değildir", "yok", "yoktur")
    # Bir SORUYA verilen onay ve ret. Cümle içindeki olumsuzluktan (`denials`)
    # ayrı: "yok" ikisinde de geçiyor ama biri yüklemi olumsuzluyor, öteki
    # sorulan şeyi reddediyor. Bunlar `lmm/cli.py`'de sabit duruyordu ve orası
    # dilin yaşadığı yer değil — ikinci bir dil eklenince oraya dokunmak
    # gerekirdi. Dil burada yaşar.
    affirmations = ("evet", "e", "ee", "aynen", "tabii", "tabi", "olur",
                    "öğren", "kaydet")
    refusals = ("hayır", "hayir", "yok", "olmaz", "istemiyorum", "boşver")
    # DÜZELTME açılışı: "hayır penguen uçamaz". Ret sözcüğüyle aynı kelimeler
    # ama işlevi başka — tek başına duruyorsa ret, arkasından cümle geliyorsa
    # düzeltmedir. Bildirilmediği için ayrıştırıcı "hayır"ı KAVRAM sanıyordu:
    #
    #   > hayır penguen uçamaz
    #   öğrendim: hayır penguen uçamaz. bunu hiç öğrenmedim: HAYIR nedir?
    #
    # Bir sistemin yanlışını düzeltmek, ona bir şey öğretmek kadar temel.
    corrections = ("hayır", "hayir", "yanlış", "yanlıs", "değil", "aslında",
                   "olmaz", "doğru değil")
    # UNUTMA: kalıcı hafızadan silmek. Bir LLM'in ağırlıklarından bilgi
    # silinemez; burada silinebiliyor ve bunun sohbetten erişilebilir olması
    # gerekiyor — dosyayı elle açmak bir arayüz değildir.
    forget_words = ("unut", "sil", "kaldır", "unutmalısın", "sil bunu")
    postpositions = ("ile", "ila", "karşı", "göre", "kadar", "gibi", "için",
                     "rağmen", "beri", "dolayı")
    intensifiers = ("çok", "daha", "en", "pek", "oldukça", "gayet", "epey",
                    "hayli", "az", "biraz")
    correlatives = ("hem", "ya", "gerek", "kah", "kâh", "ister")
    # Endings that mark a word as playing its own part in the sentence rather
    # than belonging to the noun beside it: "serçeden" is a comparison, not half
    # of a compound. Discovery finds these families; until it supplies them,
    # they are listed.
    oblique_suffixes = ("den", "dan", "ten", "tan", "yle", "yla", "ile")
    # A case ending says what a second concept is doing in the sentence, so the
    # role is read off the word instead of guessed from where it sits.
    case_roles = (
        (("den", "dan", "ten", "tan"), SOURCE),
        (("de", "da", "te", "ta"), PLACE),
    )
    genitive_suffixes = ("nın", "nin", "nun", "nün", "ın", "in", "un", "ün")
    possessive_suffixes = ("sı", "si", "su", "sü", "ı", "i", "u", "ü")
    pronouns = ("o", "onu", "onun", "bu", "bunu", "şu", "şunu")
    # Qualitative, never numeric. Which words mark how much of a kind is meant.
    quantifiers = {"bütün": ALL, "tüm": ALL, "her": ALL, "bilcümle": ALL,
                   "çoğu": MOST, "ekseri": MOST,
                   "bazı": SOME, "kimi": SOME, "birtakım": SOME,
                   "hiçbir": NO, "hiç": NO}

    # A consonant softens before the suffix: balık + ın is "balığın". Undoing
    # that is part of reading the possessor back out.
    softened = {"ğ": "k", "b": "p", "c": "ç", "d": "t"}

    # --- Aşağısı `lmm/verbs.py` ile `lmm/frames.py`'den taşındı ---------------
    #
    # O iki organ hangi dile baktığını bilmemeli, yoksa ikinci bir dil ikinci
    # bir organ demek. Ekler burada durur, orası `getattr` ile sorar ve dili
    # olmayan bir organ sessizce boş küme alır — dosyanın geri kalanındaki
    # desenin aynısı.

    vowels = "aeıioöuü"
    back_vowels = "aıou"        # kalın ünlü: ek "-mak/-maz", ince olan "-mek/-mez"
    infinitive_suffixes = ("mak", "mek")

    # Geniş zaman. Fiil keşfi bu iki listeye dayanıyor: bir kök hem olumlu hem
    # olumsuz çekimiyle metinde geçiyorsa fiildir. Bir dil eklemek bu ikisini
    # yazmaktır, `lmm/verbs.py`'deki kodu değiştirmek değil.
    aorist_suffixes = ("ar", "er", "ır", "ir", "ur", "ür", "r")
    aorist_negative_suffixes = ("maz", "mez")

    # Sıfat-fiil ekleri: fiilden sıfat yapar. "olan", "gereken", "olduğu" bir
    # kavram değil, bir yüklemin sıfatlaşmış hâli.
    participle_suffixes = ("an", "en", "dığı", "diği", "duğu", "düğü", "tığı",
                           "tiği", "acak", "ecek", "esi", "ası", "mış", "miş",
                           "muş", "müş")

    # İsim testi: bir kelime durum eki alıyorsa isimdir. Aile aile duruyor
    # çünkü ölçüt "kaç ayrı AİLE görüldü" — bir tek çekim tesadüf olabilir.
    case_families = (("i", "ı", "u", "ü"), ("e", "a"), ("de", "da", "te", "ta"),
                     ("den", "dan", "ten", "tan"), ("in", "ın", "un", "ün"),
                     ("ler", "lar"))

    # Derecelendiriciler. Türkçe'de yalnızca sıfat ve zarf derecelenir: "daha
    # zor" olur, "daha banka" olmaz.
    graders = ("daha", "en", "çok", "pek", "oldukça")

    # Yapı kelimeleri: dilbilgisi taşırlar, kavram değildirler. Asıl eleme
    # sıklıkla yapılıyor; bu liste yalnız sayımın kaçırdıklarını kapatıyor.

    # Durum ekleri: bir kelimenin cümledeki rolünü söyleyen şey. Sıra önemli —
    # uzun ek önce denenmeli, yoksa "evden" içinde "de" bulunur. Etiketler
    # `lmm/frames.py`'nin kimlik dizgileri; dil onları yalnızca EŞLİYOR.
    case_suffixes = (
        (("nden", "ndan", "den", "dan", "ten", "tan"), ABLATIVE),
        (("nde", "nda", "de", "da", "te", "ta"), LOCATIVE),
        (("nın", "nin", "nun", "nün", "ın", "in", "un", "ün"), GENITIVE),
        (("yle", "yla", "ile"), INSTRUMENTAL),
        (("ye", "ya", "e", "a"), DATIVE),
        (("yı", "yi", "yu", "yü", "nı", "ni", "nu", "nü",
          "ı", "i", "u", "ü"), ACCUSATIVE),
    )

    # Yüklem ekleri. Ölçüldü: ilk altısı yüklemlerin ~%75'ini kapsıyor.
    # Kip ekleri koşaçtan ÖNCE denenmeli: "yapılmalıdır" koşaç kuralına
    # takılıp "yapılmalı" + koşaç diye okunuyordu, oysa gereklilik kipidir.
    predicate_suffixes = (
        (("malıdır", "melidir", "malı", "meli"), OBLIGATION),
        (("sa", "se", "ysa", "yse"), CONDITION),
        (("maktadır", "mektedir"), PROGRESSIVE),
        (("mıştır", "miştir", "muştur", "müştür",
          "mıştı", "mişti", "muştu", "müştü"), PERFECT),
        (("abilir", "ebilir"), ABILITY),
        (("tur", "tır", "dur", "dır", "tür", "tir", "dür", "dir"), COPULA),
        (("dı", "di", "du", "dü", "tı", "ti", "tu", "tü"), PAST),
        (("ar", "er", "ır", "ir", "ur", "ür", "r"), AORIST),
    )
    # Koşaç, zaman ekinin üstüne de gelir: "uçacak-tır", "uçmakta-dır".
    copula_on_tense = ("dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür")

    # --- Aşağısı `lmm/lexicon.py`'den taşındı --------------------------------
    #
    # Sözlük bir kelimeyi tanırken çekim katmanlarını soyuyor. Soyma İŞLEMİ
    # dilden bağımsız — en dıştan içe, katman katman — ama katmanların neye
    # benzediği bu dile ait ve orada yazılıydı.

    # Geniş zaman dışındaki çekimler. Olumsuzluk ayrı bir ek olarak okunuyor;
    # ikinci sütun ekin KENDİSİNİN olumlu olup olmadığını söylüyor.
    tense_suffixes = (
        (("ıyor", "iyor", "uyor", "üyor", "yor"), True),     # şimdiki zaman
        (("acak", "ecek", "acağı", "eceği"), True),          # gelecek
        (("mış", "miş", "muş", "müş"), True),                # duyulan geçmiş
        (("dı", "di", "du", "dü", "tı", "ti", "tu", "tü"), True),  # görülen geçmiş
    )
    # Olumsuzluk ve yetersizlik: "uçmuyor", "uçamıyor". Ünlü uyumu yüzünden
    # ek ünlüsü değişiyor, o yüzden hepsi sayılıyor — kapalı bir sınıf.
    negation_marks = ("amı", "emi", "amu", "emü", "mı", "mi", "mu", "mü",
                      "ma", "me")
    # Kişi ekleri. Fiil çekiminin en dış katmanı ve soyulmadan sözlük kelimeyi
    # tanımıyordu: "söyleyebilirsin", "biliyorsun", "konuşuyoruz" hepsi
    # "bilmiyorum" cevabı alıyordu. Kapalı sınıf: bir dilde altı kişi vardır
    # ve ünlü uyumuyla çoğalırlar.
    person_suffixes = ("sınız", "siniz", "sunuz", "sünüz",
                       "ım", "im", "um", "üm", "yım", "yim", "yum", "yüm",
                       "sın", "sin", "sun", "sün",
                       "ız", "iz", "uz", "üz", "yız", "yiz", "yuz", "yüz",
                       "lar", "ler")
    # Türkçe yeteneği ayrı bir ekle işaretler: "penguen yüzebilir". Her fiil
    # için üçüncü bir biçim saklamak yerine ek soyuluyor.
    ability_suffixes = ("ebilir", "abilir")

    # Bu dilin harfleri. Metinden kelime çıkarırken gerekiyor ve İngiliz
    # alfabesi Türkçe'yi kesiyor: "çalışır" ile "calisir" ayrı kelimeler.
    letters = "a-zçğıöşü"

    # --- Aşağısı `lmm/intuition.py` ile `lmm/inquiry.py`'den taşındı ---------
    #
    # İkisi de motor: biri cümleyi okuyor, diğeri kaynak metni. Hangi dile
    # baktıklarını bilmemeleri gerekirken içlerinde elle yazılmış Türkçe ek ve
    # kalıp duruyordu. İşlem orada kaldı, veri buraya geldi.

    # Küçük harfe indirirken hangi harf hangisine düşer. Python "İ"yi "i" artı
    # birleşen noktaya çeviriyor, "I"yı da "i"ye — ikisi de bu dilde yanlış ve
    # bedeli ölçülmüştü: "İnsanlar" ile "insan" ayrı kavram oluyordu. Hangi
    # çiftlerin düzeltileceği DİLE ait; indirme işlemi değil.
    lowercase_pairs = (("İ", "i"), ("I", "ı"))

    # Bir çekimin en DIŞ katmanları: kişi eki, çoğul, koşaç. Bir kelimenin
    # bilinen bir fiil olup olmadığına bakarken soyuluyorlar.
    #
    # `person_suffixes`in aynısı DEĞİL ve bilerek: buradaki liste kaynaşma
    # ünsüzlü biçimleri ("yım", "yiz") almıyor, buna karşılık koşacın yalnız
    # d'li dördünü ("dır", "dir", "dur", "dür") alıyor. İki listeyi birleştirmek
    # soyulan katman kümesini genişletir, yani davranışı değiştirir; taşımak
    # onu değiştirmemeli.
    outer_suffixes = ("sınız", "siniz", "sunuz", "sünüz", "sın", "sin", "sun",
                      "sün", "ım", "im", "um", "üm", "ız", "iz", "uz", "üz",
                      "lar", "ler", "dır", "dir", "dur", "dür")

    # Sözlüğün doğduğu andaki çekirdek dağarcık: yüzey -> (mastar, olumlu_mu).
    #
    # Derlem varken üstüne binlerce fiil biniyor (bkz. `lexicon.seed_verbs`);
    # yokken yalnız bu kalıyor. Silinemez oluşunun sebebi ölçülebilir: derlem
    # olumsuzu kuralla üretiyor ("uçmaz"), oysa bu dilde yetersizlik ayrı bir
    # ektir ("uçamaz") ve sistemin KONUŞTUĞU biçim odur. Sıra bu yüzden önemli —
    # sözlük ilk gördüğü yüzeyi saklıyor, o da buradan gelmeli.
    core_verbs = {
        "uçar": ("uçmak", True), "uçamaz": ("uçmak", False),
        "yüzer": ("yüzmek", True), "yüzemez": ("yüzmek", False),
        "koşar": ("koşmak", True), "koşamaz": ("koşmak", False),
        "okur": ("okumak", True), "okuyamaz": ("okumak", False),
        "içer": ("içmek", True), "içemez": ("içmek", False),
        "konuşur": ("konuşmak", True), "konuşamaz": ("konuşmak", False),
        # Bu dil "yapmaz" ile "yapamaz"ı ayırır: ikisi de duyuluyor ve ikisi de
        # aynı ilişkiye düşüyor.
        "uçmaz": ("uçmak", False), "yüzmez": ("yüzmek", False),
        "koşmaz": ("koşmak", False), "okumaz": ("okumak", False),
        "içmez": ("içmek", False), "konuşmaz": ("konuşmak", False),
    }

    # Ansiklopedik tanım kalıpları. Motorun yaptığı iş "ilk cümlede bir tanım
    # kipi var mı" diye sormak; kipin neye BENZEDİĞİ bu dile ait.
    #
    # `konu` ve `tur` grup adları iki taraf arasındaki sözleşme: hangi dil
    # olursa olsun kalıbı yazan bu iki grubu adlandırır, okuyan da onları sorar.
    #
    # Birincisi koşaçsız tanım — Türkçe koşacı düşürebiliyor ve ansiklopedi
    # bunu sık yapıyor: "Vaşak, ... hayvan türlerinin ortak adı." cümlesinde
    # yüklem eki yok, o yüzden çerçeve okuyucusu hiçbir şey bulamıyordu.
    # İkincisi "... X türlerinin ortak adı" — en sık ikinci kalıp, içinde tür
    # bilgisi var.
    definition_shapes = (
        re.compile(r"^(?P<konu>[^,.(]{2,40})\s*[,(].{0,200}?"
                   r"\bbir\s+(?P<tur>[a-zçğıöşü]+)"
                   r"(?:dır|dir|dur|dür|tır|tir|tur|tür)?\b", re.I | re.S),
        re.compile(r"^(?P<konu>[^,.(]{2,40})\s*[,(].{0,200}?"
                   r"\b(?P<tur>[a-zçğıöşü]+)\s+(?:tür|cins|familya)\w*\s+"
                   r"(?:ortak|genel|bilimsel)\s+adı", re.I | re.S),
    )

    # Kavram ya da hedef OLAMAYACAK kelimeler. Hepsi kapalı sınıf; grafta

    def genitive_readings(self, word):
        """Every way this word could be a possessor, longest stem first."""
        found = []
        for suffix in self.genitive_suffixes:
            if not (word.endswith(suffix) and len(word) > len(suffix) + 1):
                continue
            stem = word[: -len(suffix)]
            if suffix.startswith("n") != (stem[-1] in "aeıioöuü"):
                continue
            found.append(stem)
            hardened = stem[:-1] + self.softened.get(stem[-1], stem[-1])
            if hardened != stem:
                found.append(hardened)
        return sorted(set(found), key=len, reverse=True)

    accusative_suffixes = ("yı", "yi", "yu", "yü", "ı", "i", "u", "ü")

    def strip_accusative(self, word):
        return self._strip(word, self.accusative_suffixes)

    def strip_genitive(self, word, known=()):
        """"kuşun" -> "kuş", "balığın" -> "balık".

        The -nIn form follows a vowel and the -In form a consonant, so the
        candidates are checked against that rather than taken longest-first:
        "penguenin" is penguen + in, not pengue + nin.
        """
        candidates = []
        for suffix in self.genitive_suffixes:
            if not (word.endswith(suffix) and len(word) > len(suffix) + 1):
                continue
            stem = word[: -len(suffix)]
            after_vowel = stem[-1] in "aeıioöuü"
            if suffix.startswith("n") != after_vowel:
                continue                # the wrong form for what precedes it
            candidates.append(stem)
            hardened = stem[:-1] + self.softened.get(stem[-1], stem[-1])
            if hardened != stem:
                candidates.append(hardened)
        # "insanın" splits two ways and both obey the rule — insa+nın and
        # insan+ın — so only knowing the word decides. Where nothing decides,
        # nothing is returned: guessing here would write a concept that does
        # not exist and never say so.
        # Kopya YALNIZ gerekiyorsa. Çağıran artık küme geçiyor ve her çağrıda
        # 16 bin kavramı kopyalamak profilde 122 bin çağrıda 11,4 saniye
        # tutuyordu — üyelik sınaması için kopyaya gerek yok.
        if not isinstance(known, (set, frozenset, dict)):
            known = set(known)
        for candidate in candidates:
            if candidate in known:
                return candidate
        return candidates[0] if len(candidates) == 1 else word

    def role_of(self, word):
        """(stem, role) when the word carries a case ending, else (word, None)."""
        for suffixes, role in self.case_roles:
            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 1:
                    return word[: -len(suffix)], role
        return word, None

    def is_oblique(self, word):
        """Does this word carry a case ending, so it stands on its own?

        Every ending that marks a role counts, not just the ones listed for
        instruments — a locative slipped into a noun phrase once and made
        "penguen kutupta" a concept.
        """
        if self._has(word, self.oblique_suffixes):
            return True
        return any(self._has(word, suffixes) for suffixes, _ in self.case_roles)

    def has_copula(self, word):
        return self._has(word, self.copula_suffixes)

    def strip_copula(self, word):
        return self._strip(word, self.copula_suffixes)

    def strip_plural(self, word):
        return self._strip(word, self.plural_suffixes)

    @staticmethod
    def _has(word, suffixes):
        return any(word.endswith(s) and len(word) > len(s) + 1 for s in suffixes)

    @staticmethod
    def _strip(word, suffixes):
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[: -len(suffix)]
        return word


# Order matters: "kim uçar" has the shape of "kuşlar uçar", so the question has
# to be tried before the lesson.
# Elle yazılmış 100 kalıp SİLİNDİ (5 Ağustos 2026). Kalıp artık yalnız
# ÖĞRENİLMİŞ olandan gelir: model dosyasındaki kalıplar (sohbette öğrenilen,
# şu an 103) ve eğitilmiş okuyucular (core/intent.py, core/reader.py).
#
# Silmeden önce ölçüldü: cevaplamada elle kalıpların payı yalnız 1,5 puandı
# (%50,5 -> %49,0) — öğrenilmiş yol taşıyordu. Öğretme kalıplara bağlıydı ve
# onun öğrenilmiş verisi hazır: data/tr-niyet-tumu.txt içinde 58.399 gerçek
# TEACH örneği. Eğitimi: scripts/niyet_egitim.py --veri data/tr-niyet-tumu.txt
PATTERNS = []


# One example each is enough for discovery to find the whole system: what a
# suffix *means* cannot be read off its shape, but everything else can.
ANCHORS = {"çoğul": ("kuş", "kuşlar"), "koşaç": ("kuş", "kuştur")}


def turkish(words=None, known=(), meanings=None):
    """The grammar. Given words, its suffix rules are discovered rather than read.

    Discovery needs enough words to see a pattern; below that it falls through
    to the declared lists, so a thin memory degrades instead of breaking.
    """
    morphology = TurkishMorphology()
    if words:
        from lmm.discovered import DiscoveredMorphology
        morphology = DiscoveredMorphology(morphology, words, ANCHORS)
    return Grammar(PATTERNS, morphology, known, meanings)
