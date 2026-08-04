"""Bir olguyu SÖYLEMENİN yollarını, insanların yazdığı cümlelerden öğrenir.

Bu dosya, projenin en büyük elle yazılmışlık boşluğunu kapatmak için var.

Okuma yarısı öğreniyor: kalıplar sohbetten, fiiller derlemden sayarak,
kavramlar metinden, gömme sayımla. Konuşma yarısı öğrenmiyordu — cümleyi nasıl
kuracağını programcı yazmıştı:

    "{kavram} bir {hedef}dır"      "Ayrıca ..."      "... olduğu için ..."

Bir dil modelinde bunların hiçbiri yok. O, tanım cümlesinin nasıl kurulduğunu
kimse söylemediği hâlde milyonlarca tanım görerek çıkardı; denetim sinyali
verinin kendisiydi (sonraki kelime). Bizim aynı işi yapacak sinyalimiz zaten
kurulu ve ölçülü: GERİ OKUMA. Bir cümle kurulur, sistemin kendi dilbilgisiyle
okunur, aynı olgu çıkıyor mu diye bakılır.

    derlemde   "Kartal, yırtıcı bir kuş türüdür."
    okundu     (kartal, type, kuş)
    ÖĞREN      "{kavram} bir {hedef}-DIr"  -> bu olguyu söylemenin bir yolu
    konuş      yolu seç, doldur, geri oku, olgu tuttu mu

Elle yazılmış tek şey kalmıyor: hangi kelimelerin kullanılacağını da, sıranın
ne olacağını da metin söylüyor. Şablonun kendisi veri — model dosyasında
saklanıyor, sohbette büyüyor, ikinci bir dilde sıfırdan öğreniliyor.

SINIR — ölçülmeden kabul edilmiyor. Bir söyleyiş ancak geri okunduğunda aynı
olguyu veriyorsa saklanıyor. Uydurma riski artmıyor çünkü söylenen olgu yine
graftan geliyor; değişen yalnız onu kuran cümle.

ÖLÇÜM — 217.002 (cümle, olgu) çifti, `models/graph/birlesik.lmm` grafından
ilişki başına 50 rastgele olgu, geri okuma sınavı. İlk sürüm cümlenin
TAMAMINI şablon sayıyordu ve sonuç sıfırdı:

    25.442 şablon adayı            -> 22'si geri okundu, hepsi tek tanıklı,
                                      hepsi property (o kalıp fazladan
                                      kelimeye aldırmıyor)
    type / has / can / cannot      -> hiçbir aday geri okunamadı

Sebebi iki ölçümle bulundu ve ikisi de bu dosyada karşılandı:

    KIRPMA   80 örnekte geri okuma 64 kez UNKNOWN verdi, 16 kez kavramı
             kaydırdı ("şehir ekiplerinden fiorentino ..." -> kavram=şehir).
             İkisinin de sebebi yuvadan ÖNCEKİ kelimeler: Türkçe'de olguyu
             yüklem taşır, baştaki giriş o cümlenin konusudur. Şablon artık
             ilk yuvadan başlıyor.
    EK       "Kartal bir kuştur" cümlesinde hedef kelime `kuştur`; yuva eki
             de yutuyordu ve "fiorentino bir şehir" çıkıyordu. Yuva artık
             yuttuğu eki NOT ediyor ({hedef}-DIr) ve giydirirken aynı eki
             biçimbilgisi organı yeni kelime için üretiyor.

İkisinden sonra, ilişki başına en çok tanıklı şablon şu notu aldı:

    type     {kavram} bir {hedef}-DIr     49/50   (elle yazılmışın tavanı da 49/50)
    property {kavram} {hedef}-DIr         50/50   (tavan 50/50)
    can      {kavram} {hedef}-Ar          50/50   (tavan 50/50)
    cannot   {kavram} {hedef}-mAz         50/50   (tavan 50/50)
    has      —                             0      (aşağıda)

`has` ÇIKMADI ve sebebi şablonda değil: derlem "vardır" diyor (325 tanık),
dilbilgisi ise iyelik kalıbını yalnız "var" ile okuyor. Elle kurulup denenen
"{kavram}-In {hedef} var" 50/50 alıyor ama derlemde SIFIR tanığı var. Yani
bu ilişkiyi öğrenmenin önündeki engel süzgeç değil, ansiklopedi dilinin
"var" dememesi.
"""
import collections

MARK_CONCEPT = "{kavram}"
MARK_TARGET = "{hedef}"

# Bir söyleyişin saklanması için kaç ayrı cümlede görülmesi gerekiyor. Tek
# cümleden öğrenmek, o cümlenin tuhaflığını dil sanmaktır: "Kartal, Ahmet'in
# dediğine göre bir kuştur" da bir tanım cümlesidir ama söyleyiş değildir.
LEAST_WITNESSES = 2

# Şablonun (yani ilk yuvadan sonun) en çok kaç belirteci olabilir. Uzun
# şablon, olgunun kendisinden çok o cümlenin bağlamını taşır ve başka bir
# olguya giydirilince saçmalar.
LONGEST = 5

# Şablon çıkarılırken bakılan en uzun cümle. Kırpma zaten kısaltıyor; bu sınır
# yalnız 217 bin satırlık taramayı ucuz tutmak için var ve ölçüm bu sınırla
# yapıldı.
LONGEST_SENTENCE = 30


def endings(morphology=None):
    """Yuvanın yutabileceği ekler: ad -> çekimli hâli üreten kurallar.

    Ekleri LİSTEDEN ÇÖZÜMLEMİYORUZ, ÜRETİP KARŞILAŞTIRIYORUZ: organa "kuş"un
    çekimli hâllerini ürettiriyoruz ve cümledeki kelimeyle birebir eşleşen
    varsa yuva o eki not ediyor. Böylece giydirme tanım gereği doğru — şablon
    yalnız "burada koşaç vardı" diyor, o koşacı yeni kelime için üreten yine
    dilbilgisi organı.

    Dile ait olan tek yer burası ve dışarıdan verilebilir; şablonun kendisi
    hiçbir dil bilgisi taşımıyor.
    """
    if morphology is not None:
        return morphology
    from lmm import phrasing
    return (
        ("-DIr", lambda word: word + phrasing.copula(word)),
        ("-In", phrasing.genitive),
        ("-I", phrasing.possessed),
        ("-Ar", lambda word: phrasing.verb_form(word, True)),
        ("-mAz", lambda word: phrasing.verb_form(word, False)),
    )


def _ending_of(word, name, rules):
    """Bu kelime, o adın hangi ekli hâli? Çıplak hâliyse "", hiçbiriyse None."""
    word = word.strip(",.;:!?\"'()").lower()
    name = str(name).lower()
    if not name:
        return None
    if word == name:
        return ""
    # Ek ÜRETMEK pahalı ve 217 bin cümlede her kelime için beş kez yapılıyordu:
    # öğrenme 320 saniye sürüyordu. Ek kelimenin BAŞINI değiştirmediği için
    # ilk iki harf tutmuyorsa üretmeye hiç girmiyoruz — aynı şablonlar,
    # 320 sn yerine 41 sn. (İki harf, bir değil: "uçmak" -> "uçar" gövdenin
    # üçüncü harfini yiyor, ilk ikisini hiçbir ek yemiyor.)
    if word[:2] != name[:2]:
        return None
    for label, make in rules:
        try:
            if make(name) == word:
                return label
        except Exception:      # üretilemeyen ek bir eşleşme değildir
            continue
    return None


def _is_slot(word):
    return word.startswith(MARK_CONCEPT) or word.startswith(MARK_TARGET)


def template_of(sentence, concept, target, morphology=None):
    """Cümleyi, kavram ve hedefi yuvalara çevirerek şablona döndürür.

    İkisi de bulunamazsa None — bulunamayan yuva, doldurulamayan şablon
    demektir ve doldurulamayan şablon başka bir olguyu söyleyemez.

    Dönen şablon cümlenin tamamı değil, İLK YUVADAN SONUNA kalan parçadır.
    Ölçüldü: tamamıyla 25.442 adayın 22'si geri okunabildi ve hiçbiri type,
    can, cannot değildi; kırpmayla ilişki başına 49-50/50 alan şablonlar
    çıktı. Atılan baş kısım cümlenin konusudur, olgunun söylenişi değil.
    """
    rules = endings(morphology)
    words = sentence.split()
    if not (2 <= len(words) <= LONGEST_SENTENCE):
        return None
    found_concept = found_target = False
    shaped = []
    for word in words:
        mark = None
        if not found_concept:
            ending = _ending_of(word, concept, rules)
            if ending is not None:
                mark = MARK_CONCEPT + ending
                found_concept = True
        if mark is None and target and not found_target:
            ending = _ending_of(word, str(target), rules)
            if ending is not None:
                mark = MARK_TARGET + ending
                found_target = True
        shaped.append(mark if mark else word)
    if not found_concept:
        return None
    if target and not found_target:
        return None
    first = next(i for i, word in enumerate(shaped) if _is_slot(word))
    kept = shaped[first:]
    if not (2 <= len(kept) <= LONGEST):
        return None
    return " ".join(kept)


def _is_general(template, counts):
    """Şablonun yuva dışı kelimeleri DİLBİLGİSİ mi, yoksa o cümlenin konusu mu.

    İlk ölçümde şablonlar öğrenildi ama giydirilemez çıktı:

        "Gökyüzünde {kavram} {hedef} yönünde bulunur."     34 cümlede
        "Lig ekiplerinden {kavram} {hedef} giymektedir."    10 cümlede

    Hepsi gerçek cümle, hepsi doğru okunmuş, ve hiçbiri başka bir olguyu
    söyleyemez — çünkü yuvaların arasında kalan kelimeler o cümlenin konusu.

    Bir dil modelinde bu ayrım kendiliğinden oluşuyor: milyonlarca örnek
    üzerinde ortak olan (dilbilgisi) hayatta kalıyor, tek örneğe ait olan
    (konu) siliniyor. Bizde o ayrımı yapan organ zaten var ve ölçülü —
    sıklık SIRASI. Bir dilin en sık kelimeleri her derlemde aynı türdendir:
    bağlaç, edat, koşaç. Kalanı konudur.

    ŞART %100 KALDI ve bu bir ölçüm sonucu. Gevşetmek denendi: 217 bin çiftte
    iki tanıklı 315 aday üzerinde yüzde eşiği 1,0'dan 0,0'a indirildi
    (1,0 -> 0 şablon, 0,6 -> 6, 0,5 -> 24, 0,34 -> 93, 0,0 -> 315) ve geri
    okuma sınavını geçen sayısı hiç değişmedi: 315 adayın 1'i, o da yanlış
    okuma ("ōkuninushi büyük Milletvekilliği yapmıştır"). Yani sıfırın sebebi
    eşiğin sertliği değildi, havuzun kendisiydi; havuz kırpma ve ek işaretiyle
    düzelince %100 eşiği geri okunan şablonların HEPSİNİ geçirdi ve konuya
    yapışık olanları ("{kavram} toplulukları {hedef}-Ar", 3 tanık) eledi.
    """
    from lmm.verbs import is_structural
    if not counts:
        return True
    for word in template.split():
        if _is_slot(word):
            continue
        plain = word.strip(",.;:!?\"'()").lower()
        if not plain:
            continue
        if not is_structural(plain, counts):
            return False
    return True


def learn(rows, morphology=None, least=LEAST_WITNESSES, counts=None):
    """(cümle, olgu) çiftlerinden söyleyişleri çıkarır.

    Dönen: {ilişki: [(şablon, kaç cümlede görüldü)]}, çoktan aza.

    Sayım TANIK sayıyor, geçiş değil: aynı şablonu iki AYRI cümlede görmek,
    bir cümleyi iki kez görmekten başka bir şeydir.
    """
    seen = collections.defaultdict(collections.Counter)
    for sentence, concept, relation, target in rows:
        if not (sentence and concept and relation):
            continue
        shaped = template_of(sentence, concept, target, morphology)
        if shaped:
            seen[relation][shaped] += 1
    found = {}
    for relation, counted in seen.items():
        held = [(shape, count) for shape, count in counted.most_common()
                if count >= least and _is_general(shape, counts)]
        if held:
            found[relation] = held
    return found


def say(template, concept, target=None, morphology=None):
    """Şablonu doldurur. Hedef gerekiyorsa ve yoksa None.

    Yuvanın not ettiği eki burada dilbilgisi organı üretiyor: "{hedef}-DIr" +
    "şehir" -> "şehirdir", "kuş" -> "kuştur". Ünlü uyumunu şablon bilmiyor,
    bilmesi de gerekmiyor.
    """
    rules = dict(endings(morphology))
    said = []
    for word in template.split():
        for mark, value in ((MARK_CONCEPT, concept), (MARK_TARGET, target)):
            if not word.startswith(mark):
                continue
            if value is None:
                return None
            label = word[len(mark):]
            if not label:
                word = str(value)
            elif label in rules:
                word = rules[label](str(value))
            else:
                return None         # tanınmayan ek: doldurulamaz
            break
        said.append(word)
    return " ".join(said)


def verified(templates, relation, concept, target, language, kind):
    """Bu olgu için, GERİ OKUNDUĞUNDA aynı olguyu veren ilk söyleyiş.

    Ölçüt bu projenin kendi denetimi: bir cümle, sistemin kendi dilbilgisiyle
    okunduğunda başladığı olguya dönüyorsa doğru kurulmuştur. Dönmüyorsa
    şablon bu olguya uymuyor demektir ve bir sonrakine geçilir.

    Hiçbiri tutmazsa None döner ve çağıran eski söyleyişine düşer — öğrenme
    bir kolaylık, bağımlılık değil.

    SINIRI ÖLÇÜLDÜ: geri okuma gerekli ama yeterli değil. property kalıbı
    cümlenin sonundaki fazladan kelimeye aldırmıyor, bu yüzden
    "ōkuninushi büyük kahverengidir" de sınavı geçiyor (27/50). Onları eleyen
    şey sınav değil, `_is_general` ile tanık sayısı: aynı havuzda en çok
    tanıklı şablon "{kavram} {hedef}-DIr" (72 tanık) ve o 50/50 alıyor.
    """
    for template, _ in templates.get(relation, ()):
        said = say(template, concept, target)
        if not said:
            continue
        read = language.understand(said)
        if (read.kind == kind and read.concept == concept
                and read.relation == relation
                and str(read.target or "").startswith(str(target or "")[:4])):
            return said
    return None
