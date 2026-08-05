"""Cümlenin hangi parçasının kavram, hangisinin hedef olduğunu ÖĞRENİR.

Sistemin bugünkü okuyucusu iki parçalı: eğitilmiş ağ niyeti söylüyor
(`core/intent.py`, ölçüldü %66-69), ama KAVRAMI ve HEDEFİ hâlâ kurallı
ayrıştırıcı buluyor — ek soyarak, `lmm/turkish.py`'deki 416 elle yazılmış
ögeye bakarak.

`turkish.py`'nin yaşamasının tek sebebi bu. Bir dil modelinde o dosyanın
karşılığı yok, çünkü onun ayrıştırıcısı yok: okuma da veriden öğrenilmiş.

Burada aynısı yapılıyor. Girdi bir cümle, çıktı her PARÇA için bir etiket:

    Kartal, yırtıcı bir kuş türüdür.
    K K K   O O O   O   H H H  O      <- K kavram · H hedef · O dışarısı

Etiketler ELLE YAZILMIYOR, hizalamayla çıkıyor: graf "kartal" diyor, metinde
nerede geçtiği bulunuyor, gerisi veri. Hiçbir ek listesi, hiçbir kalıp,
hiçbir sözcük sınıfı kullanılmıyor — ne eğitimde ne çalışırken.

ÇİFT YÖNLÜ. `core/model.py` dikkati `causal=False` ile ileriye de bakabiliyor
ve bu dosyanın kendi belgesinde gerekçesi yazılı: bir cümleyi anlamak için
sondaki eki görmek gerekiyor. Üretimde nedensellik şart, anlamada tam tersi.

`lmm/` paketinin dışında duruyor ve bu bilinçli: bellek, muhakeme ve
epistemik kapı hiçbir zaman torch'a muhtaç olmamalı.
"""
import json
import os

OUT, CONCEPT, TARGET = 0, 1, 2
NAMES = {OUT: "dışarısı", CONCEPT: "kavram", TARGET: "hedef"}


def span_of(text, name):
    """Adın metindeki yeri — çekimli hâliyle de olsa.

    Graf "kuş" tutuyor, metin "kuştur" diyor. Kökün başladığı yer bulunuyor ve
    kelime sonuna kadar uzatılıyor: ek de kavramın parçası sayılıyor, çünkü
    okuyucu ekli hâli görecek ve kökü çıkarmayı KENDİSİ öğrenmeli. Hangi ekin
    olduğu hiç sorulmuyor — sorulsaydı bu dosya da bir ek listesine muhtaç
    olurdu.
    """
    if not name:
        return None
    lowered, name = text.lower(), name.lower()
    at = lowered.find(name)
    if at < 0:
        return None
    end = at + len(name)
    while end < len(text) and (text[end].isalpha() or text[end] == "'"):
        end += 1
    return at, min(end, len(text))


def labels_for(sentence, concept, target):
    """Cümlenin her harfi için etiket. Hizalanamıyorsa None."""
    if not (sentence and concept):
        return None
    marks = [OUT] * len(sentence)
    place = span_of(sentence, concept)
    if place is None:
        return None
    for at in range(*place):
        marks[at] = CONCEPT
    if target:
        found = span_of(sentence, target)
        if found is None:
            return None
        # Çakışma varsa hizalama güvenilmez: aynı harfler iki role verilemez.
        if any(marks[at] == CONCEPT for at in range(*found)):
            return None
        for at in range(*found):
            marks[at] = TARGET
    return marks


def rows_from(paths, most=None, longest=2048):
    """Okuma çıktılarından etiketli örnekler — CÜMLE BAŞINA BİR TANE.

    İlk yazışta her olgu ayrı bir örnekti ve veri kendiyle çelişiyordu.
    Ölçüldü: 171.933 cümleden 147.061'i (yani %86) birden çok olgu veriyor ve
    aynı cümle iki kez, iki farklı hedef işaretlenmiş hâlde görünüyordu —

        "Kitaplarının bazıları Türkiye'de yasadışıdır."
          örnek 1: hedef = yasadışıdır
          örnek 2: hedef = Türkiye'de

    Model ikisini birden "doğru" diye görüyor ve hangisini işaretleyeceğini
    öğrenemiyor. İlk eğitimin sonucu bunu gösterdi: kavram %37,9, ama kavram
    ve hedef BİRLİKTE %0,9.

    Bir cümlenin bir öznesi vardır ve birden çok şey söylenir. Doğru biçim de
    o: kavram bir kez, hedeflerin hepsi. Öznesi farklı olan olgular
    düşürülüyor — onlar yan cümleden geliyor ve aynı diziye sığmazlar.
    """
    held = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            sentence = row.get("cümle")
            concept = (row.get("kavram") or "").strip()
            relation = (row.get("ilişki") or "").strip()
            target = (row.get("hedef") or "").strip()
            if not (sentence and concept and relation):
                continue
            if len(sentence) > longest:
                continue
            held.setdefault(sentence, []).append((concept, relation, target))

    found = []
    for sentence, facts in held.items():
        # Öznesi en çok tekrarlanan kavram: cümlenin konusu o.
        counted = {}
        for concept, _, _ in facts:
            counted[concept] = counted.get(concept, 0) + 1
        subject = max(counted, key=counted.get)
        place = span_of(sentence, subject)
        if place is None:
            continue
        marks = [OUT] * len(sentence)
        for at in range(*place):
            marks[at] = CONCEPT
        relations = []
        for concept, relation, target in facts:
            if concept != subject:
                continue
            if not target:
                # HEDEFSİZ ÖRNEK GEÇERLİ. Soru cümlelerinde hedef yok —
                # "kartal nedir" sorusunda söylenecek bir hedef yoktur, ve
                # onları elemek ağı yalnız bildirme cümlesiyle eğitiyordu.
                # Ölçüldü ve tam bu yüzden sorularda çalışmıyordu:
                #     "Kartal, yırtıcı bir kuş türüdür." -> kavram=Kartal ✓
                #     "kartal nedir"                     -> kavram='rtal'  ✗
                relations.append(relation)
                continue
            where = span_of(sentence, target)
            if where is None:
                continue
            if any(marks[at] != OUT for at in range(*where)):
                continue        # çakışma: aynı harf iki role verilemez
            for at in range(*where):
                marks[at] = TARGET
            relations.append(relation)
        if not relations:
            continue
        found.append((sentence, marks, relations[0]))
        if most and len(found) >= most:
            return found
    return found


def spans_from(text, marks):
    """Etiket dizisinden (kavram, hedef) metinlerini geri okur.

    Eğitimin tersi: ağ etiket veriyor, bu onu olguya çeviriyor. En uzun
    kesintisiz dizi alınıyor — ağ ortada bir harf kaçırsa bile parça
    bulunabilsin.
    """
    found = {}
    for mark in (CONCEPT, TARGET):
        best, run, start = (0, 0), 0, None
        for at, one in enumerate(list(marks) + [OUT]):
            if one == mark:
                if start is None:
                    start = at
                run += 1
            else:
                if start is not None and run > best[1] - best[0]:
                    best = (start, at)
                start, run = None, 0
        if not best[1]:
            found[mark] = ""
            continue
        # KELİME SINIRINA UZAT. Ağ doğru bölgeyi buluyor ama kelimenin
        # ortasında kesiyor — ölçüldü: "Kartal nedir?" için `rta`, "kartal
        # nedir" için `kart`. Harf düzeyinde çalışan bir ağın "kelime nerede
        # biter" diye bir kavramı yok ve onu da öğrenmesi gerekiyor; 20 tur
        # yetmedi.
        #
        # Bu bir DİL KURALI değil, çıktının okunma biçimi: boşluğa kadar
        # uzatmak Türkçe hakkında bir şey bilmeyi gerektirmiyor. Ek listesi,
        # kelime listesi, kalıp yok — yalnız "boşluk kelimeleri ayırır".
        start, end = best
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        while end < len(text) and not text[end].isspace():
            end += 1
        found[mark] = text[start:end].strip(" ,.;:!?'\"")
    return found[CONCEPT], found[TARGET]
