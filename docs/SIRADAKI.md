# Sıradaki işler

Her madde bir ölçüme dayanıyor. Tahmin yok: hangi sayının neyi söylediği,
işin neden o sırada olduğu, ve bittiğinde neye bakılacağı yazılı.

## Bugünkü durum (3 Ağustos 2026)

    graf              128.485 olgu · 58.245 kavram
    ilişki dağılımı   type %48,5 · property %42,7 · has %7,5 · can %1,1
    olcut             tohum 7 %97,5 · tohum 31 %97,5
    sohbet sınavı     açılış %98,3 · bağlam tutma %100
    gerçek sorular    cevaplıyor %14,5 · UYDURMA 0
    hız               kurulum 1,33 sn · soru başına 56 ms
    testler           702

Cevaplama %1,0'dan %14,5'e çıktı ve bu **veri eklenmeden** oldu. Sebebi
teşhisin yanlış olmasıydı: "graf bilmiyor" sanılan şeyi graf biliyordu.

    "Ormanların önemi ve faydaları nelerdir?"
      ayrıştırıcı -> 'ormanların önemi'   grafta yok
      gerçek özne -> 'orman'              grafta 15 olgu VAR

Ayrıştırıcı bir cümleden TEK kavram çıkarıyordu ve sonrası ona bağlıydı.
Artık aday üretiliyor (alt öbek, ek soyma, gömme komşusu), graf onları
ağırlıklandırıyor, cevap üreten kazanıyor, kapı değişmedi.

**Sayıyı okurken.** Ara ölçümde %31,5 görünmüştü ve o sayı dürüst değildi:
cevapların yarısı sorulmayan soruya veriliyordu ("Fosfor hangi besinlerde
bulunur?" -> "Kükürt bir elementtir"). Gömme komşusu **başka bir şeydir**;
artık yalnız kimliği koruyan aday (alt öbek, ek soyma) anlatabiliyor.
%14,5 elle sayılmış: 29 cevabın 24'ü sorulan öznede.

---

## 1. Okuma — %33,5 hâlâ ayrıştırılamıyor

**Neden.** Kalan en büyük dilim bu. Ama tavanı ölçüldü ve düşük: okunamayan
67 sorunun **%76'sında grafın hiç bilmediği bir konu var**. Oralarda ret
zaten doğru davranış:

    "Öyleyse her şey yolunda mı?"          grafta karşılığı yok
    "Ne o, yoksa onlardan korkuyor musunuz?"  grafta karşılığı yok

Kazanılabilir olan %24, yani tüm soruların **%8'i**.

**İş.** Round-trip'i geçen okumaları kalıp tümevarımına sinyal olarak vermek
(kendi kendine denetim). Sistemin doğru okuduğu her cümle, benzer cümleleri
okumayı öğretir.

**Bittiğinde bakılacak.** `1_OKUNAMADI` payı; `olcut` düşmemeli.

---

## 2. Kavram grafta yok — %39,5

**Neden.** Aday üretimi bunun bir kısmını kurtardı, kalanında çıkarılan şey
zaten kavram değil: `urfa’da` (graf `urfa`yı hiç bilmiyor), `fibroadenom`,
`2çalışma`, `»genel`. Yapı sözcükleri (`bu`, `böyle`, `olan`) artık eleniyor.

**İş.** İki ayrı şey karışıyor ve ayrılması lazım:

    ÖZNE ÇIKARMA   "turizmle" -> "turizm". Araç eki ünsüzden sonra `-le`,
                   envanterde yalnız ünlü sonrası `-yle` var (`lmm/turkish.py`
                   `oblique_suffixes`). Geniş bir liste eklemek riskli:
                   `damla` -> `dam`, `tarla` -> `tar`. Graf doğrulaması var
                   ama önce ölçülmeli.
    GERÇEK EKSİK   `fibroadenom` grafta yok ve olmalı. Burası veri işi.

**Bittiğinde bakılacak.** `3_KAVRAM_GRAFTA_YOK` payı; uydurma 0 kalmalı.

---

## 3. Anlam seçimi — kuruldu, sınırı var

**Yapıldı.** Soru artık hangi anlamın konuşacağını seçiyor:

    "Kerevizli dip sos nasıl yapılır?"  -> dash var, soya var, mirin var, tatlı
    "Bu oyunda sos nasıl oynanır?"      -> Sos bir oyundur, eğlencelidir

Ölçüt provenans: aynı cümleden çıkan olgular aynı anlama ait. Yakınlığı
gömme veriyor ama gömme yalnız SEÇİYOR, hiçbir olgu yazmıyor.

**Kalan sınır.** Seçilen anlamın tür kaydı yoksa soyağacı susuyor ve cevap
"Ayrıca ..." diye başlıyor. Doğru ama kırık bir cümle. Öteki anlamın
olguları da sona atılıyor, atılmıyor — bilerek, çünkü elemek ölçümde sınavı
düşürmüştü.

---

## 4. Şeyleştirme — bileşik ifade

**Neden.** Olgu birimi `(kavram, ilişki, hedef)` üçlüsü ve nitelemeye yer yok.
"Kırmızı olmayan büyük kuş" ifade edilemiyor.

**İş.** Olgunun kendisi düğüm olsun (RDF şeyleştirmesi, Davidson olay
semantiği). `lmm/memory.py` `Edge` ve kimlik anahtarı, `lmm/frames.py`
`to_fact`, `lmm/grammar.py` `_capture_span`.

**Sınır.** Bu mimari bir değişiklik; 702 testin çoğu üçlü varsayıyor. Önce
küçük bir dilim, ölçüm, sonra genişletme.

---

## 5. Ek keşfini bitir

**Neden.** Elle yazılan tek katman kaldı: 122 ek ve 54 kapalı sınıf sözcüğü.
Keşif organı çalışıyor ama %47'de:

    bulunan      koşaç 8/8 · ayrılma 7/7 · çoğul 2/2 · tamlayan 7/8
    bulunamayan  zaman 0/4 · yeterlilik 0/2 · koşul 0/2

**Neden en sonda.** İkinci bir dil eklemenin önündeki engel bu, ama Türkçe
çalışırken kimseyi durdurmuyor.

---

## Sırayı bozmamak için

Her madde bittiğinde **üç ölçüm birden** koşulacak, çünkü bir kazanç başka bir
kaybı gizleyebiliyor:

    python3 scripts/durum.py --graf models/graph/birlesik.lmm
    python3 -m unittest discover -s tests

Uydurma satırı **her zaman 0** olmalı; olmuyorsa iş geri alınır.

**Ve sayının içine bakılacak.** Bu gece cevaplama %31,5 göründü, testler
geçti, uydurma 0'dı — ve cevapların yarısı çöptü. Bir sayının yükselmesi
işin doğru olduğunu göstermez; cevapları elle okumak gösterir.
