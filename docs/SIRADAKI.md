# Sıradaki işler

Her madde bir ölçüme dayanıyor. Tahmin yok: hangi sayının neyi söylediği,
işin neden o sırada olduğu, ve bittiğinde neye bakılacağı yazılı.

## Bugünkü durum (3 Ağustos 2026)

    graf              128.485 olgu · 58.245 kavram
    ilişki dağılımı   type %48,5 · property %42,7 · has %7,5 · can %1,1
    olcut             tohum 7 %97,3 · tohum 31 %98,3
    sohbet sınavı     açılış %98,3 · bağlam tutma %100
    gerçek sorular    anlıyor %86,5 · cevaplıyor %5,0 · dürüst ret %91 · UYDURMA 0
    hız               128 binlik grafta kurulum 1,26 sn · soru başına 21 ms
    testler           702

Darboğaz artık dil değil **bilgi**: 400 gerçek sorunun %56,1'inde grafın hiç
duymadığı bir kavram var (`glokom`, `sendromu`, `belirtileri`). Ayrıştırıcı
cümlelerin %86,5'ini okuyor ama graf cevap veremiyor.

---

## 1. Gömmeyi derlemle kur ve bilinmeyen kelimeye bağla

**Neden.** `lmm/vectors.py` yazılmış, ölçülmüş, kullanılmıyor. Ölçüm 2.105
cümlelik oyuncak bir derlemle bile rastgelenin iki katı:

    sınıf      kelime   komşusu aynı sınıf   rastgele olsa
    nitelik       264              %24            %17
    kavram        550              %61            %34
    TOPLAM                         %49            %25

    kuş  ->  hayvandır, memeli, arı, kediler, tüylüdür, balık

Elimizde 808 MB derlem var ve hiç kullanılmadı. Gradyan yok, GPU yok,
rastgelelik yok — Levy & Goldberg 2014'e göre gömme zaten bir PMI
çarpanlaması, gradyan yalnızca ona varmanın bir yolu.

**İş.** `scripts/` altına bir kurucu: `data/tr-metin.txt`'ten sayarak vektör
üret, model dosyasına yaz. Sonra `gate._dont_know` yolunda bilinmeyen kelime
için en yakın BİLİNEN kavramı öner.

**Sınır — mutlak.** Geometri aday bulur, **kararı kapı verir**. Dosyanın kendi
belgesi bunu söylüyor: dağılımsal benzerlik zıtları ayıramaz, "neden oldu" ile
"engelledi" aynı çevrede geçer. Vektör asla bir olgu yazmamalı.

**Bittiğinde bakılacak.** `scripts/durum.py` — "gerçek cümleler" bölümünde
`cevapladı` yükselmeli, `uydurma` 0 kalmalı. Kalmıyorsa iş yanlış yapılmıştır.

---

## 2. Bağlamdan anlam seçimi

**Neden.** Çokanlamlılık kalan en büyük mantık kusuru ve graf büyüdükçe
kötüleşiyor: 16 binlik grafta kavramların %1,8'i çok türlüydü, 35 binlikte
%8,1. Bugün `kartal` hem kuş hem İstanbul ilçesi ve nitelikleri karışıyor:

    kartal ile penguen arasındaki fark nedir
    -> "kartal: avlanmak yapabilir, hızlı, istanbul sahip, kurul"

Kalıtımda yayılmasını durdurduk (`reasoning.lineage`), kökte duruyor.

**İş.** `lmm/spreading.py` kişiselleştirilmiş PageRank yapıyor ama oturum
odağına bağlı değil. Bağlanacak: sohbet neyden bahsediyorsa o anlam seçilsin.
Tam çözüm düğüm kimliğine anlam eklemek (`kartal#kuş`) — daha büyük iş, önce
bağlam seçimi ölçülmeli.

**Bittiğinde bakılacak.** "kuşlardan bahsederken kartal" ile "şehirlerden
bahsederken kartal" farklı cevap vermeli, ve `olcut` düşmemeli.

---

## 3. Gövde metnini oku

**Neden.** Tanımlar bitti — 99.218 tanım okundu, 123.631 olgu geçti. Ama
tanım bir şeyin NE OLDUĞUNU söyler, NASIL olduğunu değil. `can` ilişkisi hâlâ
%1,1 (1.450 olgu) ve "ne yapabilir" soruları bundan besleniyor.

**İş.** Aynı okuyucu hattı (`scripts/okuyucu_dene.py` + `okuma_al.py`),
girdi `data/tr-metin.txt`. Cümle seçimi gerekiyor: 808 MB'ın tamamı değil,
özne+yüklem taşıyan sade cümleler.

**Uyarı — ölçüldü.** Ham gövde metnini KENDİ ayrıştırıcımızla okumak %9 verim
ve çöp kavram veriyordu (`aşağıdaki kupa`, `sukhbaatar kızıl bayrak`).
Okuyucusuz denemeye değmez.

**Bittiğinde bakılacak.** `can` ve `has` payı; "ne yapabilir" cevaplarının
zenginliği.

---

## 4. Şeyleştirme — bileşik ifade

**Neden.** Olgu birimi `(kavram, ilişki, hedef)` üçlüsü ve nitelemeye yer yok.
"Kırmızı olmayan büyük kuş" ifade edilemiyor. Bu gece nitelemenin özneye
YAPIŞMASINI engelledik ama söyleyemiyoruz hâlâ.

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

Üretken gövde ölçütü fiil eklerini 0'dan 4'e çıkardı ama iyelik 4/8'den 0/8'e
düştü — net kazanç iki ek. Organ bitmedi.

**Neden en sonda.** İkinci bir dil eklemenin önündeki engel bu, ama Türkçe
çalışırken kimseyi durdurmuyor. Önceki dört madde ölçülebilir kazanç veriyor,
bu vermiyor.

---

## Sırayı bozmamak için

Her madde bittiğinde **üç ölçüm birden** koşulacak, çünkü bir kazanç başka bir
kaybı gizleyebiliyor — bu gece iki kez oldu:

    python3 scripts/durum.py --graf models/graph/birlesik.lmm
    python3 -m unittest discover -s tests

`durum.py` beşini bir arada basıyor: graf, olcut (iki tohum), sohbet sınavı,
gerçek cümleler, hız. Uydurma satırı **her zaman 0** olmalı; olmuyorsa iş
geri alınır.
