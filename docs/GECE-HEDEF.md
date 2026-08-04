# Gece hedefi ve sonucu — 4 Ağustos 2026

Ölçülebilir olmayan hedef, hedef değildir. Sabah bakılacak sayılar bunlar.

## Nerede başladık, nereye geldik

| ölçü | gece başı | şimdi | hedef |
|---|---|---|---|
| graf | 139.912 olgu · 61.628 kavram | **252.209 · 90.022** | — |
| `can` payı | %2,7 | **%11,2** | — |
| nesne taşıyan olgu | %1,2 | **%8,0** | %15 |
| gerçek soruda cevaplama | %22,5 | **%35,5** | %50 |
| okunamayan cümle | %27,5 | **%19,0** | %15 |
| olcut (açık uçlu) | %95,8 / %91,7 | **%80,8 / %80,8** | — |
| olcut (doğrudan) | — | **%96,3 · 0 yanlış** | %95 |
| uzun sohbet: hamle | 32/32 | **32/32** | tam |
| uzun sohbet: eksiltili | 24/24 | **24/24** | tam |
| uzun sohbet: gerçek soru | 2/16 | **7/16** | — |
| cevap uzunluğu (anlat) | 391 kr | **716 kr** | LLM 1372 |
| uzunluk / soruya uyum | 5,1x | **9,4x** | LLM ~4x |
| UYDURMA | 0 | **0** | 0 |

## Gecenin en önemli bulgusu

`olcut` düştü ve bu, aylardır çözülemeyen bilmecenin ta kendisiydi. Cevap:

    aynı olgular AÇIK UÇLU sorulduğunda   %95,8 -> %80,8
    aynı olgular DOĞRUDAN sorulduğunda    %96,7 -> %96,3   (0 yanlış)

`olcut` "biliyor mu"yu değil "sayıyor mu"yu ölçüyor. Kavram başına olgu
arttıkça belirli birinin ilk sekizde çıkma olasılığı yapısal olarak düşüyor.
Bilgi kaybolmadı. `scripts/olcut_dogrudan.py` artık ikinci ölçü olarak duruyor
ve **düşerse gerçek gerileme odur**.

Bu bulgu bir grafın çöpe atılmasını önledi: `DENEMELER.md`de kayıtlı
"%97,3 -> %78,0" çöküşü de aynı şeydi.

## Ne yapıldı

**Veri.** 73.768 web cümlesi okundu (2,2 saat, 42M+5,5M token), 123.864 olgu
kapıdan geçti. Nesne çıkarımı ilk kez açıldı.

**Nesir.** Cevap artık türlerine göre bölünüyor: ne olduğu, ne yaptığı, nesi
olduğu ayrı cümlelerde. Uzunluk soruya bağlı — `anlat` geniş, `kısaca` dar,
cümle içindeki istek oturum yönergesini geçici olarak ezer.

**Güven.** Beş basamak (`işletmeci > belge > hat > çıkarım > yabancı`) ve
kaynak sınıfı artık BİÇİMDEN okunuyor: belgenin dosya adı vardır, hattın
etiketi. Önek listesi iki kez sessizce kırıldı; yapısal kural kırılmıyor.
Doğrulanmamış yabancı sözü cevapta işaretli görünüyor.

**Sohbet.** Konu açma, tepki, itiraz, vazgeçme, söylem arası, sondaki açıcı,
eksiltili takip, ikili gönderme — hepsi çalışıyor ve kapalı sınıf olarak
`turkish.py`de duruyor.

**Hız.** Bir cümle öğretmek 1,00 sn -> 0,26 sn (tümevarım artık ailede hiç
kimsede olmayan niteliği denemiyor).

## Kalan ve neden

**Kavram başına ortanca 2 olgu.** Hedef 6'ydı, tutmadı — çünkü her okuma
turu yeni KAVRAM da getiriyor ve ortanca yerinde sayıyor. Ortalama 3,0.

**Gerçek soruda %35,5.** Kalanın %28'inde kavram grafta yok, %15,5'inde
kavram var ama sorulan olgu yok. İkisi de veri.

**`olcut` %80,8.** Açık uçlu ölçü; bilgi ölçüsü (%96,3) sağlam. Buradaki
kayıp "hangi sekizini söylüyoruz" sorusudur.

## Değişmez sınır — tutuldu

Uydurma satırı gecenin her ölçümünde **0**.
