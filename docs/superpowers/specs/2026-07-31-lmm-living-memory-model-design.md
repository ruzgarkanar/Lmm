# LMM — Living Memory Model (Yaşayan Bellek Modeli)
## Tasarım Dokümanı v1.0

**Tarih:** 31 Temmuz 2026
**Sahip:** Rüzgar Ersin Kanar
**Durum:** Tasarım onaylandı — implementasyon planı bekliyor
**İlgili araştırma:** `docs/arastirma/2026-07-31-arastirma-haritasi.md`

---

## 1. Vizyon ve Kategori Tanımı

LMM, LLM'in bir iyileştirmesi değil, yeni bir model kategorisidir. Tanımlayıcı fark:

> **LLM'de bilgi, eğitim anında ağırlıklara dondurulur. LMM'de bilgi, yaşayan bir bellekte durur: her an yazılabilir, kalıcıdır, kaynaklıdır ve modelin "biliyorum/bilmiyorum" ayrımının temelidir.**

| Boyut | LLM | LMM |
|---|---|---|
| Bilginin yeri | Ağırlıklarda donmuş | Yaşayan bellekte, anında yazılır |
| Öğrenme | Eğitimde bir kere; sonrası donuk | Her etkileşimde, anında, kalıcı |
| Bilmediğinde | Olasılıksal tahmin üretir (halüsinasyon) | Yapısal olarak üretemez; sorar/araştırır |
| Güç kaynağı | Veri + GPU ölçeği | Mekanizma: sezgi + bellek + muhakeme ayrımı |
| Çelişen bilgi | Sessizce ezilir/karışır | Fark edilir, sorgulanır, istisna olarak işlenir |

### Dört sütun (başarı tanımı)
1. **Sürekli kalıcı öğrenme** — yeniden eğitim yok; öğrenilen bilgi restart sonrası durur.
2. **Epistemik dürüstlük** — bilmediğini bilir; uydurmak mimari olarak imkânsızdır.
3. **Minimal kaynak** — sıradan bir bilgisayarda, saf Python'da çalışır.
4. **Mantık birinci sınıf** — çıkarım, istisna ve çelişki yönetimi çekirdek mekanizmadır.

## 2. Kapsam (v0)

- **Dünya:** küçük kontrollü Türkçe dil dünyası. Sınırlı kelime dağarcığı ve cümle kalıplarıyla başlar; konuştukça büyür.
- **Etkileşim:** komut satırında metin sohbeti (öğretmen–öğrenci modeli). Kullanıcı öğretir, sorar; sistem cevaplar, sorar, öğrenir.
- **Kısıt:** saf Python, sıfır harici paket (yalnızca standart kütüphane). Her satır bizim.
- **Kapsam dışı (v0):** açık uçlu genel dil, çoklu dil, web'den otonom araştırma (v2 hedefi), ağırlık konsolidasyonu (v2 hedefi).

## 3. Mimari — Beş Organ

### 3.1 Dil Sezgisi Çekirdeği (nöral)
- Mini nöral ağ: tokenizer + embedding + küçük attention/MLP; tamamı elle yazılmış (bkz. `docs/llm.docx` matematiği).
- **Tek görevi dil:** cümleyi (niyet, kavramlar, ilişki) üçlüsüne çevirmek. Örn. "Penguen bir kuştur" → `(ÖĞRET, penguen —tür→ kuş)`.
- **Bilgi taşımaz.** Bilgiyi bellekten alması, ağın küçük kalabilmesinin tezidir (büyük verinin asıl tükettiği şey bilgiydi).
- Girdi/çıktı sözleşmesi: `anla(cümle) -> Niyet(tür, kavramlar, ilişki, güven)`.

### 3.2 Yaşayan Bellek
- Kavram ağı (graph): düğüm = kavram, kenar = ilişki (`tür`, `yapabilir`, `sahiptir`, `değildir`...).
- Her kenar metadata taşır: **kaynak** (kim öğretti), **güven skoru**, **zaman damgası**, **istisna bayrağı**.
- Anında yazılır; JSON olarak diske kalıcılaştırılır; restart'ta yüklenir.
- Bilgi silinmez; güncellenir veya çelişki/istisna olarak işaretlenir (iz kaybolmaz).
- Sözleşme: `sorgula(kavram, ilişki) -> [Kenar] | ISKA`, `yaz(kenar, kaynak) -> sonuç`.

### 3.3 Epistemik Kapı
- Her soru önce belleğe çarpar. **Cevap yalnızca bellek içeriğinden üretilebilir.**
- İsabet → cevap + kaynak. Iska → "bilmiyorum" + öğrenme isteği ("öğretir misin?").
- Düşük güvenli bilgi → çekimser cevap ("emin değilim, bana X demişti").
- Halüsinasyonun engellenme biçimi filtre değil, **üretim yolunun yokluğudur.**

### 3.4 Muhakeme Motoru
- Bellek üstünde zincirleme çıkarım: tür hiyerarşisinden kalıtım (`penguen —tür→ kuş`, `kuş —yapabilir→ uçmak` ⇒ "penguen uçabilir" *varsayımı*).
- **İstisna yönetimi:** doğrudan kenar, kalıtılan kenarı ezer (`penguen —yapamaz→ uçmak` kazanır).
- **Çelişki tespiti:** yeni bilgi mevcutla çatışırsa körce yazılmaz; sistem çelişkiyi söyler ve sorar ("Kuşlar uçar demiştin; penguen kuşsa nasıl uçamaz?") — cevaba göre istisna işler.
- Çıkarımla üretilen her cevap, dayandığı zinciri gösterebilir (açıklanabilirlik).

### 3.5 Öğrenme Döngüsü
- Akış: yeni bilgi → sezgi çekirdeği ayrıştırır → muhakeme çelişki kontrolü yapar → bellek yazımı → onay ("öğrendim").
- Aynı bilgi tekrar öğretilirse güven skoru artar; çelişen kaynak gelirse skor düşer/soru doğar.
- **v2 (bu spec kapsamı dışı, yön olarak):** sık kullanılan bellek örüntülerinin ağırlıklara yavaş sindirilmesi ("uyku/konsolidasyon"); dış kaynağa (dosya/web) otonom araştırma.

## 4. Veri Akışı

```
kullanıcı cümlesi
   → [Dil Sezgisi Çekirdeği] niyet + kavramlar
   → niyet == SORU  → [Epistemik Kapı] → [Muhakeme + Bellek]
                        isabet → cevap (+kaynak/zincir)
                        ıska   → "bilmiyorum, öğret"
   → niyet == ÖĞRET → [Muhakeme: çelişki kontrolü]
                        temiz   → [Bellek yazımı] → "öğrendim"
                        çelişki → soru → cevaba göre istisna/red
```

## 5. Hata ve Uç Durumlar

- **Anlaşılamayan cümle:** sezgi çekirdeği güveni eşik altındaysa sistem tahmin etmez; "bunu anlamadım, şöyle söyler misin?" der (epistemik dürüstlük dile de uygulanır).
- **Bellek dosyası bozuk/yok:** boş bellekle başlar, durumu bildirir; yazımlar atomiktir (geçici dosya + rename).
- **Döngüsel tür ilişkisi** (a türü b, b türü a): yazım anında reddedilir.
- **Kaynak çatışması:** aynı ilişkiye zıt bilgiler farklı kaynaklardan gelirse ikisi de saklanır, cevapta belirsizlik beyan edilir.

## 6. Test ve Kabul Kriterleri (v0 kanıt senaryoları)

1. **Kalıcılık:** "Penguen bir kuştur" öğret → programı kapat/aç → "Penguen nedir?" → doğru cevap, tekrar sormuyor.
2. **Epistemik dürüstlük:** hiç öğretilmemiş kavram sor → sistem cevap ÜRETEMİYOR, "bilmiyorum" diyor. (Halüsinasyon oranı tanım gereği %0; test bunun delinmediğini doğrular.)
3. **Penguen testi (mantık):** "kuşlar uçar" + "penguen kuştur" + "penguen uçamaz" öğret → çelişkiyi FARK ETMESİ, sorması ve istisna işlemesi; sonra "penguen uçar mı?" → "hayır", "serçe uçar mı?" → "evet (kuş olduğu için)".
4. **Güven/kaynak:** aynı soruya cevapta kaynağını söyleyebilmesi ("bunu sen öğretmiştin").
5. **Kaynak tüketimi:** tüm senaryolar sıradan laptop'ta, harici paketsiz, insan-algısında anlık (<1 sn) cevapla geçer.
6. Her organ bağımsız test edilir: tokenizer/ağ, bellek CRUD+kalıcılık, kapı isabet/ıska, muhakeme zincir/istisna/çelişki.

## 7. Riskler ve Dürüst Sınırlar

- **Dil esnekliği:** v0'da sezgi çekirdeği küçük olduğundan kalıp dışı cümlelerde "anlamadım" sık duyulacak. Bu kabul edilmiş bir v0 sınırıdır; çekirdek konuştukça eğitilir.
- **Bellek kirlenmesi:** yanlış öğretilen bilgi sistemin gerçeği olur. v0 savunması kaynak+güven metadata'sı; v1'de çok kaynaklı doğrulama.
- **Ölçek iddiası yok:** v0, mekanizma kanıtıdır; genel bilgi genişliğinde LLM ile yarışmaz. İddia güç değil, LLM'in yapısal olarak yapamadığı davranışlardır.

## 8. Yol Haritası

- **v0:** bu spec — 5 organ, CLI sohbet, kanıt senaryoları.
- **v1:** kelime dağarcığının etkileşimle büyümesi, çok kaynaklı güven, daha zengin ilişki türleri.
- **v2:** konsolidasyon (bellek→ağırlık sindirimi), dış kaynaktan (dosya/doküman) otonom öğrenme, "merak → araştır → doğrula → yaz" tam döngüsü.
