# LMM — Living Memory Model

> LLM'de bilgi, eğitim anında ağırlıklara dondurulur.
> LMM'de bilgi, yaşayan bir bellekte durur: her an yazılabilir, kalıcıdır,
> kaynaklıdır ve sistemin "biliyorum / bilmiyorum" ayrımının temelidir.

Saf Python, sıfır harici paket. Sıradan bir laptopta çalışır, GPU istemez.

## Dene

```bash
python3 -m lmm.cli memory.json
```

```
> penguen nedir
bilmiyorum. penguen hakkında bunu bana öğretir misin?
> kuşlar uçar
öğrendim: kuş uçar.
> penguen bir kuştur
öğrendim: penguen bir kuştur.
> penguen uçar mı
evet, çünkü penguen bir kuş ve kuş uçar.
> penguen uçamaz
bir çelişki fark ettim: şu an bildiğim: penguen bir kuş çünkü kuş uçar.
yine de 'penguen uçamaz' olarak öğreneyim mi? (evet/hayır)
> evet
öğrendim (istisna olarak işledim).
> penguen uçar mı
hayır, çünkü penguen uçamaz (doğrudan bilgi, kaynak: sen).
```

Programı kapatıp yeniden açın: hatırlar, bir daha sormaz.

## Bildiği bilgi türleri

| Tür | Öğretme | Sorma |
|---|---|---|
| Tür | `penguen bir kuştur` / `penguen bir memeli değildir` | `penguen nedir` |
| Yetenek | `kuşlar uçar` / `penguen uçamaz` | `penguen uçar mı`, `kimler uçar`, `penguen ne yapabilir`, `penguen neden uçamaz` |
| Özellik | `kuşlar tüylüdür` / `penguen tüylü değildir` | `penguen tüylü mü`, `penguen nasıldır` |

Türkçe, türü özellikten tek kelimeyle ayırıyor: **"bir"**. `penguen bir kuştur`
onun *ne olduğunu*, `penguen siyahtır` *nasıl olduğunu* söyler. Sistem bu ayrımı
dilin kendisinden alıyor. Her iki bilgi türü de kalıtımla iner ve istisna kabul
eder — aynı muhakeme, farklı ilişki.

```
> kuşlar tüylüdür
öğrendim: kuş tüylüdür. bu arada, bunu hiç öğrenmedim: kuş nedir?
> penguen bir kuştur
> penguen tüylü mü
evet, çünkü penguen bir kuş ve kuş tüylüdür.   ← kalıtım
> penguen siyahtır
> penguen nasıldır
penguen tüylü ve siyahtır.
> penguen tüylü değildir
bir çelişki fark ettim: şu an bildiğim: penguen bir kuş çünkü kuş tüylüdür...
```

## Konuşabildikleri

```
> kimler uçar
kuş ve serçe uçar.                          ← penguen hariç: istisnaya saygı
> penguen neden uçamaz
çünkü penguen uçamaz (doğrudan bilgi, kaynak: sen).
> serçe neden uçar
çünkü serçe bir kuş ve kuş uçar.            ← muhakeme zinciri görünür
> penguen neden uçar
aslında penguen uçamaz.                     ← yanlış varsayımı düzeltir
> penguen ne yapabilir
penguen uçamaz.
> penguen bir kuş değildir
bir çelişki fark ettim: şu an bildiğim: penguen bir kuştur (kaynak: sen)...
> kimler konuşur
konuşur diye bir şeyi hiç duymadım.         ← duymadığı eylemi uydurmaz
```

## Dil organı takılabilir

`lmm/language.py` tek bir sözleşme tanımlar: cümleyi alıp niyet döndürmek.
Bellek, muhakeme, kapı ve merak organları hiçbir zaman bir cümle görmez — kavram
ve ilişkilerle çalışırlar. Kendi `Intuition`ımız bu sözleşmeyi uygular; isteyen
yerine bir dil modeli takabilir ve o model **neyin doğru olduğu konusunda hiçbir
yetkiye sahip olmaz**, çünkü doğruluk bellekte yaşar ve cevabı kapı kurar.

```python
session = Session("memory.json", language=BenimDilOrganım())
```

## Tümevarım — kimsenin söylemediğini öğrenmek

Verilen bilgileri zincirlemek öğrenmek değildir. Bir LLM'in asıl üstünlüğü
**genelleme**dir: kimsenin yazmadığı şeyler hakkında cevap verebilir. LMM bunu
açıkta yapar:

```
> serçe bir kuştur
> kartal bir kuştur
> serçe uçar
> kartal uçar
öğrendim: kartal uçar. şunu fark ettim: serçe ve kartal — sanırım kuş uçar.
> güvercin bir kuştur
> güvercin uçar mı
evet, çünkü güvercin bir kuş ve kuş uçar.        ← kimse söylemedi
```

Kural, bilgi gibi belleğe yazılır ama **kaynağı sistemin kendisidir** ve güveni
düşük tutulur:

```
> kuşlar uçar mı
evet, çünkü kuş uçar (kendi çıkarımım).
```

En önemlisi, kendi tahminini savunmaz. Bir insan aksini söylediğinde tartışmadan
geri adım atar — ama **öğretilmiş** bilgiyi aynı sertlikte korumaya devam eder:

```
> penguen uçamaz
bunu çıkarımla varsaymıştım, seninkini üstün tutuyorum: penguen uçamaz.
```

Bir LLM de genelleme yapar. Ama sana genelleme yaptığını söyleyemez, kendisini
ikna eden örnekleri gösteremez, ve tek bir noktada düzeltilemez.

## Okuyarak öğrenme — konuşmadan önce

Sohbet tek giriş kanalı değil. Bir metin dosyası verildiğinde LMM anladığı
cümleleri kalıcı olarak öğrenir, anlamadıklarını **tahmin etmeden atlar**,
bildiğiyle çelişenleri ise yazmadan raporlar:

```bash
python3 -m lmm.reading memory.json packs/hayvanlar.txt
```

```
hayvanlar.txt: 16 cümle okundu — 14 yeni bilgi, 0 pekişen, 1 çelişkili, 1 atlandı.
  çelişki: penguen — şu an bildiğim: penguen bir kuş çünkü kuş uçar
  atlandı (anlaşılmadı): Bu cümle sistemin bilmediği bir kalıpta yazılmıştır
```

Çelişkiyi kendi başına çözmez, çünkü klavyenin başında karar verecek kimse
yoktur ve iki kaynak arasında sessizce taraf tutmak tahmindir. Rapor insana
gider, insan sohbette karara bağlar. Okuduktan sonra sistem soru sormaz,
cevaplar — ve bilgiyi hangi dosyadan aldığını söyler:

```
> balık tüylü mü
hayır, çünkü balık tüylü değildir (doğrudan bilgi, kaynak: hayvanlar.txt).
> kimler uçar
kuş, penguen, serçe ve kartal uçar.
```

## Hedef takibi — ajanın başka türlüsü

Cevabı bilmediği bir soru sorulduğunda LMM omuz silkmez. **Cevaplayabilmek için
neyi bilmesi gerektiğini belleğinden çıkarır**, onu ister, ve öğrenir öğrenmez
kendiliğinden ilk soruya döner:

```
> penguen uçar mı
bunu bilmiyorum. cevaplayabilmem için önce şunu öğrenmem lazım: penguen nedir?
> penguen bir kuştur
öğrendim: penguen bir kuştur. kuş uçar mı?
> kuşlar uçar
öğrendim: kuş uçar. şimdi ilk soruna dönebilirim: evet, çünkü penguen bir kuş ve kuş uçar.
```

Bir LLM ajanı hangi aracı çağıracağını tahmin eder. Buradaki plan tahmin değil,
hiyerarşiden okunur: "penguen bir kuş, öyleyse bana kuşları anlat" adımını
belleğin kendisi dikte eder. Hedef varken merak susar — amaçlı soru, başıboş
merakı bastırır.

## Merak — kimsenin kapatmadığı döngü

Sorulunca "bilmiyorum" demek dürüstlüktür. Asıl fark, **kimse sormadan kendi
eksiğini fark edip sormaktır**:

```
> kuşlar uçar
öğrendim: kuş uçar. bu arada, bunu hiç öğrenmedim: kuş nedir?
> kuş bir hayvandır
öğrendim: kuş bir hayvandır. bu arada, bunu hiç öğrenmedim: hayvan nedir?
> balık bir hayvandır
öğrendim: balık bir hayvandır. bu arada, bunu hiç öğrenmedim: balık uçar mı?
```

Sorular uydurulmaz; bellekteki gerçek boşluklardan doğar: hiyerarşide yeri
olmayan bir kavram, ya da dünyada var olduğu bilinen bir eylemle hiç
bağlanmamış bir kavram. Sorulan soru kalıcı olarak işaretlenir — restart sonrası
bile aynı soru bir daha sorulmaz. Fark et → sor → öğren → bir daha sorma.

## Dört sütun

1. **Sürekli kalıcı öğrenme** — yeniden eğitim yok; öğrenilen bilgi restart sonrası durur.
2. **Epistemik dürüstlük** — bilmediğini bilir; uydurmak filtrelenmez, ulaşılamazdır.
3. **Minimal kaynak** — 100 soru < 1 saniye, saf Python.
4. **Mantık birinci sınıf** — kalıtım, istisna ve çelişki yönetimi çekirdekte.

Her sütunun çalıştığının kanıtı: `tests/test_proof_scenarios.py`.

## Bellek Paketleri — LMM'in "model dosyası"

Bir LLM ağırlık dosyası olarak dağıtılır: kimsenin okuyamadığı milyarlarca sayı.
LMM'de dağıtılan şey **Bellek Paketi**dir: okunabilir, diff'lenebilir, tek tek
silinebilir bilgiler — her biri kaynağıyla birlikte.

```bash
python3 -m lmm.pack export memory.json kuslar-tr.json kuslar-tr   # paketle
python3 -m lmm.pack merge  memory.json kuslar-tr.json             # kur
```

Kurduktan sonra sistem bilgiyi nereden aldığını söyler:

```
> penguen nedir
penguen bir kuştur (kaynak: kuslar-tr@1.0).
```

**Paketler kelime de öğretir.** Bir alan, kendi fiillerini paketle birlikte
gönderir; motor koduna hiç dokunulmaz. `packs/lmm-tanitim.json` sistemin hiç
bilmediği dört kelimeyi (öğrenmek, hatırlamak, unutmak, uydurmak) getirir:

```bash
python3 -m lmm.pack merge memory.json packs/lmm-tanitim.json
# 4 yeni kelime, 14 yeni bilgi, 0 pekişen, 0 çelişkili.
```

```
> lmm ne yapabilir
lmm öğrenir, hatırlar, unutur ve uyduramaz.
> llm ne yapabilir
llm öğrenemez, unutamaz ve uydurur.
```

Kelimeler bilgiyle aynı dosyada saklanır ve restart sonrası durur — kelime de
öğrenilen bir şeydir. Yeni kelime öğrenmek hiçbir şeyi yeniden eğitmez.

Ağırlık dosyalarının yapamadığı üç şey:

- **Birleştirme çelişkiyi ortaya çıkarır.** İki paket aynı konuda ters şey
  söylüyorsa LMM ortalama almaz, sessizce taraf tutmaz — çelişkiyi raporlar,
  bilgi yazılmaz.
- **İstisnalar paketle seyahat eder.** "Kuşlar uçar" ile birlikte "penguen
  uçamaz" da taşınır; alıcı sistem çıkarımı doğru yapar.
- **Unutma gerçektir.** `memory.forget("penguen")` → sistem gerçekten bilmez
  hale gelir, "bilmiyorum" demeye döner. Eğitilmiş ağırlıklardan seçici silme
  pratikte imkânsızken burada bir liste filtresi.

## Organlar

| Modül | Görev |
|---|---|
| `lmm/memory.py` | Yaşayan bellek: kavram ağı, kaynak/güven, atomik kalıcılık |
| `lmm/reasoning.py` | Kalıtım, istisna, çelişki tespiti |
| `lmm/gate.py` | Epistemik kapı: cevap yalnızca bellekten |
| `lmm/intuition.py` | Cümle → niyet (kalıp ayrıştırıcı) |
| `lmm/network.py` | Elle yazılmış softmax ağı — güven sinyali |
| `lmm/learning.py` | Çelişki kontrollü kalıcı öğrenme |
| `lmm/curiosity.py` | Kendi boşluğunu fark etme ve soru sorma |
| `lmm/pursuit.py` | Hedef takibi: eksik halkayı bulup isteme |
| `lmm/reading.py` | Dokümandan öğrenme, atlama ve çelişki raporu |
| `lmm/induction.py` | Tümevarım: örüntüden kural çıkarma, işaretleme |
| `lmm/distill.py` | LLM'den bilgi damıtma, denetleyerek |
| `lmm/harvest.py` | Merakın sorduğunu LLM'e sorup mutabakatla alma |
| `lmm/trust.py` | Kaynak güven sıralaması |
| `lmm/lexicon.py` | Kelime dağarcığı — paketlerle büyür |
| `lmm/pack.py` | Bellek paketi: dışa aktarma, kurma, çelişkili birleştirme |
| `lmm/phrasing.py` | Türkçe yüzey dili (ünlü uyumu dahil) |
| `lmm/cli.py` | Sohbet oturumu |

## Test

```bash
python3 -m unittest discover -s tests
```

## Dokümanlar

- Tasarım: [docs/superpowers/specs/2026-07-31-lmm-living-memory-model-design.md](docs/superpowers/specs/2026-07-31-lmm-living-memory-model-design.md)
- Alan araştırması ve boşluk analizi: [docs/arastirma/2026-07-31-arastirma-haritasi.md](docs/arastirma/2026-07-31-arastirma-haritasi.md)
