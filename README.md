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
