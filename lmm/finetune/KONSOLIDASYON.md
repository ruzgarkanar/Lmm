# Uyku-Konsolidasyon Döngüsü — tasarım

Beynin complementary learning systems'inin yapay hâli: **graf = hipokampus**
(hızlı, sürekli, retrain'siz), **periyodik LoRA = korteks/uyku** (yavaş, toplu
konsolidasyon). Naif "konuşarak ağırlık güncelle" unutma/kararsızlık/zehir
yüzünden bozulur (v2 dersi); bu döngü aynı kazancı güvenle verir.

## Döngü (bir "gün")

```
YAŞA          python3.11 -m lmm.chat
              → graf büyür (kapıdan), her tur logs/*.jsonl'e
                "learned" (KAPI-ONAYLI üçlüler) yazılır
   ↓
HASAT ET      python3.11 -m lmm.finetune.consolidate
              → günlükler → data/train/consolidate.jsonl
              → birikim < --min-turns ise "eğitime değmez" der, çıkar
   ↓
EĞİT (İNSAN)  python3.11 -m lmm.finetune.train --data ... --out models/lmm/lora_next
              → yerel M-serisi gece, ya da A100 (bkz. A100.md)
   ↓
DOĞRULA       scratchpad regresyonu (sınıflandırıcılar + extraction + A/B sohbet)
              → v2 dersi: doğrulamadan ASLA değiştirme
   ↓
DEĞİŞTİR      lora_next → models/lmm/lora   (eskisini lora_prev diye sakla)
              → istenirse merge + GGUF tazele (ucuz-CPU dağıtımı)
```

## Güvenlik ilkeleri (pazarlık yok)

1. **Model kendi ham çıktısından öğrenmez.** Hasat edilen tek hedef, kapının
   o tur GERÇEKTEN kabul ettiği üçlülerdir (`Session.last_written` →
   günlükteki `learned`). Bozuk cevaplar ("Penguene...") asla hedef olmaz.
2. **Eğitimi İNSAN başlatır** (yarı-otonom guardrail'ın aynısı): consolidate
   yalnız veriyi hazırlar ve komutu ÖNERİR. Otomatik tetik yok.
3. **Doğrulamadan değiştirme yok.** Yeni adapter önce regresyondan geçer;
   geçemezse `lora_next` silinir, motor eskisiyle devam eder.
4. **Bilgi grafta kalır** (condition-3): bu döngü BİLGİ öğretmez, ARAYÜZ
   davranışını (extraction/recall/dil-tutarlılığı) tazeler. Graf olmadan model
   olguları gene bilmez — split bozulmaz.

## Tetik politikası

Şimdilik: kullanıcı ara sıra `consolidate` koşar; `--min-turns` (varsayılan 20)
altındaysa döngü "uykuya değmez" der. İleride `mind.status()`'a bakan bir
öneri eklenebilir ("motor-graf açığı büyüdü — konsolidasyon zamanı"), yine
insan-onaylı.

## Bilinen v4 hedefleri (bu döngünün ilk gündemi)

- Latin-alfabeli dil sızıntısı (naber→Lehçe): make_data dil filtresi yalnız
  CJK süzüyor — Latin-dışı-Türkçe tespiti eklenecek.
- Öğrenme onayı üslubu ("Benim için bu bilgiye ulaştım") — çeşitli gerçek
  onay örnekleri.
- Değerlerde kopula eki ("içecektir" — lemma "içecek" hedeflenmeli; şimdilik
  link.resolve simetrik çekim eşlemesi telafi ediyor).
- extraction: tür-dışı yüklemler ("Nortlann adasında çıkarılır" kaçıyor),
  çok-sözcüklü özne taşması ("zerbalit mavi renklidir" → özne=cümle).
- "olur" gibi kısa onaylar (is_affirmative), kısa "neden" soruları
  (is_causal_question).
