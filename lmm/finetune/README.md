# LoRA ince-ayar (opsiyonel geliştirme)

**Durum:** çekirdek LMM bu adım OLMADAN çalışır. Bu, Qwen-3B'nin Türkçe'de ara
sıra yaşadığı örnekleme sapmasını (bkz. `docs/LMM.md` §sınırlar) azaltmak için
bir **davranış** ince-ayarıdır. Otonom olarak KOŞULMADI — çok saatlik, doğrulama
ister; hazır bırakıldı, sen başlat.

## Ne öğretir (ve ne öğretmez)

Öğretir: **davranış** — "yalnız enjekte edilen olgudan konuş, olgu yoksa reddet,
kimliğini graftan söyle". Türkçe tutarlılığı artar.

Öğretmez: **bilgi**. Yeni gerçekler hâlâ GRAFA girer, retrain'siz (condition-3).
LoRA sadece grafın DAHA TUTARLI kullanılmasını sağlar. Bu yüzden veri de elle
yazılmaz — hedef cümleler ya gerçek veriden ya sistemin şu anki doğru
davranışının **öz-damıtımından** gelir (condition-5 korunur).

## Adımlar

```bash
# 0) bağımlılık
python3.11 -m pip install peft datasets

# 1) veri üret (graftan/gerçek veriden; Qwen öz-damıtımı — yavaş, küçük tut)
python3.11 -m lmm.finetune.make_data --limit 600 --out data/train/lora.jsonl

# 2) eğit (Apple M5 / MPS)
python3.11 -m lmm.finetune.train --data data/train/lora.jsonl --epochs 1

# 3) çıktı: models/lmm/lora/  (adapter)
```

## Donanım / süre

- **M5 (MPS), fp16, rank 16, ~1000 örnek, 1 epoch:** kabaca 1–3 saat. İlk kez
  dene, kaybını izle.
- **Bellek dararsa (MPS OOM):** `--rank 8 --batch 1 --grad-accum 16 --max-len 384`.
- **FALLBACK (MPS derleme/çökme):** `--device cpu` — çok yavaş ama garanti biter;
  ya da CUDA'lı bir kutu.

## Adapter'ı kullanmak

`lmm/runtime.py` `_load()` içine, model yüklendikten sonra:

```python
from peft import PeftModel
adapter = os.path.join(_root(), "models/lmm/lora")
if os.path.exists(adapter):
    _MODEL = PeftModel.from_pretrained(_MODEL, adapter)
```

GGUF yolu için: adapter'ı base'e **merge** edip (`model.merge_and_unload()`)
tekrar GGUF'a dönüştür (bkz. kök dizindeki dönüşüm adımları), sonra
`LMM_BACKEND=gguf` yine int4/CPU hızında ama ince-ayarlı çalışır.

## Değerlendirme

Eğitim sonrası `docs/LMM.md`'deki kabul senaryolarını tekrar koştur:
öğren→sor→ret→çelişki→kimlik. Beklenen: Türkçe "seni kim yaptı" ve "X nedir"
cevapları tutarlı; uydurma hâlâ 0 (verify kapısı zaten koruyor, LoRA onu
GEVŞETMEZ — yalnız modeli kapıya daha uyumlu yapar).
