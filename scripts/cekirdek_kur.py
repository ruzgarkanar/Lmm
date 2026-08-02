"""Colab'da eğitilen çekirdeği ve niyet ağını yerine koyar.

Eğitim uzakta yapıldı ama çalışma yerelde ve GPU'suz — projenin ana iddiası bu.
Bu betik iki dosyayı alıp yerine koyuyor ve **koymadan önce denetliyor**, çünkü
yanlış bir dosyayı yerine koymak sessizce çalışan ama saçmalayan bir sistem
bırakır.

Denetimler:

    yapı        kayıt bir `config` taşıyor mu, ağırlıklar o ayara uyuyor mu
    boyut       modelin gerçek parametre sayısı ayarla tutarlı mı
    yükleme     model gerçekten kurulabiliyor mu
    üretim      birkaç cümle üretip gözle bakılabiliyor mu
    niyet       sınıflandırıcı yükleniyor ve gerçek soruları okuyabiliyor mu

Eskisi silinmiyor, `.onceki` uzantısıyla saklanıyor: yeni çekirdek kötü çıkarsa
tek komutla geri dönülür.

Kullanım:
    python3.11 scripts/cekirdek_kur.py ~/Downloads/cekirdek.pt ~/Downloads/niyet.pt
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_AT = os.path.join(ROOT, "lmm-cekirdek.pt")
INTENT_AT = os.path.join(ROOT, "core", "niyet", "niyet-cekirdek.pt")
PIECES = os.path.join(ROOT, "data", "tr-parcalayici.model")

# Sürüm geçişlerinden kalan, artık kullanılmayan anahtarlar. Konum tablosu
# RoPE'a geçildiğinde kaldırıldı; eski kayıtlar hâlâ taşıyor ve bu bir hata
# değil.
LEGACY = ("position.weight",)

SAMPLES = ("Penguen bir", "Kartal", "Türkiye'nin başkenti", "Bilgisayar")
QUESTIONS = ("kartal nedir", "penguen uçar mı", "kuşlar ne yapabilir",
             "bana penguenlerden bahseder misin")


def check_core(path):
    """Çekirdek kaydı sağlam mı ve ne kadar büyük."""
    import torch
    from core.model import Config, Core
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in saved or "model" not in saved:
        return None, "kayıtta 'config' ya da 'model' yok"
    setting = saved["config"]
    config = Config(**setting) if isinstance(setting, dict) else setting
    core = Core(config)
    # Eskimiş anahtarlara izin var: konum tablosu RoPE'a geçilince kaldırıldı
    # ve eski kayıtlar hâlâ taşıyor. EKSİK anahtar ise gerçek bir uyumsuzluk —
    # o zaman model yarım yüklenir ve sessizce saçmalar.
    weights = {k: v for k, v in saved["model"].items() if k not in LEGACY}
    missing = core.load_state_dict(weights, strict=False)
    if missing.missing_keys:
        return None, f"eksik ağırlık: {missing.missing_keys[:3]}"
    if missing.unexpected_keys:
        return None, f"tanınmayan ağırlık: {missing.unexpected_keys[:3]}"
    return (core, config, saved.get("step", "?")), None


def show_samples(core, config):
    import torch
    import sentencepiece as spm
    pieces = spm.SentencePieceProcessor(model_file=PIECES)
    core.eval()
    for opening in SAMPLES:
        seed = torch.tensor([pieces.encode(opening)])
        with torch.no_grad():
            grown = core.continue_from(seed, length=30, temperature=0.8,
                                       top_k=40)
        print(f"    {opening!r} -> {pieces.decode(grown[0].tolist())[:90]}")


def keep_old(path):
    if os.path.exists(path):
        shutil.copy(path, path + ".onceki")
        return True
    return False


def main(argv):
    given = [a for a in argv[1:] if not a.startswith("--")]
    if not given:
        print(__doc__.strip().splitlines()[-1])
        return 1
    core_file = given[0]
    intent_file = given[1] if len(given) > 1 else None
    if not os.path.exists(core_file):
        print(f"yok: {core_file}")
        return 1

    print("ÇEKİRDEK DENETİMİ")
    found, why = check_core(core_file)
    if found is None:
        print(f"  REDDEDİLDİ — {why}")
        return 1
    core, config, step = found
    print(f"  {core.size/1e6:.1f}M parametre · {config.layers} katman · "
          f"{config.dimensions} boyut · {config.context} bağlam · adım {step}")
    print("  üretim denemesi:")
    show_samples(core, config)

    if "--onayla" not in argv:
        print("\n  Üretilen cümleler makul görünüyorsa --onayla ile tekrar çalıştır.")
        return 0

    print("\nKURULUM")
    if keep_old(CORE_AT):
        print(f"  eskisi saklandı: {os.path.basename(CORE_AT)}.onceki")
    shutil.copy(core_file, CORE_AT)
    print(f"  {CORE_AT}")

    if intent_file and os.path.exists(intent_file):
        os.makedirs(os.path.dirname(INTENT_AT), exist_ok=True)
        if keep_old(INTENT_AT):
            print(f"  eskisi saklandı: {os.path.basename(INTENT_AT)}.onceki")
        shutil.copy(intent_file, INTENT_AT)
        # sınıf adları kayıttan çıkarılıyor: ayrı dosya taşımak bir uyumsuzluk
        # kaynağı, kayıt zaten kendi etiketlerini taşıyor
        import torch
        saved = torch.load(intent_file, map_location="cpu", weights_only=False)
        if "labels" in saved:
            with open(os.path.join(os.path.dirname(INTENT_AT),
                                   "siniflar.json"), "w",
                      encoding="utf-8") as out:
                json.dump(saved["labels"], out, ensure_ascii=False)
        print(f"  {INTENT_AT}")

        print("\nNİYET DENETİMİ")
        from core.intent import Reader
        reader = Reader("cekirdek")
        if not reader.ready:
            print("  YÜKLENEMEDİ — eski hâle dönmek için .onceki dosyalarını geri koy")
            return 1
        for question in QUESTIONS:
            name, confidence = reader.read(question)
            print(f"    {question:<34} {str(name):<16} %{confidence*100:.0f}")

    print("\nBİTTİ. Denemek için:")
    print("  python3.11 -m lmm.cli model.lmm --anla")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
