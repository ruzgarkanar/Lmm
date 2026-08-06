"""Eğitilmiş niyet okuyucusu — kalıpların yedeği, yerine geçeni değil.

Bu dosya bilerek `lmm/` paketinin dışında: `lmm/` sıfır bağımlılıkla çalışır ve
öyle kalmalı. Bilgi, muhakeme ve epistemik kapı hiçbir zaman torch'a muhtaç
olmamalı — projenin ana iddiası tam olarak bu.

Ne için var: elle yazılmış kalıpların kapsamı ölçüldü ve tavanı belli
(`docs/DENEMELER.md`). Kalıp eklemek yakınsamıyor; söyleyiş uzayı çarpımsal.
Eğitilmiş ağ o boşluğu dolduruyor — 29.397 gerçek Türkçe soruyla eğitildi ve
280 gerçek soruluk sınavda %66-69 veriyor.

Ama **yerine geçmiyor, yedek duruyor** ve sırası önemli:

    1. kalıp eşleşirse       hızlı, kesin, künyeli — ağa hiç sorulmaz
    2. eşleşmezse            ağ niyeti tahmin eder
    3. tahmin DENETLENİR     çıkan niyet grafta gerçekten cevaplanıyor mu
    4. cevap üretemezse      tahmin atılır, sistem bilmediğini söyler

Üçüncü adım olmadan bu bağlantı bir gerileme olurdu: ağ her cümleye bir niyet
uydurabilir ve uydurma niyet, uydurma cevaba açılan kapıdır. Denetimle birlikte
ağ yalnızca **kapsamı** genişletiyor, doğruluğu değil.

Model yoksa hiçbir şey olmaz: sistem kalıplarla çalışmaya devam eder.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PIECES = os.path.join(ROOT, "data", "tr-parcalayici.model")
READY = os.path.join(HERE, "turkce-cekirdek")

# Yollar künyeden geliyor, elle yazılmıyor: hangi sürümün güncel olduğu tek
# bir yerde duruyor (`models/registry.json`) ve yeni bir sürüme geçmek o
# dosyadaki tek satır. Bkz. `lmm/registry.py`.
_FOLDER = os.path.join(ROOT, "models")
_FALLBACK = {
    "core": os.path.join(_FOLDER, "core", "16m-8k.pt"),
    "intent": os.path.join(_FOLDER, "intent", "16m.pt"),
    "classes": os.path.join(_FOLDER, "intent", "classes.json"),
    "graph": os.path.join(_FOLDER, "graph", "base.lmm"),
}


def _where(kind, name=None):
    """`models/registry.json` künyesinden yol; künye yoksa yedek yol."""
    try:
        with open(os.path.join(_FOLDER, "registry.json"),
                  encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:                                       # noqa: BLE001
        manifest = {}
    found = manifest.get(kind, {})
    chosen = name or found.get("current")
    path = found.get("versions", {}).get(chosen, {}).get("path") if chosen else None
    if path:
        path = path if os.path.isabs(path) else os.path.join(_FOLDER, path)
    else:
        path = _FALLBACK.get(kind)
    return path if path and os.path.exists(path) else None

# Kaç parça okunur. Kesme SONDAN değil baştan yapılırsa Türkçe'de taşıyıcı
# bilgi gider: soru eki ve çekimli yüklem cümlenin sonundadır. Ölçüldü —
# 32'de kesilen gerçek soruların %44,3'ünde soru işareti tamamen kayboluyor,
# yani tüm soruların %7,6'sı ağa hiçbir soru sinyali taşımadan giriyordu.
#
# 64, gerçek web sorularının %96,6'sını kapsıyor (32 ise %82,8). Eğitim verisi
# 32'yi zorlamadığı için tek başına yükseltmek yetmez; asıl düzeltme kesmenin
# YÖNÜ.
LONGEST = 64
OUTSIDE = "DISARIDA"

# Ağın seçebileceği sınıflardan sistemin niyet biçimine. Kapalı bir eşleme:
# ağ yeni bir niyet icat edemez, yalnızca var olanlardan birini gösterebilir.
READINGS = {
    "ASK_DEFINITION": ("ASK", "type"),
    "ASK_ABILITY": ("ASK", "can"),
    "ASK_PROPERTY": ("ASK", "property"),
    "ASK_ABILITIES": ("ASK_ABILITIES", None),
    "ASK_PROPERTIES": ("ASK_PROPERTIES", None),
    "ASK_DESCRIBE": ("ASK_DESCRIBE", None),
    "ASK_WHY": ("ASK_WHY", "can"),
    "ASK_WHO": ("ASK_WHO", "can"),
}


class Reader:
    """Cümleden niyet sınıfı. Yüklenemezse `ready` False kalır ve çağıran
    bunu bilir — sessizce yanlış cevap vermez."""

    def __init__(self, backbone="cekirdek", folder=None):
        self.ready = False
        self.backbone = backbone
        name = {"hazir": "ready-355m"}.get(backbone)
        path = _where("intent", name)
        labels = _where("classes")
        if not (path and labels):
            return
        try:
            self._load(path, labels)
            self.ready = True
        except Exception:                                   # noqa: BLE001
            self.ready = False      # torch yoksa ya da dosya bozuksa: kalıplar

    def _load(self, path, labels_path):
        import torch
        with open(labels_path, encoding="utf-8") as handle:
            self.labels = json.load(handle)
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if self.backbone == "hazir":
            from transformers import (AutoTokenizer,
                                      AutoModelForSequenceClassification)
            self.tokenizer = AutoTokenizer.from_pretrained(READY)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            model = AutoModelForSequenceClassification.from_pretrained(
                READY, num_labels=len(self.labels))
            model.config.pad_token_id = self.tokenizer.pad_token_id
            model.load_state_dict(saved["model"])
        else:
            import sentencepiece as spm
            import torch.nn as nn
            from core.model import Config, Core
            self.pieces = spm.SentencePieceProcessor(model_file=PIECES)
            # Çekirdeğin boyutları SINIFLANDIRICI kaydından okunuyor, ayrı
            # bir dosyadan değil. Colab'da daha büyük bir çekirdek eğitilirse
            # (farklı `dimensions`/`layers`) buradaki yükleme sessizce yanlış
            # boyutla model kurup çöküyordu — kayıt zaten kendi ayarını
            # taşıyor, sorulması yeterliydi.
            ayar = saved.get("config")
            if ayar is None:
                core_at = _where("core")
                core_saved = torch.load(core_at, map_location="cpu",
                                        weights_only=False)
                ayar = core_saved["config"]
            config = Config(**ayar) if isinstance(ayar, dict) else ayar

            class Classifier(nn.Module):
                def __init__(self, core, classes):
                    super().__init__()
                    self.core = core
                    self.drop = nn.Dropout(0.1)
                    self.head = nn.Linear(core.config.dimensions, classes)

                def forward(self, tokens, mask):
                    x = self.core.drop(self.core.token(tokens))
                    for block in self.core.blocks:
                        x = block(x)
                    x = self.core.final(x)
                    weight = mask.unsqueeze(-1).float()
                    pooled = (x * weight).sum(1) / weight.sum(1).clamp(min=1)
                    return self.head(self.drop(pooled))

            model = Classifier(Core(config), len(self.labels))
            model.load_state_dict(saved["model"])
        model.to(self.device).eval()
        self.model = model
        self.torch = torch

    def read(self, sentence):
        """(sınıf, güven). Model yoksa ya da kapsam dışıysa (None, 0.0)."""
        if not self.ready:
            return None, 0.0
        torch = self.torch
        with torch.no_grad():
            if self.backbone == "hazir":
                found = self.tokenizer([sentence], truncation=True,
                                       max_length=LONGEST,
                                       padding="max_length",
                                       return_tensors="pt").to(self.device)
                logits = self.model(**found).logits
            else:
                # SONDAN kesiliyor: Türkçe'de yüklem ve soru eki sonda durur, baştan
                # kesmek cümlenin taşıdığı sinyali atmak demek.
                piece = self.pieces.encode(sentence)[-LONGEST:]
                pad = LONGEST - len(piece)
                ids = torch.tensor([piece + [0] * pad]).to(self.device)
                mask = torch.tensor([[1] * len(piece) + [0] * pad]).to(self.device)
                logits = self.model(ids, mask)
            shares = torch.softmax(logits, dim=-1)[0]
            best = int(shares.argmax())
        name = self.labels[best]
        if name == OUTSIDE or name not in READINGS:
            return None, float(shares[best])
        return name, float(shares[best])


    def read_many(self, sentences, batch=256):
        """Yığın hâlinde okuma — [(sınıf, güven)].

        Tek tek okumak süzgeç için kullanılamaz: 4,2 milyon soruyu tek tek
        geçirmek saatler sürüyor. Yığın, aynı işi onlarca kat hızlı yapıyor.
        """
        if not self.ready:
            return [(None, 0.0)] * len(sentences)
        torch = self.torch
        found = []
        with torch.no_grad():
            for start in range(0, len(sentences), batch):
                piece = sentences[start:start + batch]
                if self.backbone == "hazir":
                    encoded = self.tokenizer(piece, truncation=True,
                                             max_length=LONGEST,
                                             padding="max_length",
                                             return_tensors="pt").to(self.device)
                    logits = self.model(**encoded).logits
                else:
                    ids, masks = [], []
                    for sentence in piece:
                        got = self.pieces.encode(sentence)[:LONGEST]
                        pad = LONGEST - len(got)
                        ids.append(got + [0] * pad)
                        masks.append([1] * len(got) + [0] * pad)
                    logits = self.model(torch.tensor(ids).to(self.device),
                                        torch.tensor(masks).to(self.device))
                shares = torch.softmax(logits, dim=-1)
                best = shares.argmax(-1)
                for index in range(len(piece)):
                    name = self.labels[int(best[index])]
                    strength = float(shares[index][int(best[index])])
                    found.append((None if name == OUTSIDE
                                  or name not in READINGS else name, strength))
        return found


def reading_of(name, has_target=True):
    """Sınıf adından (niyet türü, ilişki). Tanınmayan ad için None.

    Hedef yoksa tekil soru çoğula düşer: ağ "ASK_PROPERTY" diyor ama HANGİ
    nitelik olduğunu söylemiyor ve graf hedefsiz bir nitelik sorusunu
    cevaplayamıyor. "kartal nasıl bir hayvan" sorusunun makul okuması
    "niteliklerini say"dır — uydurmak değil, elindekini vermek.
    """
    found = READINGS.get(name)
    if found is None:
        return None
    if not has_target:
        found = WITHOUT_TARGET.get(name, found)
    return found


# Hedefsiz kaldığında hangi okumaya düşülür.
WITHOUT_TARGET = {
    "ASK_PROPERTY": ("ASK_PROPERTIES", None),
    "ASK_ABILITY": ("ASK_ABILITIES", None),
}
