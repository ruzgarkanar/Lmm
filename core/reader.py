"""Tamamı ÖĞRENİLMİŞ okuyucu: niyeti de kavramı da ağ söyler.

Bugüne kadar okuma ikiye bölünmüştü. Eğitilmiş ağ niyet TÜRÜNÜ veriyordu
("bu bir tanım sorusu"), ama cümlenin hangi parçasının kavram olduğunu
kurallı yol buluyordu — ek soyarak, `lmm/turkish.py`'deki 484 elle yazılmış
ögeye bakarak. O dosyanın yaşamasının tek sebebi buydu.

Burada ikisi birleşiyor ve elle yazılmış hiçbir şey kalmıyor:

    niyet      core/intent.py    29.397 gerçek soruyla eğitildi
    kavram     core/tagger.py    166.392 örnekle, harf düzeyinde

İkisi de veriden. Ne ek listesi var, ne kelime sınıfı, ne kalıp. Alfabe bile
sayılarak bulundu (965 harf) — bildirilen tek şey yok.

DENETİM DEĞİŞMİYOR. Ağın söylediği okuma yine grafta cevaplanabiliyor mu diye
sınanıyor ve cevap üretemiyorsa atılıyor. Öğrenilmiş okuma, denetimsiz okuma
demek değil: uydurma niyet, uydurma cevaba açılan kapıdır.

`lmm/` paketinin dışında duruyor. Bellek, muhakeme ve epistemik kapı hiçbir
zaman torch'a muhtaç olmamalı — bu projenin ana iddiası tam olarak bu.
"""
import os

from core.tagger import spans_from


class Tagged:
    """Eğitilmiş etiketleyici. Model yoksa `ready` False ve sistem eskisi gibi."""

    def __init__(self, folder="models/etiketci"):
        self.ready = False
        self.model = None
        # A100 koşusundan gelen dosya varsa o; yoksa yerel eğitimin çıktısı.
        path = next((os.path.join(folder, name)
                     for name in ("etiketci-a100.pt", "etiketci.pt")
                     if os.path.exists(os.path.join(folder, name))), None)
        if path is None:
            return
        try:
            self._load(path)
        except Exception:                                   # noqa: BLE001
            self.ready = False

    def _load(self, path):
        import torch
        from torch import nn
        from core.model import Config, Core

        held = torch.load(path, map_location="cpu", weights_only=False)
        config = Config(**held["config"])
        self.letters = held["letters"]
        # İlişki sınıfları eğitim verisinden geldi, elle yazılmadı: okuma
        # çıktısında hangi ilişkiler varsa onlar. İkinci sürüm modeli bunları
        # da veriyor; eski model dosyasında alan yoksa boş kalır ve `read`
        # ilişkisiz döner — geriye uyum, sessiz kırılma değil.
        self.kinds = held.get("kinds") or []
        self.torch = torch

        class Tagger(nn.Module):
            def __init__(self, config, kinds):
                super().__init__()
                self.core = Core(config)
                self.head = nn.Linear(config.dimensions, 3)
                if kinds:
                    self.kind_head = nn.Linear(config.dimensions, kinds)

            def forward(self, ids):
                x = self.core.token(ids)
                for block in self.core.blocks:
                    # Çift yönlü: bir cümleyi anlamak için sondaki eki görmek
                    # gerekiyor. Üretimde nedensellik şart, anlamada tersi.
                    x = block(x, causal=False)
                x = self.core.final(x)
                spans = self.head(x)
                if hasattr(self, "kind_head"):
                    return spans, self.kind_head(x.mean(dim=1))
                return spans, None

        self.model = Tagger(config, len(self.kinds))
        self.model.load_state_dict(held["model"])
        self.model.eval()
        self.round = held.get("tur")
        self.score = held.get("isabet", 0.0)
        self.ready = True

    def read(self, sentence):
        """(kavram, hedef, ilişki) — bulamazsa ("", "", None)."""
        if not self.ready or not sentence:
            return "", "", None
        torch = self.torch
        ids = torch.tensor([[self.letters.get(ch, 0) for ch in sentence]])
        with torch.no_grad():
            spans, kind_out = self.model(ids)
            marks = spans.argmax(-1)[0].tolist()
            kind = None
            if kind_out is not None and self.kinds:
                kind = self.kinds[int(kind_out.argmax())]
        concept, target = spans_from(sentence, marks[:len(sentence)])
        return concept, target, kind


class IntentNet:
    """A100'de eğitilen niyet okuyucusu — TEACH sınıflarını bilen ilk ağ.

    Eski okuyucu (`core/intent.py`) yalnız soru sınıflarını biliyordu; öğretme
    cümlesi gösterilince en yakın soruya zorluyordu. Bu ağ 178.644 gerçek
    cümleyle, 14 sınıfla eğitildi — 9 soru + 5 öğretme. Harf düzeyinde:
    parçalayıcı da yok, alfabe sayılarak bulundu.
    """

    def __init__(self, folder="models/intent"):
        self.ready = False
        path = os.path.join(folder, "niyet-a100.pt")
        if not os.path.exists(path):
            return
        try:
            self._load(path)
        except Exception:                                   # noqa: BLE001
            self.ready = False

    def _load(self, path):
        import torch
        from torch import nn
        from core.model import Config, Core

        held = torch.load(path, map_location="cpu", weights_only=False)
        config = Config(**held["config"])
        self.letters = held["letters"]
        self.labels = held["labels"]
        self.torch = torch

        class Net(nn.Module):
            def __init__(self, config, classes):
                super().__init__()
                self.core = Core(config)
                self.head = nn.Linear(config.dimensions, classes)

            def forward(self, ids):
                x = self.core.token(ids)
                for block in self.core.blocks:
                    x = block(x, causal=False)
                return self.head(self.core.final(x).mean(dim=1))

        self.model = Net(config, len(self.labels))
        self.model.load_state_dict(held["model"])
        self.model.eval()
        self.score = held.get("isabet", 0.0)
        self.ready = True

    def read(self, sentence):
        """(sınıf, güven) — hazır değilse (None, 0.0)."""
        if not self.ready or not sentence:
            return None, 0.0
        torch = self.torch
        ids = torch.tensor([[self.letters.get(ch, 0)
                             for ch in sentence[:256]]])
        with torch.no_grad():
            shares = torch.softmax(self.model(ids), dim=-1)[0]
        best = int(shares.argmax())
        return self.labels[best], float(shares[best])


def load_intent(folder="models/intent"):
    """Niyet ağını kurar. Yoksa None — kolaylık, bağımlılık değil."""
    found = IntentNet(folder)
    return found if found.ready else None


def load(folder="models/etiketci"):
    """Etiketleyiciyi kurar. Yoksa None — kolaylık, bağımlılık değil."""
    found = Tagged(folder)
    return found if found.ready else None
