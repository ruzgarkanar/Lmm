"""Tamamı öğrenilmiş okuma yolu: kalıp yok, ek yok, şablon yok.

Bu dosya, bu projenin en uzun süren tartışmasının sonucu. Sistem bugüne kadar
cümleyi elle yazılmış kalıplarla okuyordu ve o kalıplar `lmm/turkish.py`'deki
484 ögeye — eklere, bağlaçlara, soru sözcüklerine — bağlıydı. Bir dil
modelinde bunların hiçbiri yok: okuma da veriden öğrenilmiş.

Burada okuma tek bir ağdan çıkıyor ve ağ üç şeyi birden söylüyor:

    cümle  ->  (kavram, ilişki, hedef)  ->  graf  ->  cevap

Üçü de öğrenildi. Kavram ve hedef harf düzeyinde etiketleme (166.392 örnek,
etiketler hizalamayla), ilişki cümlenin tamamından (48 sınıf). Alfabe bile
bildirilmedi, sayıldı: 965 harf.

DENETİM DURUYOR. Ağın verdiği okuma grafta cevap üretmiyorsa atılıyor.
Öğrenilmiş okuma, denetimsiz okuma demek değil — uydurma niyet uydurma cevaba
açılan kapıdır ve bu mimaride o kapı kapalı kalmalı.

BEDELİ ÖLÇÜLDÜ VE KABUL EDİLDİ. Kalıplar kapatılınca gerçek sorularda %50,5
-> %44,5, olcut'ta %82,5 -> %64,2. Kayıp gerçek. Ama kalıplarla devam etmek,
öğrenmeyen bir sistemi öğreniyormuş gibi göstermekti; düşük sayı, yanlış
mimariden iyidir ve eğitimle yükselir.
"""
import os


class Learned:
    """Cümleden olguya, yalnız eğitilmiş ağlarla."""

    def __init__(self, folder="models/etiketci2"):
        self.ready = False
        self.kinds = []
        path = os.path.join(folder, "etiketci.pt")
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
        self.kinds = held.get("kinds", [])
        self.torch = torch

        class Tagger(nn.Module):
            def __init__(self, config, kinds):
                super().__init__()
                self.core = Core(config)
                self.head = nn.Linear(config.dimensions, 3)
                self.kind_head = nn.Linear(config.dimensions, kinds)

            def forward(self, ids):
                x = self.core.token(ids)
                for block in self.core.blocks:
                    x = block(x, causal=False)
                x = self.core.final(x)
                return self.head(x), self.kind_head(x.mean(dim=1))

        self.model = Tagger(config, len(self.kinds))
        self.model.load_state_dict(held["model"])
        self.model.eval()
        self.round = held.get("tur")
        self.ready = True

    def read(self, sentence):
        """(kavram, ilişki, hedef). Okuyamazsa üçü de boş."""
        if not self.ready or not sentence:
            return "", "", ""
        from core.tagger import spans_from
        torch = self.torch
        ids = torch.tensor([[self.letters.get(ch, 0) for ch in sentence]])
        with torch.no_grad():
            spans, kinds = self.model(ids)
        marks = spans.argmax(-1)[0].tolist()
        concept, target = spans_from(sentence, marks[:len(sentence)])
        relation = self.kinds[int(kinds.argmax(-1)[0])] if self.kinds else ""
        return concept, relation, target


def load(folder="models/etiketci2"):
    """Öğrenilmiş okuyucu. Model yoksa None."""
    found = Learned(folder)
    return found if found.ready else None
