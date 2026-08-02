"""Hazır Türkçe çekirdeği LMM'in kapısına bağlayan katman.

Kendi çekirdeğimizi eğitmek yerine önce mimariyi sınıyoruz, ve sebebi bu gece
öğrenildi: çekirdeği eğitip sonra kapıya bağladık, kapı üç yerden sızdı.
Mimariyi önce sınamak o hataları saatler önce gösterirdi.

Kullanılan model YTÜ COSMOS'un Türkçe GPT-2'si (350M, MIT lisanslı, talimat
ayarlı). Bu bir tercih değil bir **ölçüm aracı**: soru şu — graf + kapı + akıcı
bir çekirdek gerçekten iyi bir sohbet üretiyor mu? Cevap evetse kendi
çekirdeğimizi neyin için eğittiğimizi bilerek eğitiriz; hayırsa mimariyi
düzeltiriz ve hiç GPU harcamamış oluruz.

LMM'in beş iddiasının hiçbiri bu değişiklikle bozulmuyor. Bilgi yine grafta,
uydurma yine kapıda engelleniyor, düzeltme yine tek satır, büyüme yine yeniden
eğitim istemiyor, çalışma yine CPU'da. Çekirdek beyin değil ağız — ne
söyleneceğine graf karar veriyor, o yalnızca cümleyi kuruyor.
"""
import os

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "turkce-cekirdek")


class Pieces:
    """HuggingFace parçalayıcısını bizim arayüzümüzle sunar.

    Kapı ve köprü sentencepiece'in imzasını bekliyor (`eos_id()`,
    `get_piece_size()`), hazır model başkasını sunuyor. Farkı burada kapatmak,
    kapının hangi parçalayıcıyla çalıştığını bilmemesini sağlıyor — yarın
    kendi çekirdeğimize dönersek kapı değişmez.
    """

    def __init__(self, inner):
        self.inner = inner

    def encode(self, text):
        return self.inner.encode(text)

    def decode(self, identifiers):
        return self.inner.decode(identifiers, skip_special_tokens=True)

    def get_piece_size(self):
        return len(self.inner)

    def id_to_piece(self, identifier):
        return self.inner.convert_ids_to_tokens(identifier)

    def eos_id(self):
        return self.inner.eos_token_id

    def bos_id(self):
        return self.inner.bos_token_id if self.inner.bos_token_id is not None \
            else self.inner.eos_token_id

    def pad_id(self):
        return self.inner.pad_token_id if self.inner.pad_token_id is not None \
            else self.inner.eos_token_id


class TurkishCore:
    """Hazır bir dil modelini, kendi çekirdeğimizin arayüzüyle sunar.

    Köprü `continue_from(tokens, allowed=...)` bekliyor; hazır model başka bir
    arayüz sunuyor. Aradaki farkı burada kapatmak, köprünün hangi modelle
    çalıştığını bilmemesini sağlıyor — kapı da öyle. Model değiştirilebilir
    kalıyor.
    """

    def __init__(self, path=MODEL, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device is None:
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.pieces = Pieces(AutoTokenizer.from_pretrained(path))
        self.model = AutoModelForCausalLM.from_pretrained(path).to(device)
        self.model.eval()
        self.torch = torch

    @property
    def size(self):
        return sum(p.numel() for p in self.model.parameters())

    @property
    def context(self):
        return self.model.config.n_positions

    def encode(self, text):
        return self.pieces.encode(text)

    def decode(self, identifiers):
        return self.pieces.decode(identifiers)

    def continue_from(self, tokens, length=60, temperature=0.8, top_k=40,
                      allowed=None):
        """Kendi çekirdeğimizle aynı imza. `allowed` verilirse kapı devrede.

        Kısıtlama üretim sırasında uygulanıyor, sonradan değil: izin verilmeyen
        parçanın olasılığı eksi sonsuz. Bu gece ölçtük, parça düzeyinde
        kısıtlama tek başına yetmiyor — köprü ayrıca cümle düzeyinde denetliyor.
        """
        torch = self.torch
        with torch.no_grad():
            for _ in range(length):
                window = tokens[:, -self.context:]
                logits = self.model(window).logits[:, -1, :]
                logits = logits / max(temperature, 1e-6)
                if allowed is not None:
                    blocked = torch.full_like(logits, float("-inf"))
                    blocked[:, allowed] = logits[:, allowed]
                    logits = blocked
                if top_k:
                    cut = min(top_k, logits.shape[-1])
                    threshold = torch.topk(logits, cut).values[:, [-1]]
                    logits = logits.masked_fill(logits < threshold,
                                                float("-inf"))
                probabilities = torch.softmax(logits, dim=-1)
                following = torch.multinomial(probabilities, num_samples=1)
                tokens = torch.cat((tokens, following), dim=1)
                if following.item() == self.pieces.eos_id():
                    break
        return tokens


def available(path=MODEL):
    """Model indirilmiş mi? Eksikse sistem çekirdeksiz çalışmaya devam eder."""
    needed = ("config.json", "model.safetensors")
    return all(os.path.exists(os.path.join(path, name)) for name in needed)
