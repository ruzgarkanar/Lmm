"""Okuyucu: cümleyi belleğin ANLAYACAĞI işleme çevirir — eğitilmiş, kuralsız.

Eski sistemde okuma üç parçaya bölünmüştü: önce cümleyi bir sınıfa ayır, sonra
kavramı bul, sonra grafa sor. Üçü ayrı organdı, ayrı hatalar yapıyorlardı ve
ortadaki sınıf etiketi hiçbir işe yaramayan bir ara duraktı.

Burada tek adım var ve bir dil modelinin yaptığı işin aynısı: metin girer,
YAPI çıkar. Aradaki fark, çıkan yapının ağırlıklara gömülmesi değil, belleğe
yazılacak bir işlem olması.

    "kartal bir kuştur"   ->  WRITE(kartal, tür, kuş)
    "kartal nedir"        ->  ASK(kartal, ?, ?)
    "selam"               ->  PASS

Üç işlem, üç sayı. Dil değil, mimarideki karşılıkları:

    PASS   bilgi taşımıyor — sohbeti sürdüren söz
    ASK    bellekten bir şey isteniyor
    WRITE  belleğe bir şey konuyor (kapıdan geçmek şartıyla)

Bu dosyada ağ EĞİTİLMEZ, yalnız YÜKLENİR ve çalıştırılır. Eğitim ayrı bir
tezgahın işi; buradaki tek sorumluluk, eğitilmiş ağı belleğin diline
bağlamak. Ağ yoksa `ready` False olur ve sistem susar — kolaylık,
bağımlılık değil: `v3/` torch'a muhtaç olmamalı.

Dile ait hiçbir şey yoktur: ne ek, ne kalıp, ne kelime listesi. Ağ harf
düzeyinde çalışır ve alfabesini kendi verisinden sayarak bulur.
"""
import os

PASS, ASK, WRITE = 0, 1, 2


class Operation:
    """Okuyucunun çıktısı: belleğe uygulanacak tek bir işlem.

    `subject`, `predicate`, `value` metin parçalarıdır — henüz kimlik değil.
    Onları kimliğe çevirmek geometrinin işi (aynı yazılış birden çok kavram
    olabilir ve hangisi olduğunu bağlam söyler).
    """

    __slots__ = ("kind", "subject", "predicate", "value", "confidence")

    def __init__(self, kind, subject=None, predicate=None, value=None,
                 confidence=0.0):
        self.kind = kind
        self.subject = subject
        self.predicate = predicate
        self.value = value
        self.confidence = confidence

    def __repr__(self):
        return (f"Operation({self.kind}, {self.subject!r}, "
                f"{self.predicate!r}, {self.value!r}, {self.confidence:.2f})")


class Reader:
    """Eğitilmiş okuyucu. Model yoksa `ready` False."""

    def __init__(self, folder="models/v3", name="reader.pt"):
        self.ready = False
        self.model = None
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            return
        try:
            self._load(path)
        except Exception:                                   # noqa: BLE001
            self.ready = False

    def _load(self, path):
        import torch
        from torch import nn

        held = torch.load(path, map_location="cpu", weights_only=False)
        self.torch = torch
        self.letters = held["letters"]
        width = self.width = held.get("width", 256)
        size = held.get("size", 256)
        layers = held.get("layers", 6)
        heads = held.get("heads", 8)

        class Net(nn.Module):
            """Harf dizisi -> (işlem türü, harf başına rol).

            İki başlık tek gövdeyi paylaşır: biri cümlenin ne yapmak
            istediğini, öteki hangi harflerin özne/yüklem/değer olduğunu
            söyler. Tek gövde, çünkü ikisi aynı okumanın iki yüzü.

            Konum tablosunun boyu KAYITTAN okunur, burada varsayılmaz. İlk
            yazışta 4096 sabitti ve eğitim 257 ile kaydetmişti; model
            yüklenemedi ve okuyucu sessizce hazır değil kaldı. Bir sayıyı iki
            yerde yazmak, er geç iki farklı sayı yazmaktır.
            """

            def __init__(self):
                super().__init__()
                self.token = nn.Embedding(len(held["letters"]) + 1, size)
                self.place = nn.Embedding(width + 1, size)
                block = nn.TransformerEncoderLayer(
                    size, heads, size * 4, batch_first=True,
                    norm_first=True, dropout=0.1)
                self.body = nn.TransformerEncoder(block, layers)
                self.final = nn.LayerNorm(size)
                self.kind_head = nn.Linear(size, 3)
                self.role_head = nn.Linear(size, 4)

            def forward(self, ids):
                steps = torch.arange(ids.shape[1],
                                     device=ids.device).unsqueeze(0)
                x = self.token(ids) + self.place(steps)
                x = self.final(self.body(x))
                return self.kind_head(x.mean(dim=1)), self.role_head(x)

        self.model = Net()
        self.model.load_state_dict(held["model"])
        self.model.eval()
        self.score = held.get("score", 0.0)
        self.ready = True

    def read(self, sentence):
        """Cümle -> Operation. Ağ yoksa PASS döner (susmak, uydurmaktan iyi)."""
        if not self.ready or not sentence:
            return Operation(PASS)
        torch = self.torch
        ids = torch.tensor([[self.letters.get(ch, 0)
                             for ch in sentence[:self.width]]])
        with torch.no_grad():
            kind_out, role_out = self.model(ids)
            kind = int(kind_out.argmax())
            confidence = float(torch.softmax(kind_out, dim=-1)[0].max())
            roles = role_out.argmax(-1)[0].tolist()
        subject, predicate, value = spans(sentence, roles)
        return Operation(kind, subject, predicate, value, confidence)


def spans(text, roles):
    """Harf rollerinden üç parçayı geri okur.

    Roller: 0 dışarısı · 1 özne · 2 yüklem · 3 değer. Her rol için en uzun
    kesintisiz dizi alınır ve kelime sınırına genişletilir — ağ ortada bir
    harf kaçırırsa parça yine bulunsun. Kelime sınırı bir DİL KURALI değil:
    boşluğun kelimeleri ayırdığını bilmek, Türkçe bilmeyi gerektirmiyor.
    """
    found = {}
    for role in (1, 2, 3):
        best, run, start = (0, 0), 0, None
        for at, one in enumerate(list(roles[:len(text)]) + [0]):
            if one == role:
                if start is None:
                    start = at
                run += 1
            else:
                if start is not None and run > best[1] - best[0]:
                    best = (start, at)
                start, run = None, 0
        if not best[1]:
            found[role] = None
            continue
        low, high = best
        while low > 0 and not text[low - 1].isspace():
            low -= 1
        while high < len(text) and not text[high].isspace():
            high += 1
        piece = text[low:high]
        # Elle noktalama kümesi yerine Unicode'un kendisi: baştan ve sondan,
        # harf/rakam olmayan her şey kırpılır. Küme yazmak, alfabe saymış
        # bir hattın içine ASCII varsayımı sokmaktı.
        start, stop = 0, len(piece)
        while start < stop and not piece[start].isalnum():
            start += 1
        while stop > start and not piece[stop - 1].isalnum():
            stop -= 1
        piece = piece[start:stop]
        from v3.dataset import fold
        found[role] = fold(piece) if piece else None
    return found[1], found[2], found[3]
