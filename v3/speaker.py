"""Konuşucu: kayıtlardan cümle kurar — eğitilmiş, şablonsuz.

Eski sistemde konuşmayı programcı yazmıştı: "{kavram} bir {hedef}dır". Bir
dil modelinde bunun karşılığı yoktur; orada cümle, kelime kelime, olasılıkla
üretilir ve her cümle yenidir. Bu dosya o mekanizmanın buradaki karşılığıdır.

    girdi   [ (kartal, tür, kuş) ‹hayvanlar.txt› , (kartal, özellik, hızlı) ]
    çıktı   "Kartal, hızlı bir kuştur."

Fark, girdinin nereden geldiğinde: bir dil modeli cümleyi ağırlıklarından
çıkarır ve neye dayandığı hesaplanamaz. Burada üretim KAYITLARA koşulludur ve
her cümlenin arkasında kayıt anahtarları durur.

İKİ KAPI. Üretim serbest değildir:

    ÖNCE   ağ yalnız kendisine verilen kayıtları görür — bilmediği bir şeyi
           kurgulayacak malzemesi yoktur
    SONRA  kurulan cümle geri okunur (`v3/reader.py`) ve çıkan işlemler
           kayıtlarla karşılaştırılır; tutmayan cümle DÜŞER

İkinci kapı olmadan akıcılık uydurmaya açılır. Bu mimaride akıcılık ile
doğruluk arasında takas yoktur: akıcı olmayan cümle atılır, yanlış olan da.

Bu dosyada ağ EĞİTİLMEZ, yalnız yüklenir. Ağ yoksa `ready` False olur ve
sistem ham kayıtları döker — konuşamaz ama yalan da söylemez.

Dile ait hiçbir şey yoktur: ne şablon, ne ek, ne kelime listesi.
"""
import os

# Bir cevapta en çok kaç aday üretilir. Tartım bunların arasından seçer; tek
# aday üretmek, düşünmeden konuşmaktır.
CANDIDATES = 4

# Üretimin durduğu uzunluk — harf. Ağ bitiş işaretini kendi öğrenir; bu yalnız
# sonsuz döngüye karşı emniyet.
LONGEST = 400


class Speaker:
    """Eğitilmiş konuşucu. Model yoksa `ready` False."""

    def __init__(self, folder="models/v3", name="speaker.pt"):
        self.ready = False
        self.model = None
        # Yol depo köküne göre: cwd'ye göre olunca kök dışından açılan
        # oturum ağı SESSİZCE bulamıyordu (3. tur gözlemi) — hatasız ama kör.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, folder, name)
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
        self.inverse = {value: key for key, value in self.letters.items()}
        size = held.get("size", 256)
        layers = held.get("layers", 6)
        heads = held.get("heads", 8)
        # Bitiş işareti kayıttan gelir ve 0 OLAMAZ: 0 dolgu/bilinmeyen-harf
        # kimliği — bitişle çakışırsa üretim ilk tanınmayan harfte kesilir.
        # Eğitim betiği bitişi len(letters)+1 olarak ayırmak zorunda.
        self.stop = held.get("stop")
        width = held.get("width", 2048)
        self.width = width

        class Net(nn.Module):
            """Kayıt dizisi + şu ana kadarki harfler -> sonraki harf.

            Nedensel: üretirken gelecek görülmez. Okuyucu çift yönlüydü
            çünkü anlamak için sonu görmek gerekiyor; üretmek tersidir.
            """

            def __init__(self):
                super().__init__()
                # +2: dolgu (0) ve BİTİŞ kimliği (len+1). Defter böyle eğitti;
                # +1 kuran yükleyici boyut uyuşmazlığıyla sessizce düşüyordu.
                self.token = nn.Embedding(len(held["letters"]) + 2, size)
                self.place = nn.Embedding(width + 1, size)
                block = nn.TransformerEncoderLayer(
                    size, heads, size * 4, batch_first=True,
                    norm_first=True, dropout=0.1)
                self.body = nn.TransformerEncoder(block, layers)
                self.final = nn.LayerNorm(size)
                self.head = nn.Linear(size, len(held["letters"]) + 2)

            def forward(self, ids, mask):
                import torch as _torch
                steps = _torch.arange(ids.shape[1],
                                      device=ids.device).unsqueeze(0)
                x = self.token(ids) + self.place(steps)
                x = self.body(x, mask=mask)
                return self.head(self.final(x))

        self.model = Net()
        self.model.load_state_dict(held["model"])
        self.model.eval()
        self.score = held.get("score", 0.0)
        self.ready = True

    def say(self, prompt, count=CANDIDATES, warmth=0.8):
        """Kayıt dizisinden aday cümleler üretir.

        `prompt`: kayıtların düz metin serimi — ağın gördüğü tek şey.
        Dönen: [cümle]. Aynı girdiden birden çok aday çıkar ve seçimi
        tartım yapar; bu, "böyle mi desem şöyle mi" adımının karşılığı.
        """
        if not self.ready or not prompt:
            return []
        torch = self.torch
        found = []
        for at in range(count):
            ids = [self.letters.get(ch, 0) for ch in prompt]
            start = len(ids)
            made = []
            with torch.no_grad():
                for step in range(LONGEST):
                    window = torch.tensor([ids[-self.width:]])
                    mask = torch.nn.Transformer.generate_square_subsequent_mask(
                        window.shape[1])
                    logits = self.model(window, mask)[0, -1]
                    # DÖNGÜ ENGELİ (no-repeat n-gram): üretilen kısımda son n-1
                    # harfle başlayan bir n-gram DAHA ÖNCE geçtiyse, onu
                    # tamamlayacak harfi yasakla. Az eğitilmiş ağ açgözlü
                    # üretimde "bir kaç tane bir kaç tane" gibi kilitleniyordu;
                    # bu, model hatası değil çözümleme hatası — burada kırılır.
                    for ban in self._loops(ids, start):
                        logits[ban] = float("-inf")
                    # İlk aday az ısıtılır (kararlı), sonrakiler çeşitlensin.
                    heat = 0.5 if at == 0 else max(warmth, 0.6)
                    probs = torch.softmax(logits / heat, dim=-1)
                    # TOP-P (çekirdek örnekleme): olasılığı toplam p'yi geçen en
                    # küçük kümeden seç. Greedy'nin döngüsünü de, düz ısıtmanın
                    # gürültü kuyruğunu da atlar — akıcı ama takılmayan üretim.
                    pick = self._nucleus(probs, 0.92)
                    if self.stop is not None and pick == self.stop:
                        break
                    ids.append(pick)
                    made.append(self.inverse.get(pick, ""))
            text = "".join(made).strip()
            if text and text not in found:
                found.append(text)
        return found

    def _loops(self, ids, start, size=10):
        """Üretilen dizide son (size-1) harfle başlayan bir n-gram tekrar
        ediyorsa, onu tamamlayacak harfleri döndürür — birebir döngü engeli.
        Yalnız ÜRETİLEN kısma bakar (girdi/prompt sayılmaz)."""
        made = ids[start:]
        if len(made) < size:
            return set()
        prefix = made[-(size - 1):]
        banned = set()
        for at in range(len(made) - (size - 1)):
            if made[at:at + size - 1] == prefix:
                banned.add(made[at + size - 1])
        return banned

    def _nucleus(self, probs, p):
        """Çekirdek (top-p) örnekleme: en olasıdan başlayıp toplam olasılık
        p'yi geçene dek biriktir, o kümeden örnekle."""
        torch = self.torch
        order = torch.argsort(probs, descending=True)
        keep, cum = [], 0.0
        for idx in order.tolist():
            keep.append(idx)
            cum += float(probs[idx])
            if cum >= p:
                break
        weights = torch.tensor([float(probs[i]) for i in keep])
        return keep[int(torch.multinomial(weights, 1))]


def prompt_of(records, memory):
    """Kayıtları ağın göreceği düz metne çevirir.

    Bu bir CÜMLE ŞABLONU DEĞİL, veri serimidir: ağın girdisi, insanın
    okuyacağı bir şey değil. Ayraçlar dil-bağımsız işaretler; hangi kelimenin
    nereye geleceğine ağ karar verir.
    """
    lines = []
    for record in records:
        subject = _label(memory, record.subject)
        predicate = _label(memory, record.predicate)
        value = _label(memory, record.value)
        lines.append(f"{subject}\t{predicate}\t{value}")
    return "\n".join(lines) + "\n\n"


def _label(memory, key):
    """Kimlik anahtarından ilk etikete. Kimlik değilse değerin kendisi."""
    # None yüklem "None" değil BOŞ olmalı: ağın girdisine "kalp\tNone\t..."
    # gibi çöp girip çıktıya "none" sızıyordu (ölçüldü).
    if key is None:
        return ""
    held = memory.identities.get(key) if isinstance(key, int) else None
    if held is not None and held.labels:
        return held.labels[0]
    return str(key)
