"""Dil çekirdeğini eğitir. Süreç görünür, kesilebilir ve kaldığı yerden devam eder.

Her ölçüm aralığında dört şey basılır: eğitim kaybı, doğrulama kaybı, hız ve
modelin o anda ürettiği bir örnek. Örnek en önemlisi — kaybın düşmesi soyut bir
sayıdır, ama ağın harf yığınından heceye, heceden kelimeye, kelimeden cümleye
geçişini gözle görmek eğitimin gerçekten yürüdüğünü anlamanın en dürüst yoludur.

Doğrulama kümesi baştan ayrılır ve eğitimde asla kullanılmaz. Yalnızca eğitim
kaybına bakmak, modelin ezberlemeye mi yoksa öğrenmeye mi başladığını gizler.

Kullanım:
    python3 scripts/cekirdek_egit.py                     # varsayılan
    python3 scripts/cekirdek_egit.py --adim 20000 --boyut 512 --katman 8
    python3 scripts/cekirdek_egit.py --devam             # kaldığı yerden
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings                                         # noqa: E402

import torch                                            # noqa: E402

warnings.filterwarnings("ignore", message=".*not writable.*")
import sentencepiece as spm                             # noqa: E402

from core.model import Core, Config                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from core.intent import _where                              # noqa: E402
TOKENS = os.path.join(ROOT, "data", "tr-parcalar.bin")
TOKENIZER = os.path.join(ROOT, "data", "tr-parcalayici.model")
CHECKPOINT = _where("core") or os.path.join(ROOT, "models", "core", "yeni.pt")

VALIDATION_SHARE = 0.005        # doğrulama için ayrılan pay
REPORT_EVERY = 100
SAMPLE_EVERY = 500
SAVE_EVERY = 1000
PROMPT = "Kartal, "


def device_of():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def synchronise(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class Corpus:
    """Belleğe sığmayan bir parça dizisinden rastgele pencereler verir."""

    def __init__(self, path, context, batch, device):
        # memmap salt okunur; torch bunu yazılabilir sanıp uyarıyor. Kopya
        # almıyoruz — 400 MB'ı belleğe çekmek anlamsız — yalnızca okunacağını
        # söylüyoruz.
        import numpy
        raw = torch.from_numpy(numpy.asarray(_read_tokens(path)).view(numpy.uint16))
        split = int(len(raw) * (1 - VALIDATION_SHARE))
        self.train = raw[:split]
        self.validate = raw[split:]
        self.context = context
        self.batch = batch
        self.device = device

    def draw(self, which="train"):
        data = self.train if which == "train" else self.validate
        start = torch.randint(len(data) - self.context - 1, (self.batch,))
        x = torch.stack([data[i:i + self.context].long() for i in start])
        y = torch.stack([data[i + 1:i + 1 + self.context].long() for i in start])
        return x.to(self.device), y.to(self.device)


def _read_tokens(path):
    import numpy
    return numpy.memmap(path, dtype=numpy.uint16, mode="r")


@torch.no_grad()
def estimate(model, corpus, rounds=20):
    """Gürültüyü azaltmak için birkaç toplu örnek üzerinden ortalama."""
    model.eval()
    losses = {}
    for which in ("train", "validate"):
        total = 0.0
        for _ in range(rounds):
            x, y = corpus.draw(which)
            _, loss = model(x, y)
            total += loss.item()
        losses[which] = total / rounds
    model.train()
    return losses


def learning_rate(step, total, peak, warmup):
    """Isınma, sonra kosinüs sönümü — küçük modellerde en güvenilir program."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= total:
        return peak * 0.1
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)) * 2 * 0.5)


def argument(argv, flag, fallback, cast=int):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else fallback


def main(argv):
    for path in (TOKENS, TOKENIZER):
        if not os.path.exists(path):
            print(f"eksik: {path}\nönce: python3 scripts/parcalayici_egit.py")
            return 1

    steps = argument(argv, "--adim", 8000)
    batch = argument(argv, "--yigin", 24)
    context = argument(argv, "--baglam", 256)
    dimensions = argument(argv, "--boyut", 384)
    layers = argument(argv, "--katman", 6)
    heads = argument(argv, "--kafa", 6)
    peak = argument(argv, "--hiz", 6e-4, float)
    resume = "--devam" in argv

    device = device_of()
    pieces = spm.SentencePieceProcessor(model_file=TOKENIZER)
    config = Config(vocabulary=pieces.get_piece_size(), dimensions=dimensions,
                    layers=layers, heads=heads, context=context)
    model = Core(config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=peak,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    start_step = 0
    if resume and os.path.exists(CHECKPOINT):
        saved = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        start_step = saved["step"]
        print(f"devam ediliyor: {start_step}. adımdan")

    corpus = Corpus(TOKENS, context, batch, device)
    per_step = batch * context
    print("=" * 72)
    print(f"  DİL ÇEKİRDEĞİ EĞİTİMİ")
    print(f"  aygıt        {device}")
    print(f"  parametre    {model.size/1e6:.1f}M "
          f"(boyut {dimensions}, katman {layers}, kafa {heads})")
    print(f"  sözlük       {config.vocabulary} parça")
    print(f"  derlem       {len(corpus.train)/1e6:.1f}M parça eğitim, "
          f"{len(corpus.validate)/1e6:.2f}M doğrulama")
    print(f"  adım         {steps}, her adımda {per_step} parça "
          f"(toplam {steps*per_step/1e6:.0f}M)")
    print("=" * 72, flush=True)

    model.train()
    started = time.time()
    window = time.time()
    for step in range(start_step, steps):
        rate = learning_rate(step, steps, peak, warmup=min(200, steps // 20))
        for group in optimiser.param_groups:
            group["lr"] = rate
        x, y = corpus.draw()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)

        if (step + 1) % REPORT_EVERY == 0:
            synchronise(device)
            elapsed = time.time() - window
            window = time.time()
            speed = REPORT_EVERY * per_step / elapsed
            left = (steps - step - 1) * per_step / speed
            measured = estimate(model, corpus, rounds=5)
            print(f"  adım {step+1:>6}/{steps}  "
                  f"kayıp {measured['train']:.3f}  "
                  f"doğrulama {measured['validate']:.3f}  "
                  f"şaşkınlık {math.exp(min(measured['validate'], 20)):>7.1f}  "
                  f"{speed/1000:>5.1f}K parça/sn  kalan {left/60:>5.1f} dk",
                  flush=True)

        if (step + 1) % SAMPLE_EVERY == 0:
            seed = torch.tensor([pieces.encode(PROMPT)], device=device)
            produced = model.continue_from(seed, length=50, temperature=0.8)
            text = pieces.decode(produced[0].tolist())
            print(f"     örnek: {text[:200]}".replace("\n", " "), flush=True)
            model.train()

        if (step + 1) % SAVE_EVERY == 0 or step + 1 == steps:
            torch.save({"model": model.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "config": config.to_dict(), "step": step + 1},
                       CHECKPOINT)

    total = time.time() - started
    print("=" * 72)
    print(f"  BİTTİ  {total/60:.1f} dakika, {CHECKPOINT} "
          f"({os.path.getsize(CHECKPOINT)/2**20:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
