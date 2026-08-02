"""LMM'in dil çekirdeği: yalnızca dil bilen, bilgi bilmeyen küçük bir ağ.

Bu dosya bilerek `lmm/` paketinin dışında duruyor. `lmm/` sıfır bağımlılıkla
çalışır ve öyle kalmalı: bellek, muhakeme ve epistemik kapı hiçbir zaman torch'a
ihtiyaç duymamalı, çünkü LMM'in savunduğu iddia tam olarak bilginin bir kütüphane
ya da bir GPU olmadan taşınabilmesi.

Çekirdeğin işi dar ve bilinçli olarak dar: **ne söyleneceğine değil, nasıl
söyleneceğine** karar verir. Bilgi graftadır; burada yalnızca Türkçe'nin ek
düzeni, sözcük sırası ve akışı öğrenilir. Ölçtüğümüz şey bunu haklı çıkardı —
sözcük anlamının içsel boyutu yüksek (varyansın yarısı için 41 boyut), ama o
yüksek boyut grafta bedava; ağın öğrenmesi gereken kısım geriye kalan dilbilgisi.

Mimari standart bir çözücü (decoder) yığını. Burada yenilik aramıyoruz; yenilik
çekirdeğin **kapıya bağlanma biçiminde**, çekirdeğin kendisinde değil. Bilinen
bir parçayı bilinmeyen bir yere takıyoruz.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, vocabulary=16000, dimensions=384, layers=6, heads=6,
                 context=256, dropout=0.1):
        self.vocabulary = vocabulary
        self.dimensions = dimensions
        self.layers = layers
        self.heads = heads
        self.context = context
        self.dropout = dropout

    def to_dict(self):
        return dict(vocabulary=self.vocabulary, dimensions=self.dimensions,
                    layers=self.layers, heads=self.heads, context=self.context,
                    dropout=self.dropout)


def rotation(width, length, device, base=10000.0):
    """RoPE için açılar: konumu mutlak değil GÖRELİ kodlar.

    Öğrenilmiş mutlak konum gömmesi bir tablodur ve tablo eğitim anında
    donar — 256'da eğitilen model 257'nci parçayı hiç göremez, dolayısıyla
    bağlamı sonradan uzatmak imkânsızdır, baştan eğitmek gerekir.

    Döner gömmede konum, vektörün açısına yazılır ve iki parça arasındaki
    fark açılar arasındaki farktan çıkar. Model "arada 5 parça var" bilgisini
    öğrenir, "17. sıradayım" bilgisini değil — ve bu, eğitildiğinden uzun
    dizilere taşınabilir.
    """
    inverse = 1.0 / (base ** (torch.arange(0, width, 2, device=device).float()
                              / width))
    positions = torch.arange(length, device=device).float()
    angles = torch.outer(positions, inverse)
    return torch.cos(angles), torch.sin(angles)


def rotate(x, cosine, sine):
    """Vektörü konumuna göre döndürür. Çiftler hâlinde, karmaşık sayı gibi."""
    first, second = x[..., 0::2], x[..., 1::2]
    turned = torch.stack((first * cosine - second * sine,
                          first * sine + second * cosine), dim=-1)
    return turned.flatten(-2)


class Attention(nn.Module):
    """Nedensel öz-dikkat: her parça yalnızca kendinden öncekilere bakar."""

    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.project = nn.Linear(config.dimensions, 3 * config.dimensions)
        self.output = nn.Linear(config.dimensions, config.dimensions)
        self.dropout = config.dropout

    def forward(self, x, causal=True):
        """causal=False: her parça ileriye de bakar.

        Üretim için nedensellik şart — model gelecekteki kelimeyi göremez. Ama
        bir cümleyi *anlamak* için tersi doğru: "penguen uçar" ile "penguen uçar
        mı" arasındaki farkı son kelime belirler ve öznenin rolü ona bakılmadan
        verilemez.
        """
        batch, length, dimensions = x.shape
        query, key, value = self.project(x).split(dimensions, dim=2)
        shape = (batch, length, self.heads, dimensions // self.heads)
        query, key, value = (t.view(shape).transpose(1, 2)
                             for t in (query, key, value))
        cosine, sine = rotation(dimensions // self.heads, length, x.device)
        cosine, sine = cosine[None, None], sine[None, None]
        query, key = rotate(query, cosine, sine), rotate(key, cosine, sine)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, dimensions)
        return self.output(attended)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.before_attention = nn.LayerNorm(config.dimensions)
        self.attention = Attention(config)
        self.before_forward = nn.LayerNorm(config.dimensions)
        self.forward_pass = nn.Sequential(
            nn.Linear(config.dimensions, 4 * config.dimensions),
            nn.GELU(),
            nn.Linear(4 * config.dimensions, config.dimensions),
            nn.Dropout(config.dropout))

    def forward(self, x, causal=True):
        x = x + self.attention(self.before_attention(x), causal=causal)
        return x + self.forward_pass(self.before_forward(x))


class Core(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocabulary, config.dimensions)
        # Konum artık burada değil, dikkatin içinde ve göreli (RoPE). Tablo
        # kaldırıldı çünkü bağlamı dondurmanın tek sebebi oydu.
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.final = nn.LayerNorm(config.dimensions)
        self.head = nn.Linear(config.dimensions, config.vocabulary, bias=False)
        self.head.weight = self.token.weight     # girişle çıkış aynı sözlüğe bakar
        self.apply(self._start)
        for name, parameter in self.named_parameters():
            if name.endswith("output.weight") or name.endswith("2.weight"):
                nn.init.normal_(parameter, mean=0.0,
                                std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _start(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        x = self.drop(self.token(tokens))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.final(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                               ignore_index=-1)
        return logits, loss

    @property
    def size(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def continue_from(self, tokens, length=60, temperature=0.8, top_k=40,
                      allowed=None):
        """Metni sürdürür.

        `allowed` verilirse yalnızca o parçalar üretilebilir. Kapının çekirdeğe
        bağlandığı yer burasıdır: çekirdek neyi söyleyebileceğini kendi seçmez,
        izin verilen küme dışarıdan gelir. Uydurmanın yapısal olarak imkânsız
        kalması bu tek satıra dayanır.
        """
        self.eval()
        for _ in range(length):
            window = tokens[:, -self.config.context:]
            logits, _ = self(window)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if allowed is not None:
                blocked = torch.full_like(logits, float("-inf"))
                blocked[:, allowed] = logits[:, allowed]
                logits = blocked
            if top_k:
                cut = min(top_k, logits.shape[-1])
                threshold = torch.topk(logits, cut).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probabilities = F.softmax(logits, dim=-1)
            following = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, following), dim=1)
        return tokens
