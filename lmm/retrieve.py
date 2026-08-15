"""Graftan olgu çekip Qwen'e verilecek FACTS bloğunu kurar.

RAG'den üç farkı: (1) çekilen şey yapısal DOĞRULANMIŞ üçlü, serbest metin
parçası değil; (2) her olgu KAYNAK taşır; (3) cevap üretildikten SONRA aynı
olgulara geri-denetlenir (verify.py) — klasik RAG çıktıyı denetlemez.
"""
from v3.gate import SPEAK
from lmm import link


def gather(memory, subject_key, most=4, associative=True):
    """Öznenin çevresindeki ilgili kayıtlar.

    `associative=True` (varsayılan): doğrudan olgular (kartal→kuş) + öznenin
    DEĞERLERİ hakkındaki bir-hop olgular (kuş→hayvan) — çağrışımsal bağlam.
    `associative=False` (cevap yolu): YALNIZ doğrudan olgular. KENAR-denetimiyle
    (verify) uyum için: bir-hop "kuş→hayvan" enjekte edilince model "kartal
    hayvandır" diyor ama {kartal,hayvan} doğrudan kenar değil (is-a henüz geçişli
    değilse türetilmemiş) → cevap düşüyordu. Doğrudan olgu verilince model
    temellenmiş "kartal kuştur" der, kenar-denetiminden geçer. trust>=SPEAK.
    """
    if subject_key is None:
        return []
    direct = [r for r in memory.about(subject_key) if r.trust >= SPEAK]
    records = list(direct)
    if associative:
        for r in direct:                    # bir-hop: değerlerin çevresi
            if isinstance(r.value, int):
                records += [x for x in memory.about(r.value) if x.trust >= SPEAK]
    seen, uniq = set(), []
    for r in records:
        if r.key not in seen:
            seen.add(r.key)
            uniq.append(r)
    uniq.sort(key=lambda r: -r.trust)
    return uniq[:most]


def facts_block(memory, records):
    """Kayıtları Qwen'e verilecek numaralı metne çevirir. Kaynak etiketi
    (#operator vb.) İÇERİDE kalır, enjekte metne konmaz — yoksa Qwen onu
    cevaba kopyalıyor (ölçüldü). Provenans kayıt anahtarında durur, gerekirse
    ayrıca gösterilir."""
    lines = []
    for i, r in enumerate(records, 1):
        subject = link.label_of(memory, r.subject)
        value = link.label_of(memory, r.value)
        lines.append(f"[{i}] {subject} → {value}")
    return "\n".join(lines)
