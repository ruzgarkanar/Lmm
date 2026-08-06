"""Sohbet sınavı: tek soru değil, KONUŞMA ölçülüyor.

`scripts/olcut.py` grafın bildiği bir olguyu tek soruyla soruyor ve cevabı
tartıyor. Gerçek bir konuşma bundan fazlası: ikinci tur birinciye dayanır,
üçüncü ikisine. Bir sistemin tek soruda %95 alıp sohbette dağılması
mümkündür — ve bu projede bir kez gerçekten oldu: "penguen ve kartal kuş mu"
doğru cevaplanıyor, hemen ardından "emin misin" -> "henüz bir şey söylemedim
ki" deniyordu. Tek soruluk sınav bunu göremez.

İki şey burada başka:

    sorular ELLE YAZILMIYOR      Cümleler sistemin kendi konuşma organından
                                 (`lmm/phrasing.py`) kuruluyor. Yani sınav,
                                 sistemin "böyle sorulur" dediği biçimi
                                 kullanıyor; betikte tek bir Türkçe cümle
                                 sabiti yok. İkinci bir dil eklendiğinde bu
                                 sınav o dile kendiliğinden geçer.

    takip turları ÖLÇÜLÜYOR      Eksiltili sorular (`emin misin`, `nereden
                                 biliyorsun`, `başka ne biliyorsun`) grafa
                                 değil ÖNCEKİ TURA dayanır. Bunlar sistemin
                                 kendi dilbilgisi kalıplarından alınıyor:
                                 kavram yuvası olmayan soru kalıpları tam da
                                 bağlama dayanan sorulardır.

Dört sonuç ayrı sayılıyor, toplamları anlamlıdır:

    DOĞRU       graftaki kayda uyan cevap
    YANLIŞ      kendinden emin ama kayda uymayan      <- en pahalısı
    BİLMİYOR    dürüst reddediş
    ANLAMADI    cümleyi çözemedi

ve takip turları için ayrıca:

    TAŞIDI      bağlam sürdü; cevap önceki tura dayanıyor
    UNUTTU      "henüz bir şey söylemedim" türünden bir kopukluk

Bu sınavın KENDİ SINIRI: sorular sistemin kendi söyleyişinden kurulduğu için
sistemin zaten konuştuğu biçimleri sınıyor. Yani "anlamadı" oranı gerçekte
olduğundan düşük çıkar — bir kullanıcı hiç desteklenmeyen bir biçimde sorarsa
bu sınav onu göremez. `scripts/olcut.py` bağımsız soru biçimleri kullandığı
için ikisi birbirinin yerine geçmez; birlikte okunmalı.

Kullanım:
    python3 scripts/sohbet_sinavi.py [--graf models/graph/birlesik.lmm]
                                     [--sohbet 60] [--tohum 7] [--anla]
                                     [--dok]
"""
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm import serialize, registry                         # noqa: E402
from lmm.relations import IS_A, CAN, HAS_PROPERTY           # noqa: E402
from lmm.inflect import verb_form, case_form                # noqa: E402
from lmm.cli import Session                                 # noqa: E402
from lmm.memory import Memory                               # noqa: E402
from lmm.relations import CAN, CANNOT, HAS_PROPERTY, IS_A   # noqa: E402
from lmm.turkish import turkish                             # noqa: E402

# Cevabın hangi türden olduğunu anlamak için kullanılan işaretler. Bunlar
# sınavın Türkçesi DEĞİL: sistemin kendi ürettiği cevabı sistemin kendi
# `is_a_refusal` ölçütüyle tanıyoruz, artı çözümlenememe işaretleri.
UNPARSED_MARKS = ("anlamadım", "demek istedin")


def _asked(edge, lexicon):
    """Bir olgunun sorusu — sistemin KENDİ konuşma organıyla kurulmuş.

    Elle cümle yazmak yerine `phrasing`'e soruyoruz: "sen bu olguyu nasıl
    sorardın?". Böylece sınav dile bağlı kalmıyor ve sistemin desteklediğinden
    başka bir biçimi sınamıyor — sorunun anlaşılmaması, sistemin kendi
    söyleyişini okuyamaması demek olur ki bu gerçek bir kusurdur.
    """
    if edge.relation == IS_A:
        return f"{edge.concept} → {serialize.label(IS_A)} → ?"
    if edge.relation in (CAN, CANNOT):
        obj = case_form(edge.object, edge.role) if edge.object else None
        return f"{serialize.fact(edge.concept, CAN, verb_form(edge.target, True), True, obj)} ?"
    if edge.relation == HAS_PROPERTY:
        obj = case_form(edge.object, edge.role) if edge.object else None
        return f"{serialize.fact(edge.concept, HAS_PROPERTY, edge.target, True, obj)} ?"
    return None


def follow_ups(grammar):
    """Bağlama dayanan soru kalıpları — dilbilgisinin kendisinden.

    Kavram yuvası olmayan bir soru kalıbı, öznesini cümleden alamaz; demek ki
    onu SOHBETTEN alıyor. Yani "hangi sorular bağlam ister" sorusunun cevabı
    zaten dilbilgisinde yazılı ve burada elle liste tutmaya gerek yok.
    """
    found = []
    for pattern in grammar.patterns:
        if not pattern.kind.startswith("ASK"):
            continue
        if pattern.concept is not None or pattern.target is not None:
            continue
        if any(token.startswith("{") for token in pattern.tokens):
            continue        # hâlâ doldurulacak bir yuva var
        found.append((pattern.kind, " ".join(pattern.tokens)))
    return found


def context_bound(tails, graph, reader, opening):
    """Hangi takip soruları GERÇEKTEN bağlama dayanıyor — ölçerek.

    Dilbilgisinde kavram yuvası olmayan her soru bağlam sorusu değil:
    "hafızanda ne var" öznesiz ama cevabı önceki tura hiç bakmıyor. İlk
    yazışta bunlar da sayılmıştı ve sınav kendi kusurunu sistemin hatası diye
    raporladı.

    Ayrım liste tutarak değil DENEYLE yapılıyor: soru bir kez boş oturumda, bir
    kez bir turdan sonra soruluyor. Cevap değişmiyorsa o soru bağlama bağlı
    değildir. Bu, dilden de bağımsız — ikinci bir dilde de aynı deney çalışır.
    """
    import shutil
    import tempfile
    found = []
    for kind, tail in tails:
        cold_path = os.path.join(tempfile.mkdtemp(), "soguk.lmm")
        shutil.copy(graph, cold_path)
        cold = Session(cold_path, intent=reader).respond(tail)
        warm_path = os.path.join(tempfile.mkdtemp(), "sicak.lmm")
        shutil.copy(graph, warm_path)
        warm_session = Session(warm_path, intent=reader)
        warm_session.respond(opening)
        warm = warm_session.respond(tail)
        if cold != warm:
            found.append((kind, tail))
    return found


def conversations(memory, grammar, count, seed, tails=None):
    """Sohbetler: her biri bir olgu sorusu + ona dayanan takip turları."""
    edges = [edge for edge in memory.edges
             if edge.relation in (IS_A, CAN, CANNOT, HAS_PROPERTY)
             and edge.target and len(edge.concept) > 2
             and " " not in edge.concept]
    chance = random.Random(seed)
    chance.shuffle(edges)
    tails = follow_ups(grammar) if tails is None else tails
    if not tails:
        return []
    found = []
    for edge in edges:
        opening = _asked(edge, memory.lexicon)
        if opening is None:
            continue
        # İki takip turu: biri kaynağa, biri sürdürmeye dair. Hangi ikisi
        # olduğunu sınav seçmiyor, dilbilgisi veriyor.
        picked = chance.sample(tails, min(2, len(tails)))
        found.append((opening, edge, picked))
        if len(found) >= count:
            break
    return found


INFINITIVE = ("mak", "mek")


def _stem(word):
    """Hedefi gövdesine indirir: kayıt mastar tutar, cevap çekimli konuşur."""
    word = word.split()[-1]
    for ending in INFINITIVE:
        if word.endswith(ending) and len(word) > len(ending) + 1:
            return word[: -len(ending)]
    return word


def judge(said, edge):
    """Açılış turunun sonucu."""
    lowered = said.lower()
    if any(mark in lowered for mark in UNPARSED_MARKS):
        return "ANLAMADI"
    if serialize.is_refusal(said):
        return "BİLMİYOR"
    if edge.target and _stem(edge.target)[:4] not in lowered:
        return "YANLIŞ"
    positive = edge.relation != CANNOT
    if lowered.startswith("hayır") and positive:
        return "YANLIŞ"
    if lowered.startswith("evet") and not positive:
        return "YANLIŞ"
    return "DOĞRU"


def carried(said, edge, first_answer):
    """Takip turu bağlamı taşıdı mı.

    Ölçüt kelime listesi değil BAĞ: cevap ya konuyu (kavramı), ya kaynağı, ya
    da ilk cevabın taşıdığı bir şeyi anmalı. Hiçbirini anmıyorsa ve reddediş
    değilse, bağlam kopmuştur.

    "Bilmiyorum" burada UNUTTU sayılmıyor: bilmemek dürüsttür ve bu sınavın
    ölçtüğü şey değil.
    """
    lowered = said.lower()
    if serialize.is_refusal(said):
        return None                 # sayılmıyor, dürüst reddediş
    if edge.concept.lower() in lowered:
        return True
    if edge.source and edge.source.lower() in lowered:
        return True
    if edge.target and _stem(edge.target)[:4] in lowered:
        return True
    # İlk cevabın kendine özgü sözcüklerinden biri geçiyorsa bağ kurulmuştur.
    shared = set(lowered.split()) & set(first_answer.lower().split())
    return len(shared) >= 2


def main(argv):
    graph = (argv[argv.index("--graf") + 1] if "--graf" in argv
             else registry.where("graph"))
    count = int(argv[argv.index("--sohbet") + 1]) if "--sohbet" in argv else 60
    seed = (int(argv[argv.index("--tohum") + 1]) if "--tohum" in argv
            else random.randrange(10 ** 6))

    memory = Memory.load(graph)
    grammar = turkish()
    reader = None
    if "--anla" in argv:
        try:
            from core.intent import Reader
            found = Reader("cekirdek")
            reader = found.read if found.ready else None
        except Exception:                                   # noqa: BLE001
            reader = None

    # Deney açılışı, sistemin GERÇEKTEN cevapladığı bir soru olmalı. İlk
    # yazışta rastgele bir açılış alınmıştı ve o cevaplanamayınca takip
    # cevapları da boş oturumdakiyle aynı çıktı: 23 bağlam sorusunun 21'i
    # "bağlama bağlı değil" diye elendi. Deneyin kendisi bozulunca ölçüm
    # sessizce daraldı — sınavların en tehlikeli hatası.
    probes = conversations(memory, grammar, 40, seed)
    if not probes:
        print("graftan sohbet kurulamadı")
        return 1
    import shutil
    import tempfile
    opening = None
    for candidate, edge, _ in probes:
        path = os.path.join(tempfile.mkdtemp(), "deneme.lmm")
        shutil.copy(graph, path)
        if judge(Session(path, intent=reader).respond(candidate), edge) == "DOĞRU":
            opening = candidate
            break
    if opening is None:
        print("cevaplanan bir açılış bulunamadı")
        return 1
    tails = context_bound(follow_ups(grammar), graph, reader, opening)
    if not tails:
        print("bağlama dayanan soru bulunamadı")
        return 1
    talks = conversations(memory, grammar, count, seed, tails)

    opening_tally = collections.Counter()
    context_tally = collections.Counter()
    lost, wrong = [], []
    for opening, edge, picked in talks:
        # Her sohbet TEMİZ bir oturumda: bir öncekinin bağlamı sızmasın.
        working = os.path.join(tempfile.mkdtemp(), "sohbet.lmm")
        shutil.copy(graph, working)
        session = Session(working, intent=reader)
        first = session.respond(opening)
        verdict = judge(first, edge)
        opening_tally[verdict] += 1
        if verdict == "YANLIŞ" and len(wrong) < 6:
            wrong.append((opening, first,
                          f"{edge.concept} --{edge.relation}--> {edge.target}"))
        if verdict not in ("DOĞRU",):
            continue        # takip ancak açılış tutunca anlamlı
        for kind, tail in picked:
            said = session.respond(tail)
            held = carried(said, edge, first)
            if held is None:
                context_tally["REDDETTİ"] += 1
            elif held:
                context_tally["TAŞIDI"] += 1
            else:
                context_tally["UNUTTU"] += 1
                if len(lost) < 6:
                    lost.append((opening, first, tail, said))

    total = sum(opening_tally.values())
    print(f"{os.path.basename(graph)} — {len(memory.edges):,} olgu, "
          f"{len(memory.concepts()):,} kavram")
    print(f"{total} sohbet, tohum {seed}"
          f"{', niyet ağı açık' if reader else ''}")
    print(f"sorular sistemin kendi konuşma organından kuruldu; "
          f"{len(follow_ups(grammar))} öznesiz sorunun {len(tails)} tanesi "
          f"ölçümle bağlama bağlı çıktı\n")
    print("  AÇILIŞ TURU")
    for name in ("DOĞRU", "YANLIŞ", "BİLMİYOR", "ANLAMADI"):
        share = opening_tally[name] / max(total, 1) * 100
        print(f"    {name:<10} {opening_tally[name]:>4}  %{share:>5.1f}")
    answered = opening_tally["DOĞRU"] + opening_tally["YANLIŞ"]
    print(f"    cevapladığında isabet: "
          f"%{opening_tally['DOĞRU'] / max(answered, 1) * 100:.1f}")

    followed = sum(context_tally.values())
    print("\n  TAKİP TURU (bağlam)")
    for name in ("TAŞIDI", "UNUTTU", "REDDETTİ"):
        share = context_tally[name] / max(followed, 1) * 100
        print(f"    {name:<10} {context_tally[name]:>4}  %{share:>5.1f}")
    judged = context_tally["TAŞIDI"] + context_tally["UNUTTU"]
    print(f"    bağlam tutma: "
          f"%{context_tally['TAŞIDI'] / max(judged, 1) * 100:.1f}")

    if "--dok" in argv and wrong:
        print("\n  YANLIŞ örnekleri (en pahalı hata türü):")
        for asked, said, record in wrong:
            print(f"    soru : {asked}")
            print(f"    cevap: {said[:70]}")
            print(f"    kayıt: {record}\n")
    if "--dok" in argv and lost:
        print("  BAĞLAM KOPMASI örnekleri:")
        for opening, first, tail, said in lost:
            print(f"    1) {opening}\n       {first[:66]}")
            print(f"    2) {tail}\n       {said[:66]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
