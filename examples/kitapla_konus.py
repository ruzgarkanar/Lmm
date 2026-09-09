#!/usr/bin/env python3.11
"""Bir kitapla konuşmak — LMM'i düz metin üzerinde sınamak.

    python3.11 examples/kitapla_konus.py "<kitap>.pdf"

BU BİR SINAV, VE NEYİ SINADIĞINI BİLEREK BAKMAK GEREK. LMM'in son
haftalardaki gelişmelerinin çoğu KAYIT biçimli belgeler için: "EĞİTİM
SÜRESİ: 2 Gün" gibi başlıklı satırlar, belgeler arası kıyas, alan sayımı.
Bir roman bunların hiçbirini taşımaz — tek belge, başlıksız düz anlatı.
Yani burada çalışan tek şey ERİŞİM ve KAPI: doğru cümleleri bulup
bulamadığı, ve bulamadığında uydurup uydurmadığı.

Beklenmesi gereken davranış:
  • "Alyona İvanovna kimdir?" gibi metinde AÇIKÇA yazan şeyler → cevap,
    ve cevabın dayandığı cümleler `alıntı:` ile görülebilir.
  • "Raskolnikov neden öldürdü?" gibi kitabın tamamına yayılan yorum
    soruları → çoğu zaman çekimser. Bu bir kusur değil, tasarım: cevap
    tek bir cümlede yazmıyorsa sistem onu UYDURMAK yerine susar.
  • Kitapta hiç geçmeyen bir şey → "bilmiyorum". Uydurma sıfır olmalı;
    olmuyorsa asıl bulgu odur.

İlk çalıştırma PDF'i okur (1300 sayfa ≈ 40 sn) ve depoyu PDF'in yanına
kaydeder; sonraki çalıştırmalar saniyeler içinde açılır.

Komutlar:
    alıntı: <konu>     kitaptan ilgili cümleleri ham haliyle göster
    nerede: <terim>    terim kaç cümlede geçiyor
    çıkış
"""
import glob
import os
import re
import sys
import time

# Depo ve motor: her koşu buluta gider (yerel model makineyi kasıyor).
SRC = "/Users/ruzgarkanar/Desktop/MyBOT/src"
if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault("LMM_BACKEND", "azure")
os.environ.setdefault("LMM_ENV", "/Users/ruzgarkanar/Desktop/MyBOT/.env")
os.environ.setdefault("LMM_TIMEOUT", "300")

from lmm import tables                                       # noqa: E402
from lmm.api import Memory                                   # noqa: E402

PERSONA = """Sen bir kitap arkadaşısın: elindeki tek kaynak bu kitabın
metni. Sıcak ve sade konuş, kısa cevap ver. Metinde yazmayan hiçbir şeyi
söyleme — bir şeyi bilmiyorsan bilmediğini söyle."""


def temizle(metin):
    """PDF'in satır sonu tirelerini ve görünmez boşluklarını topla.

    Tarayıcıdan gelen metinde kelimeler satır sonunda "ki­-\\n şilerinden"
    gibi bölünüyor; birleştirilmezse "kişilerinden" hiçbir aramada
    bulunmaz. Bu, kütüphanenin değil bu betiğin işi: metni okunur hale
    getirmek, kitabı sisteme vermeden önceki normal hazırlık.
    """
    metin = metin.replace("​", "").replace("­", "")
    metin = re.sub(r"(\w)[‐-―-]\s*\n\s*(\w)", r"\1\2", metin)
    return re.sub(r"[ \t]+", " ", metin)


def yukle(pdf):
    """Depoyu aç; yoksa PDF'ten kur ve kaydet."""
    depo = os.path.splitext(pdf)[0] + ".lmm"
    m = Memory(depo, persona=PERSONA, warmth=0.3, reply_tokens=400)
    if m.session.evidence.sentences:
        return m, depo, 0.0
    t0 = time.time()
    print(f"# ilk açılış: PDF okunuyor ({os.path.basename(pdf)}) …",
          flush=True)
    metin, uyarilar = tables.read_pdf(pdf)
    for u in uyarilar[:2]:
        print("  uyarı:", u[:90])
    metin = temizle(metin)
    print(f"# {len(metin):,} karakter · öğreniliyor …", flush=True)
    # deep=False: KİTAP İÇİN TEK DOĞRU AYAR. deep=True her cümle için
    # motora gider — bir romanda on binlerce çağrı, saatler ve para
    # demek. Kanıt katmanı (aradığımız şey) motorsuz kurulur.
    m.learn(metin, source="#kitap:" + os.path.basename(pdf)[:40],
            deep=False)
    m.save()
    return m, depo, time.time() - t0


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else None
    if not pdf:
        adaylar = sorted(glob.glob("*.pdf"))
        pdf = adaylar[0] if adaylar else None
    if not pdf or not os.path.exists(pdf):
        sys.exit("kullanım: python3.11 kitapla_konus.py <kitap.pdf>")

    m, depo, sure = yukle(os.path.abspath(pdf))
    ev = m.session.evidence
    if sure:
        print(f"# hazır: {sure:.0f} sn · depo {os.path.basename(depo)}")
    print(f"# {len(ev.sentences):,} cümle · {len(ev.index):,} sözcük")
    print("# komutlar: 'alıntı: <konu>' · 'nerede: <terim>' · 'çıkış'\n")

    while True:
        try:
            soru = input("siz > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not soru:
            continue
        if soru.lower() in ("çıkış", "cikis", "exit", "quit"):
            break

        if soru.lower().startswith(("alıntı:", "alinti:")):
            konu = soru.split(":", 1)[1].strip()
            for satir in ev.find(konu, most=5):
                print("   ", satir[:300])
            print()
            continue

        if soru.lower().startswith("nerede:"):
            terim = soru.split(":", 1)[1].strip()
            for kaynak, sayi in m.where(terim):
                print(f"    {sayi} cümlede geçiyor  ({kaynak})")
            print()
            continue

        t0 = time.time()
        akan = []

        def satir_geldi(satir):
            if not akan:
                print("kitap >")
            akan.append(satir)
            print("   ", satir, flush=True)

        said = m.session.respond(soru, teach=False, on_line=satir_geldi)
        gecen = time.time() - t0
        if not akan:
            print(f"kitap > {said}")
        # Çekimserlik damgası: sistem bir şey İDDİA ETTİ mi, yoksa
        # bilmediğini mi söyledi. Ürünün asıl vaadi bu damgada.
        durum = "çekimser" if m.session.last_abstained else "iddia"
        print(f"        [{durum} · {gecen:.1f} sn]\n")

    print("\ngörüşmek üzere.")


if __name__ == "__main__":
    main()
