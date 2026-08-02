"""Bir belgeyi LMM grafına derler: metin girer, denetlenmiş bilgi çıkar.

Bu, LMM'in firmaya vaat ettiği şeyin tamamı. Belge bir kez okunur — o an bir dil
modeli kullanılır — ve çıkan her olgu üç denetimden geçer: kaynak cümlesi belgede
gerçekten var mı, iddia o cümleden çıkıyor mu, kavramlar yeniden kullanılabilir
mi. Sonra dil modeli gider. Geriye kalan graf, firmanın kendi sunucusunda, CPU
üzerinde, kaynağı görünür ve tek satırla düzeltilebilir biçimde çalışır.

Birden fazla cümlenin söylediği olgular ayrıca işaretlenir. Tek bir okumanın
hatası olabilir; iki ayrı cümlenin aynı şeyi söylemesi bunu zorlaştırır.

Kullanım:
    python3 scripts/belge_derle.py <belge.docx|belge.txt> [--cikti ad.lmm]
    python3 scripts/belge_derle.py belge.txt --kuru       # ağa çıkmadan dener
"""
import os
import re
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.compiler import (compile_text, model_reader, CompileError)  # noqa: E402
from lmm.memory import Memory, Edge                      # noqa: E402
from lmm.reading import sentences                        # noqa: E402
from lmm.trust import confidence_for                     # noqa: E402

RELATION_NAMES = {"type": "type", "can": "can", "property": "property",
                  "has": "has"}
LONGEST = 400           # bundan uzun "cümle" tablo artığıdır
SHORTEST = 30


def read_document(path):
    """docx ya da düz metin. Başka biçim varsa önce metne çevrilmeli."""
    if path.endswith(".docx"):
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8")
        text = re.sub(r"<[^>]+>", " ", xml)
        return re.sub(r"\s+", " ", text)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def usable_sentences(text):
    return [s for s in sentences(text) if SHORTEST <= len(s) <= LONGEST]


def main(argv):
    if len(argv) < 2 or argv[1].startswith("--"):
        print(__doc__.strip().splitlines()[-3])
        return 1
    path = argv[1]
    if not os.path.exists(path):
        print(f"bulunamadı: {path}")
        return 1
    name = os.path.basename(path)
    output = (argv[argv.index("--cikti") + 1] if "--cikti" in argv
              else os.path.splitext(name)[0] + ".lmm")

    text = read_document(path)
    chosen = usable_sentences(text)
    print(f"belge: {name}")
    print(f"  {len(text)/1000:.0f}K karakter, {len(chosen)} kullanılabilir cümle")
    if "--kuru" in argv:
        print("  (kuru çalışma: ağa çıkılmadı)")
        for sentence in chosen[:5]:
            print(f"    {sentence[:100]}")
        return 0

    print(f"  okunuyor — {(len(chosen) + 11) // 12} istek...", flush=True)
    started = time.time()
    try:
        report = compile_text(chosen, text, model_reader())
    except CompileError as error:
        print(f"  derleme başarısız: {error}")
        return 1
    elapsed = time.time() - started

    facts = report.facts()
    corroborated = [(key, n) for key, n in facts if n > 1]
    print(f"\n{report.total} aday, {len(report.accepted)} denetimi geçti "
          f"({elapsed:.0f} sn)")
    for reason, n in sorted(report.reasons().items(), key=lambda x: -x[1]):
        print(f"    reddedildi: {reason} — {n}")
    print(f"\n  benzersiz olgu       {len(facts)}")
    print(f"  birden fazla cümlece {len(corroborated)}")
    print(f"  100 cümlede          {len(facts)/max(len(chosen),1)*100:.1f} olgu")

    memory = Memory()
    for (concept, relation, target), count in facts:
        edge = Edge(concept, RELATION_NAMES[relation], target, source=name,
                    confidence=confidence_for(name))
        written = memory.write(edge)
        for extra in range(count - 1):
            written.corroborate(f"{name}#{extra + 2}")
    memory.save(output)
    print(f"\n{output}  {os.path.getsize(output)/1024:.0f} KB, "
          f"{len(memory.concepts())} kavram  (çalışma anında GPU yok)")

    if corroborated:
        print("\n  en çok doğrulanan olgular:")
        for (concept, relation, target), count in corroborated[:10]:
            print(f"    {count}x  {concept} | {relation} | {target}")
    print("\n  örnek olgular:")
    for (concept, relation, target), _ in facts[:10]:
        print(f"    {concept} | {relation} | {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
