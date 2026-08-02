"""LMM sohbeti: graf bilgiyi verir, çekirdek dile getirir, kapı arada durur.

Her cevap iki biçimde gösterilir çünkü ikisi iki ayrı organın sorumluluğunda:

    GRAF      doğruluk buradan gelir — kaynağıyla, düz ve tartışmasız
    ÇEKİRDEK  akıcılık buradan gelir — ama yalnızca grafın kelimeleriyle

Üçüncü satır bir denetimdir: yeniden ifade edilen cümlede grafın söylemediği bir
içerik kelimesi geçti mi? Beklenen cevap "hayır"dır ve her cevapta yeniden
sorulur. Uydurmanın imkânsızlığı bir kere kanıtlanıp geçilecek bir şey değil,
her cümlede yeniden görünmesi gereken bir özellik.

Çekirdek yoksa sohbet yine çalışır, yalnızca akıcılık katmanı kapalı kalır.

Kullanım:
    python3 scripts/lmm_sohbet.py [model.lmm] [--cekirdek lmm-cekirdek.pt]
    python3 scripts/lmm_sohbet.py model.lmm --sicaklik 0.5
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.cli import Session                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lmm import registry                                    # noqa: E402
GOODBYE = ("çık", "cik", "exit", "quit", "q")


def load_turkish():
    """Hazır Türkçe çekirdek varsa onu kullan — mimariyi sınamak için."""
    try:
        from core.turkish import TurkishCore, available
        if not available():
            return None
        from core.bridge import Bridge
        core = TurkishCore()
        return Bridge(core, core.pieces, device=core.device), core.size, True
    except Exception:                                   # noqa: BLE001
        return None


def load_bridge(path, quiet=False):
    """Çekirdek isteğe bağlıdır; yoksa sohbet grafla devam eder."""
    if not os.path.exists(path):
        if not quiet:
            print(f"  (çekirdek yok: {path} — yalnızca graf konuşacak)")
        return None, 0
    try:
        from core.bridge import load
        return load(path)
    except ImportError:
        print("  (torch kurulu değil — yalnızca graf konuşacak)")
        return None, 0
    except Exception as error:                          # noqa: BLE001
        print(f"  (çekirdek yüklenemedi: {error})")
        return None, 0


def main(argv):
    graph = argv[1] if len(argv) > 1 and not argv[1].startswith("--") else registry.where("graph")
    core_path = (argv[argv.index("--cekirdek") + 1] if "--cekirdek" in argv
                 else registry.where("core"))
    temperature = (float(argv[argv.index("--sicaklik") + 1])
                   if "--sicaklik" in argv else 0.6)
    if not os.path.exists(graph):
        print(f"graf bulunamadı: {graph}")
        return 1

    session = Session(graph)
    found = load_turkish()
    ready = False
    if found is not None:
        bridge, size, ready = found
    else:
        bridge, size = load_bridge(core_path)
    print("=" * 70)
    print(f"  LMM  —  graf {len(session.memory.edges)} bilgi, "
          f"{len(session.memory.concepts())} kavram")
    if bridge:
        print(f"       —  çekirdek {size/1e6:.0f}M parametre"
              f"{' (hazır Türkçe model)' if ready else ''}")
    print("  çıkmak için: çık")
    print("=" * 70)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in GOODBYE:
            print(session.respond("çık"))
            break
        answer = session.respond(line)
        print(f"  GRAF      {answer}")
        if bridge:
            from lmm.reasoning import Reasoning
            from lmm.intuition import tokenize
            concepts = [w for w in tokenize(line)
                        if w in set(session.memory.concepts())]
            said = bridge.express(answer, line, session.memory,
                                  Reasoning(session.memory), concepts)
            # Ekrandaki denetim, üretimdeki denetimle aynı kümeye bakmalı;
            # farklı bakınca temiz bir cevap "sızdı" görünüyordu.
            from core.gated import approved
            allowed = [w.strip(".,()").lower() for w in answer.split()]
            allowed += list(approved(session.memory, concepts,
                                     Reasoning(session.memory)))
            escaped = ([] if said == answer
                       else bridge.voice.escaped(said, allowed))
            print(f"  ÇEKİRDEK  {said}")
            print(f"  KAPI      {'temiz' if not escaped else 'SIZDI: ' + str(escaped)}")
    session.save()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
