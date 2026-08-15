"""LMM sohbet — etkileşimli. Qwen (dil) + graf/kapı (doğruluk & büyüme).

Kullanım:
    python3.11 -m lmm.chat            # bellek: models/lmm/hafiza.lmm

Her tur günlüğe yazılır (logs/lmm-<zaman>.jsonl): girdi · cevap · okuyucunun
gördüğü işlem · süre. Bellek çıkışta kaydedilir — bir sonraki oturuma taşınır
(retrain'siz büyüme). Çıkış: boş satırda Ctrl+D ya da Ctrl+C.
"""
import json
import os
import time


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    os.makedirs("models/lmm", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    from lmm.session import Session

    memory_path = os.path.join("models", "lmm", "hafiza.lmm")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join("logs", f"lmm-{stamp}.jsonl")

    print("# LMM yükleniyor (Qwen-3B, ilk sefer ~10 sn)...")
    session = Session(memory_path)
    print(f"# bellek: {memory_path} ({len(session.memory.records)} kayıt)")
    print(f"# günlük: {log_path}")
    print("# çıkış: Ctrl+C ya da Ctrl+D · boş satır atlanır\n")

    log = open(log_path, "a", encoding="utf-8")
    try:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            started = time.time()
            said = session.respond(line)
            print(said if said else "[…]")
            log.write(json.dumps({
                "at": time.strftime("%H:%M:%S"),
                "in": line,
                "out": said,
                "records": len(session.memory.records),
                "ms": round((time.time() - started) * 1000),
            }, ensure_ascii=False) + "\n")
            log.flush()
    finally:
        session.save()
        log.close()
        print(f"\n# bellek kaydedildi ({len(session.memory.records)} kayıt)"
              f" · günlük: {log_path}")


if __name__ == "__main__":
    main()
