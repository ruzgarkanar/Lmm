"""Etkileşimli sohbet — her tur günlüğe yazılır, sorunlar oradan okunur.

Kullanım:
    python3.11 -m v3.chat            # bellek: models/v3/sohbet.lmm3
    (çıkmak için boş satırda Ctrl+D ya da Ctrl+C)

Günlük: logs/chat-<zaman>.jsonl — her satır bir tur:
    girdi · cevap · okuyucunun gördüğü işlem (tür/özne/değer/güven) · süre

Günlük İNCELEME içindir: okuyucu neyi yanlış okudu, hangi tur boş kaldı,
hangi kayıt yazıldı — hepsi tek dosyada, sonradan birlikte bakılır.
"""
import json
import os
import sys
import time

from v3.reader import ASK, PASS, WRITE
from v3.session import Session

KIND = {PASS: "PASS", ASK: "ASK", WRITE: "WRITE"}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    os.makedirs("logs", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join("logs", f"chat-{stamp}.jsonl")
    memory_path = os.path.join("models", "v3", "sohbet.lmm3")

    session = Session(memory_path)
    print(f"# bellek: {memory_path} ({len(session.memory.records)} kayıt)")
    print(f"# günlük: {log_path}")
    print(f"# okuyucu: {session.reader.ready} · konuşucu: {session.speaker.ready}")
    print("# çıkış: Ctrl+C ya da Ctrl+D\n")

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
            operation = session.reader.read(line)
            said = session.respond(line)
            turn = {
                "at": time.strftime("%H:%M:%S"),
                "in": line,
                "out": said,
                "read": {"kind": KIND.get(operation.kind, "?"),
                         "subject": operation.subject,
                         "predicate": operation.predicate,
                         "value": operation.value,
                         "confidence": round(operation.confidence, 3)},
                "records": len(session.memory.records),
                "ms": round((time.time() - started) * 1000),
            }
            log.write(json.dumps(turn, ensure_ascii=False) + "\n")
            log.flush()
            print(said if said else "[ ]")
    finally:
        session.save()
        log.close()
        print(f"\n# bellek kaydedildi ({len(session.memory.records)} kayıt)"
              f" · günlük: {log_path}")


if __name__ == "__main__":
    main()
