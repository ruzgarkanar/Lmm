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
    from lmm.mind import Mind
    mind = Mind(session)        # otonom zihin — turlar arası kendi düşünür (ms)
    print(f"# bellek: {memory_path} ({len(session.memory.records)} kayıt)")
    print(f"# günlük: {log_path}")
    print("# çıkış: Ctrl+C ya da Ctrl+D · boş satır atlanır\n")

    log = open(log_path, "a", encoding="utf-8")
    proposed = set()        # proaktif olarak ÖNERİLEN meraklar (tekrar etmemek için)
    try:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            # OTONOM ZİHİN — öz-farkındalık + kendi büyüme komutları:
            if line in ("?merak", "?curiosity"):
                w = mind.wonder()           # çok-kullanılan-ama-tanımsız (rafine)
                print("# merak ediyorum (sürekli kullanıp da bilmediğim): "
                      + (", ".join(w) if w else "—")
                      + ("  →  ?araştır <konu>" if w else ""))
                continue
            if line.startswith("?araştır ") or line.startswith("?research "):
                konu = line.split(" ", 1)[1].strip()
                print(f"# {konu} araştırılıyor (web, düşük güven)...")
                ok = mind.research(konu)
                print(f"# {'öğrendim ✓ (kaynaklı)' if ok else 'bulamadım'}")
                continue
            if line in ("?çelişki", "?tension"):
                tens = session.tension()
                if tens:
                    for subj, vals in tens:
                        print(f"# çelişki: {subj} → {' ↔ '.join(vals)}")
                else:
                    print("# çelişki yok")
                continue
            started = time.time()
            said = session.respond(line)
            print(said if said else "[…]")
            # OTONOM DÜŞÜNME (ms, turlar arası): mesajdan sonra kendi akıl yürütür —
            # çelişki çöz / türet / damıt. Yeni bir şey türetirse sessizce söyler.
            before = sum(1 for r in session.memory.records.values()
                         if r.source == "#inference")
            mind.run()
            after = sum(1 for r in session.memory.records.values()
                        if r.source == "#inference")
            if after > before:
                print(f"  💭 (kendim türettim: {after - before} yeni bağ)")
            # PROAKTİF MERAK: bir kavramı yeterince çok kullanıp da bilmiyorsa,
            # kendi 'bunu öğrenmek istiyorum' der (bir kez, gürültü yapmadan).
            cur = mind.top_curiosity()
            if cur and cur[0] not in proposed:
                proposed.add(cur[0])
                print(f"  💭 '{cur[0]}' sürekli geçiyor ama ne olduğunu bilmiyorum"
                      f" — ?araştır {cur[0]}")
            log.write(json.dumps({
                "at": time.strftime("%H:%M:%S"),
                "in": line,
                "out": said,
                # KAPI-onaylı üçlüler: uyku-konsolidasyonun altın verisi
                # (cümle → üçlü çifti; modelin ham çıktısı değil, kapının
                # kabul ettiği). finetune/consolidate.py bunları hasat eder.
                "learned": session.last_written,
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
