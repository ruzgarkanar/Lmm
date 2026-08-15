"""Orkestratör — LMM'in tüm organlarını tek akışta bağlar.

    mesaj
      → EXTRACT (Qwen)     kind + üçlüler  (ADAY — yazamaz)
      → WRITE  → gate.admit                 grafa, kapıdan
      → ASK    → retrieve → generate (Qwen) → verify (KAPI) → cevap
      → CHAT   → generate (Qwen, olgusuz)   → verify (KAPI) → cevap

Belleği taşır (`Memory.load/save`), yaşantıyı kaydeder. Uydurmama HER yolda
`verify` ile korunur: Qwen ne derse desin, grafta desteği yoksa olgu iddiası
çıkışta düşer. "Qwen kayıt yazamaz" (giriş) + "Qwen desteksiz olgu söyleyemez"
(çıkış) — belkemiği kural iki uçta da.

Mod: STRICT (uydurma 0 — desteksiz olgu düşer) · ASSIST (işaretlenir).
"""
import os

from v3 import dynamics
from v3.gate import Gate
from v3.memory import Memory, OPERATOR, STRANGER
from lmm import extract, generate, link, retrieve, verify

SLEEP_EVERY = 50   # kaç turda bir uyku (damıt/sol/hakemle) — v3 §117

BILMIYORUM = "Bunu bilmiyorum."


class Session:
    """Bir konuşma. Qwen (dil) + graf/kapı (doğruluk & büyüme)."""

    def __init__(self, path=None, who="#operator", mode="STRICT"):
        self.path = path
        self.memory = (Memory.load(path) if path and os.path.exists(path)
                       else Memory())
        self.gate = Gate(self.memory)
        self.who = who
        self.level = OPERATOR if who == "#operator" else STRANGER
        self.mode = mode
        self.vectors = {}
        self.turns = 0
        if self.memory.self_key is None:
            self.memory.self_key = self.memory.identify("#self")

    def respond(self, message):
        """Bir mesaja cevap. Dönen daima metin; asla desteksiz olgu.

        Sağlamlık (kod-denetimi): boş mesaj korunur, tüm akış try/except içinde
        (tek bozuk üretim konuşmayı çökertmesin — güvenli cevaba düşer)."""
        if not message or not message.strip():
            return ""
        # AUTO-UYKU: her SLEEP_EVERY turda bir damıt/sönümle/hakemle. v3'te
        # ölçüldü — bu mekanizmalar yalnız elle çağrılıyordu, sistem hiç
        # "uyumuyordu"; sarmalandı. Hata olursa konuşmayı çökertmesin.
        self.turns += 1
        if self.turns % SLEEP_EVERY == 0:
            try:
                dynamics.sleep(self.memory)
            except Exception:                               # noqa: BLE001
                pass
        try:
            op = extract.extract(message)
            if op["kind"] == extract.WRITE and op["triples"]:
                return self._write(op["triples"])
            if op["kind"] == extract.ASK:
                # ASK ama özne çıkmadıysa: boş özneyle dene → BILMIYORUM'a düşer
                subject = op["triples"][0][0] if op["triples"] else None
                return self._answer(message, subject)
            return self._chat(message)
        except Exception:                                   # noqa: BLE001
            return BILMIYORUM

    # --- yazma ---------------------------------------------------------
    def _write(self, triples):
        """Öğretileni grafa koyar — kapıdan geçerek (Qwen yazamaz, kapı yazar).

        Kod-denetimi: değersiz yarım üçlü atlanır (None değer gate'e gitmesin);
        aynı üçlü iki kez gelirse tekilleştirilir (tekrarlı 'Öğrendim' olmasın).
        """
        wrote = []
        conflicts = []
        seen = set()
        for subject, predicate, value in triples:
            if not value:
                continue                       # yarım üçlü — yazma
            sk = link.resolve(self.memory, subject, self.vectors, create=True)
            vk = link.resolve(self.memory, value, self.vectors, create=True)
            if sk is None or vk is None or (sk, vk) in seen:
                continue
            seen.add((sk, vk))
            pk = (link.resolve(self.memory, predicate, self.vectors, create=True)
                  if predicate else None)
            # ÇELİŞKİ: yeni olgu, aynı özne+yüklemde BAŞKA değer taşıyan bir
            # kayıtla çelişiyor mu — yazmadan önce bak (CONTRA bağı admit'te
            # kurulur, biz kullanıcıya YÜZEYE çıkarırız: sessizce üstüne yazma).
            clash = self.gate._contradiction(sk, pk, vk)
            if clash is not None:
                conflicts.append((link.label_of(self.memory, sk),
                                  link.label_of(self.memory, clash.value),
                                  link.label_of(self.memory, vk)))
            record, _ = self.gate.admit(sk, pk, vk, self.who, self.level)
            if record is not None:
                wrote.append((link.label_of(self.memory, sk),
                              link.label_of(self.memory, vk)))
                # TÜRETME (geçişli akıl): "kartal→kuş, kuş→hayvan ⊢ kartal→
                # hayvan". Geçişlilik veriden öğrenilir (≥2 tanıklı üçgen),
                # çıkarım #inference kaynağıyla düşük güvenle yazılır. Bu,
                # "kendi yorumunu katar" mekanizması — v3'ten sarmalandı.
                self._learn_transitive(pk)
                if pk in self.memory.transitive:
                    self._derive(sk, pk, vk)
        if not wrote:
            return ""
        self.memory.lived(str(wrote), outcome=1.0,
                          about=[self.memory.self_key])
        parts = ", ".join(f"{s} → {v}" for s, v in wrote)
        message = f"Öğrendim: {parts}"
        if conflicts:
            notes = "; ".join(
                f"{s} için '{old}' biliyordum, '{new}' ile çelişiyor"
                for s, old, new in conflicts)
            message += f"\n(Çelişki fark ettim: {notes})"
        return message

    # --- türetme (geçişli akıl — v3'ten) -------------------------------
    def _learn_transitive(self, predicate):
        """Bu yüklem geçişli mi — graf ≥2 tanıklı kapalı üçgen gördü mü.
        Geçişlilik ELLE değil VERİDEN öğrenilir: "tür" is-a örneklerinden
        öğrenir; "sever" asla (sevgi zinciri grafta kapanmaz). Tek tesadüfi
        üçgen yetmez (yanlış çıkarım deliği) — en az iki bağımsız üçgen."""
        WITNESSED = 2
        if predicate is None or predicate in self.memory.transitive:
            return
        edges = [r for records in self.memory.by_subject.values()
                 for r in (self.memory.records[k] for k in records)
                 if r.predicate == predicate and r.source != "#inference"]
        forward = {}
        for r in edges:
            forward.setdefault(r.subject, set()).add(r.value)
        triangles = 0
        for _a, bs in forward.items():
            for b in bs:
                for c in forward.get(b, ()):
                    if c in bs:
                        triangles += 1
                        if triangles >= WITNESSED:
                            self.memory.transitive.add(predicate)
                            return

    def _derive(self, subject, predicate, value):
        """Yeni kenar çevresinde İKİ YÖNLÜ zincir çıkarımı. Türetilen kayıt
        #inference kaynağıyla, DÜŞÜK güvenle, gerekçesine bağlı yazılır —
        gözlem değil çıkarım olduğu ayrılabilir, kapı onu olgu gibi söyletmez."""
        forward = [(subject, r.value, [r.key])
                   for r in self.memory.about(value, touch=False)
                   if r.predicate == predicate]
        incoming = [(r.subject, value, [r.key])
                    for records in self.memory.by_subject.values()
                    for r in (self.memory.records[k] for k in records)
                    if r.predicate == predicate and r.value == subject
                    and r.subject != value]
        for who, what, because in forward + incoming:
            if who != what and not self.gate.behind(who, predicate, what):
                self.gate.inferred(who, predicate, what, because)

    # --- cevaplama -----------------------------------------------------
    def _answer(self, question, subject_label):
        """Soruya graftan getirip Qwen'le cevap; kapı, ENJEKTE edilen olguların
        dışına çıkan iddiayı düşürür."""
        subject = (link.resolve(self.memory, subject_label, self.vectors)
                   if subject_label else None)
        records = retrieve.gather(self.memory, subject)
        block = retrieve.facts_block(self.memory, records)
        raw = generate.answer(question, block)
        allowed = verify.allowed_of(self.memory, records)
        safe = verify.verify(self.memory, raw, allowed, self.mode)
        return safe or BILMIYORUM

    # --- sohbet --------------------------------------------------------
    def _chat(self, message):
        """Sohbet cevabı — kapı, sızan olgu iddiasını yine süzer.

        DELİK KAPATILDI (F2): eskiden `safe or raw` idi — verify tüm cümleleri
        desteksiz bulup düşürürse HAM (denetimsiz) çıktı dönüyordu, kapı komple
        atlanıyordu. Artık boşsa güvenli tarafa düşer, ham uydurma dönmez.
        """
        raw = generate.chat(message)
        # Sohbette allowed=∅ → cevap OLGU iddiası taşırsa (özne+değer) düşer,
        # taşımazsa (selam/tepki) geçer. Sızan uydurma olgu böyle elenir.
        safe = verify.verify(self.memory, raw, set(), self.mode)
        return safe or BILMIYORUM

    # --- bakım ---------------------------------------------------------
    def save(self):
        if self.path:
            self.memory.save(self.path)
