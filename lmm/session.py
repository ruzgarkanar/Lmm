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
from v3.memory import Memory, OPERATOR, STRANGER, DOCUMENT
from lmm import extract, generate, link, research, retrieve, verify

SLEEP_EVERY = 50   # kaç turda bir uyku (damıt/sol/hakemle) — v3 §117

# KAYNAK-GÜVEN eşiği: bunun ALTINDAKİ güvenle konuşulan olgu cevapta ETİKETLENİR
# (çekince + kaynak). Operatör-öğretisi (0.75) kesin → etiketsiz; belge (0.6),
# damıtım (0.5), web (düşük) → "emin değilim, ...'e göre". "en sıkı" yanlış-bilgi.
CERTAIN = 0.7

BILMIYORUM = "Bunu bilmiyorum."


class Session:
    """Bir konuşma. Qwen (dil) + graf/kapı (doğruluk & büyüme)."""

    def __init__(self, path=None, who="#operator", mode="STRICT"):
        self.path = path
        self.memory = (Memory.load(path) if path and os.path.exists(path)
                       else Memory())
        self.gate = Gate(self.memory)
        # Çelişki için anlamsal RAKİP kontrolünü Qwen'e bağla (cache'li) — yüklemsiz
        # durumda "kuş/yırtıcı" bir arada, "kuş/balık" çelişki. Bkz. _are_rivals.
        self._rival_cache = {}
        self.gate.rival = self._are_rivals
        self._pending = None    # araştırma teklif edilen özne (önce-sor akışı)
        self.who = who
        self.level = OPERATOR if who == "#operator" else STRANGER
        self.mode = mode
        self.vectors = {}
        self.turns = 0
        if self.memory.self_key is None:
            self.memory.self_key = self.memory.identify("#self")
        self._identity = self._seed_identity()

    def _seed_identity(self):
        """KİMLİK grafa olgu olarak: (lmm → üretici → rüzgar). Böylece kimlik
        de 'bilinen bilgi'dir — kapıdan geçer, dil-bağımsız söylenir. Persona
        (prompts.CHAT_SYSTEM) Qwen'e adını verir; bu tohum onu grafta tutar."""
        lmm = link.resolve(self.memory, "lmm", self.vectors, create=True)
        ruzgar = link.resolve(self.memory, "rüzgar", self.vectors, create=True)
        maker = link.resolve(self.memory, "üretici", self.vectors, create=True)
        if self.gate.behind(lmm, maker, ruzgar) is None:
            self.gate.admit(lmm, maker, ruzgar, "#operator", OPERATOR)
        self._lmm_key = lmm    # kimlik öznesi — _chat gather bunu kullanır
        return {lmm, ruzgar, maker, self.memory.self_key}

    def _are_rivals(self, old_key, new_key):
        """İki değer anlamsal RAKİP mi (aynı yuva, birbirini dışlayan)? Qwen
        yargılar (dil-bağımsız), sonuç cache'lenir — aynı çift bir daha model
        çağırmaz. gate._contradiction bunu yüklemsiz çelişki kararında kullanır."""
        ck = frozenset((old_key, new_key))
        if ck in self._rival_cache:
            return self._rival_cache[ck]
        old = link.label_of(self.memory, old_key)
        new = link.label_of(self.memory, new_key)
        try:
            verdict = bool(old and new and generate.are_rivals(old, new))
        except Exception:                                   # noqa: BLE001
            verdict = False        # emin değilsek çelişki sayma (bozma)
        self._rival_cache[ck] = verdict
        return verdict

    def respond(self, message):
        """Bir mesaja cevap. Dönen daima metin; asla desteksiz olgu.

        Sağlamlık (kod-denetimi): boş mesaj korunur, tüm akış try/except içinde
        (tek bozuk üretim konuşmayı çökertmesin — güvenli cevaba düşer)."""
        if not message or not message.strip():
            return ""
        # ARAŞTIRMA ONAYI (önce-sor): geçen turda "araştırayım mı?" teklif
        # edildiyse, bu mesaj ONAY mı diye bak. Onaysa çek+öğren; değilse teklifi
        # bırak ve mesajı olağan işle (yeni soru olabilir).
        if self._pending is not None:
            (subj, orig_q), self._pending = self._pending, None
            try:
                if generate.is_affirmative(message):
                    return self._research(subj, orig_q)
            except Exception:                               # noqa: BLE001
                pass
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
                said = self._write(op["triples"], message)
                if said:
                    return said
                # Boş yazım (yanlış WRITE sınıflaması / değersiz üçlü — ör. "X
                # nedir" yanlışlıkla WRITE geldi): uydurma yerine SORU gibi ele
                # al → getir/reddet. Düşer.
            if op["kind"] in (extract.WRITE, extract.ASK):
                # ASK ama özne çıkmadıysa: boş özneyle dene → BILMIYORUM'a düşer
                subject = op["triples"][0][0] if op["triples"] else None
                return self._answer(message, subject)
            return self._chat(message)
        except Exception:                                   # noqa: BLE001
            return BILMIYORUM

    # --- yazma ---------------------------------------------------------
    def _write(self, triples, message):
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
        # DİNAMİK DİL: teyit cümlesi ELLE Türkçe değil — Qwen kullanıcının
        # dilinde üretir (öğrenilen olgu + varsa çelişki). Olgular grafa
        # yazıldığı için teyit güvenli.
        said = generate.confirm(wrote, conflicts, message)
        return said or "OK"

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
        # associative=False: cevap YALNIZ doğrudan olgulardan kurulur (kenar-
        # denetimiyle uyum — bkz. retrieve.gather / verify._has_edge).
        records = retrieve.gather(self.memory, subject, associative=False)
        block = retrieve.facts_block(self.memory, records)
        raw = generate.answer(question, block)
        allowed = verify.allowed_of(self.memory, records)
        edges = verify.edges_of(records)          # KENAR denetimi (yeniden-birleşim uydurmasını bloklar)
        safe = verify.verify(self.memory, raw, allowed, self.mode,
                             anchor="edge", edges=edges)
        if not safe:
            # BİLMİYORUM → körlemesine reddetme: ARAŞTIRMAYI TEKLİF ET (önce-sor).
            # Özne varsa teklifi kur; kullanıcı onaylarsa sonraki tur çekilir.
            if subject_label:
                self._pending = (subject_label, question)   # özne + ORİJİNAL soru
                offer = generate.offer_research(subject_label, question)
                if offer:
                    return offer
            return generate.refusal(question) or BILMIYORUM
        # KAYNAK-GÜVEN (en sıkı): cevabı temellendiren en zayıf olgu CERTAIN
        # altındaysa kesinlik düşür + kaynağı belirt. Operatör olguları etiketsiz.
        if records:
            weakest = min(records, key=lambda r: r.trust)
            if weakest.trust < CERTAIN:
                safe = self._hedge(safe, weakest, question) or safe
        return safe

    def _hedge(self, answer, record, message):
        """Düşük güvenli olguya dayanan DOĞRULANMIŞ cevaba çekince NOTU ekler.
        Cevap metni değişmez (uydurma eklenemez), yalnız sonuna kaynak+çekince."""
        source = record.source or ""
        label = source[5:] if source.startswith("#web:") else "a stored source"
        note = generate.hedge_note(label, message)
        return f"{answer} {note}".strip() if note else answer

    # --- agentic araştırma (onayla, webden öğren) ----------------------
    def _research(self, subject_label, question):
        """Kullanıcı onayladı → Wikipedia'dan çek, üçlü çıkar, #web+düşük güvenle
        grafa yaz, sonra ORİJİNAL soruyu normal cevapla (kaynak-etiketli). Web
        GÜVENİLMEZ: olgu 'biliyorum' diye değil kaynak damgalı+düşük güvenle
        girer (condition-4)."""
        text, url = research.wiki_summary(subject_label)
        if not text:
            return generate.refusal(question) or BILMIYORUM
        # Özetin ilk cümlesi genelde "X, bir Y'dir" — Qwen üçlüyü çıkarır.
        first = text.split(".")[0][:240]
        wrote = False
        for _s, _p, value in (extract.extract(first).get("triples") or []):
            if not value:
                continue
            sk = link.resolve(self.memory, subject_label, self.vectors, create=True)
            vk = link.resolve(self.memory, value, self.vectors, create=True)
            if sk is None or vk is None:
                continue
            # #web damgası + DOCUMENT (0.6 < CERTAIN) → cevapta kaynak-etiketli.
            self.gate.admit(sk, None, vk, f"#web:{url}", DOCUMENT)
            wrote = True
        if not wrote:
            return generate.refusal(message) or BILMIYORUM
        self.memory.lived(f"web:{subject_label}", outcome=1.0,
                          about=[self.memory.self_key])
        return self._answer(message, subject_label)     # artık grafta → cevap+hedge

    # --- sohbet --------------------------------------------------------
    def _chat(self, message):
        """Sohbet cevabı — kapı, sızan olgu iddiasını yine süzer.

        DELİK KAPATILDI (F2): eskiden `safe or raw` idi — verify tüm cümleleri
        desteksiz bulup düşürürse HAM (denetimsiz) çıktı dönüyordu, kapı komple
        atlanıyordu. Artık boşsa güvenli tarafa düşer, ham uydurma dönmez.
        """
        # KİMLİK olgularını graftan getir ve üretime ENJEKTE et — böylece
        # "seni kim yaptı" personaya değil grafa dayanır (dil-bağımsız).
        # Kimlik bloğu YÜKLEMLİ kurulur: facts_block yüklemi (üretici) gizler,
        # model ilişkiyi bilmeden "lmm → rüzgar" görüp "kim yaptı"yı yanıtlayamaz.
        id_records = retrieve.gather(self.memory, self._lmm_key)
        id_block = "\n".join(
            f"{link.label_of(self.memory, r.subject)} "
            f"{link.label_of(self.memory, r.predicate)} → "
            f"{link.label_of(self.memory, r.value)}" for r in id_records)
        raw = generate.chat(message, id_block)
        # Sohbette allowed = yalnız KİMLİK olguları (lmm/rüzgar/self). Böylece
        # "ben LMM'im, beni Rüzgar yaptı" gibi kimlik cümlesi geçer ama DIŞ
        # dünya olgu iddiası (grafta yoksa) yine düşer — uydurma sızmaz.
        # anchor="value": kimlik cevabında özne öz-referanslı zamir (ben/beni),
        # güvenilir çözülemez; iddia edilen NESNE'nin (rüzgar) allowed'da olması
        # yeter. Özne gerçek düğüme çözülürse kenar/izin aranır → "Python'u Rüzgar
        # yazdı" düşer. Dış uydurma (Google) yine bloklanır. Bkz. verify.verify.
        id_edges = verify.edges_of(id_records)
        safe = verify.verify(self.memory, raw, self._identity, self.mode,
                             anchor="value", edges=id_edges)
        return safe or generate.refusal(message) or BILMIYORUM

    # --- bakım ---------------------------------------------------------
    def save(self):
        if self.path:
            self.memory.save(self.path)
