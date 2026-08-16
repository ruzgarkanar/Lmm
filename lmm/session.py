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
import re

from v3 import dynamics
from v3.dataset import fold
from v3.gate import Gate
from v3.memory import Memory, OPERATOR, STRANGER, DOCUMENT
from lmm import extract, generate, link, research, retrieve, verify

SLEEP_EVERY = 50   # kaç turda bir uyku (damıt/sol/hakemle) — v3 §117

# KAYNAK-GÜVEN eşiği: bunun ALTINDAKİ güvenle konuşulan olgu cevapta ETİKETLENİR
# (çekince + kaynak). Operatör-öğretisi (0.75) kesin → etiketsiz; belge (0.6),
# damıtım (0.5), web (düşük) → "emin değilim, ...'e göre". "en sıkı" yanlış-bilgi.
CERTAIN = 0.7

BILMIYORUM = "Bunu bilmiyorum."


def _grounded_in(value, message):
    """Öğretilecek DEĞER kullanıcının MESAJINDA gerçekten geçiyor mu — Qwen
    uydurmadı mı? "söylemediğin şeyi öğretemezsin". Fold + kök eşleşmesi
    (kuş~kuştur). Bu, extract'ın soruyu ("atom nedir") WRITE sanıp olmayan bir
    değer ("birleşik") icat edip grafa yazmasını engeller — dil-bağımsız."""
    want = {fold(w) for w in re.findall(r"\w+", value) if len(w) >= 3}
    if not want:
        return True        # kısa/tokensiz değer — engelleme (nadir)
    have = {fold(w) for w in re.findall(r"\w+", message) if len(w) >= 3}
    return any(a == b or a.startswith(b) or b.startswith(a)
               for a in want for b in have)


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
        self.last_written = []  # bu turda kapının kabul ettiği üçlüler (hasat)
        self.history = []       # KISA-VADELİ konuşma bağlamı (son N tur) — graf
        # uzun-vadeli hafıza; bu, "az önce ne konuştuk" bağlamı (sohbet sürekliliği)
        self.who = who
        self.level = OPERATOR if who == "#operator" else STRANGER
        self.mode = mode
        self.vectors = {}
        self.turns = 0
        if self.memory.self_key is None:
            self.memory.self_key = self.memory.identify("#self")
        self._identity = self._seed_identity()
        # NEDENSELLİK yüklemi — nedensel olgu (sebep→sonuç) NORMAL Record olarak
        # bu ayrılmış yüklemle durur; tüm kapı/verify/ters-indeks/türetme makinesini
        # bedava miras alır. is-a'dan (tür) FİZİKSEL olarak ayrı → verify/çelişki
        # ilişki-tipini karıştırmasın. Etiket dil-nötr (#causes).
        self._causes_key = link.resolve(self.memory, "#causes", self.vectors,
                                        create=True)

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
        # HİYERARŞİK BAĞ (GRAF — LMM'in gücü): iki değer is-a zinciriyle bağlıysa
        # (kedigil→memel: kedigil bir memelidir) RAKİP DEĞİL, bir arada var olurlar
        # — "aslan hem kedigil hem memeli" çelişki değil hiyerarşidir. Qwen'e
        # sormadan grafla çöz (are_rivals fazla ateşliyordu). Yalnız BAĞSIZ
        # değerlerde Qwen'e sor (kuş/balık gibi).
        if self._connected(old_key, new_key):
            self._rival_cache[ck] = False
            return False
        old = link.label_of(self.memory, old_key)
        new = link.label_of(self.memory, new_key)
        try:
            verdict = bool(old and new and generate.are_rivals(old, new))
        except Exception:                                   # noqa: BLE001
            verdict = False        # emin değilsek çelişki sayma (bozma)
        self._rival_cache[ck] = verdict
        return verdict

    def _connected(self, a, b, depth=4):
        """a ile b graf'ta is-a zinciriyle bağlı mı (her iki yön). Bağlıysa
        hiyerarşiktir → çelişki değil. Sınırlı BFS (döngü-korumalı)."""
        for start, goal in ((a, b), (b, a)):
            seen, frontier = {start}, [start]
            for _ in range(depth):
                nxt = []
                for node in frontier:
                    for k in self.memory.by_subject.get(node, ()):
                        v = self.memory.records[k].value
                        if v == goal:
                            return True
                        if v not in seen:
                            seen.add(v)
                            nxt.append(v)
                if not nxt:
                    break
                frontier = nxt
        return False

    # --- NEDENSELLİK (ms, sembolik — sinirsel değil) -------------------
    def learn_cause(self, cause_label, effect_label, source=None):
        """Nedensel olgu öğret: sebep→sonuç, KAPIDAN geçerek (Qwen yazamaz).
        #causes yüklemiyle normal Record → is-a'dan ayrı. Şimdilik doğrudan;
        sonraki adımda extract nedensel cümleyi buraya bağlayacak."""
        ck = link.resolve(self.memory, cause_label, self.vectors, create=True)
        ek = link.resolve(self.memory, effect_label, self.vectors, create=True)
        if ck is None or ek is None or ck == ek:
            return False
        record, _ = self.gate.admit(ck, self._causes_key, ek,
                                    source or self.who, self.level)
        return record is not None

    def _learn_causal(self, message):
        """Nedensel cümleyi yakala → learn_cause. is_causal (Qwen, YÖNLÜ few-shot)
        + _grounded_in (iki varlık da mesajda mı — uydurma varlık engeli). Değilse
        None → normal is-a yazımına düşer. DÜRÜST SINIR: _grounded_in varlığı
        doğrular ama YÖNÜ değil (3B ters çıkarabilir — tasarımın en büyük riski)."""
        try:
            pair = generate.is_causal(message)
        except Exception:                                   # noqa: BLE001
            return None
        if not pair:
            return None
        cause, effect = pair
        if not (_grounded_in(cause, message) and _grounded_in(effect, message)):
            return None                    # varlık mesajda yok → Qwen uydurdu
        if not self.learn_cause(cause, effect):
            return None
        self.memory.lived(f"cause:{cause}->{effect}", outcome=1.0,
                          about=[self.memory.self_key])
        return generate.confirm_cause(cause, effect, message) or "OK"

    def _causal_answer(self, message, direction, subject):
        """Nedensel soruyu cevapla: ms-traversal (sebep/sonuç) → Qwen cümleye
        döker → verify denetler. Boşluksa reddet (uydurma yok). DÜRÜST SINIR:
        verify._has_edge yüklem-körü (is-a/causes ayırmaz) — nedensel-farkında
        anchor sonraki iş; şimdilik kenar VAR olduğu için desteksiz düşmez."""
        if direction == "causes":
            edges = [(c, subject) for c in self.causes_of(subject)]
        else:
            edges = [(subject, e) for e in self.effects_of(subject)]
        if not edges:
            return generate.refusal(message) or BILMIYORUM
        # CAUSE/EFFECT etiketli blok (iç iskele) — "kanserin sebebi ne" gibi TERS
        # yönlü soruda model oku çevirip sebebi bulabilsin (etiketsiz ok'ta
        # takılıyordu). Etiket iç prompt yapısı, çıktı dili değil.
        block = "\n".join(f"[{i}] CAUSE: {c}  EFFECT: {e}"
                          for i, (c, e) in enumerate(edges, 1))
        raw = generate.answer(message, block)
        # SÖZCÜK-KAPSAMA kapısı (benchmark bulgusu): nedensel cümle is-a
        # kalıbına uymaz — reextract "yağarsa"yı değer sanıp DOĞRU cevabı
        # düşürüyordu. İlke: cevabın TÜM içerik-sözcükleri verilen blok+sorudan
        # geliyorsa yeni iddia YOKTUR → uydurma yapısal olarak imkânsız, geç.
        # Dışına çıkan sözcük varsa eski sıkı verify işler. Dil kuralı değil —
        # küme kapsaması.
        given = {fold(w) for w in re.findall(r"\w+", block + " " + message)
                 if len(w) >= 3}
        raw_words = {fold(w) for w in re.findall(r"\w+", raw or "")
                     if len(w) >= 3}
        if raw_words and raw_words <= given:
            return raw
        allowed = set()
        for c, e in edges:
            allowed.add(link.resolve(self.memory, c))
            allowed.add(link.resolve(self.memory, e))
        safe = verify.verify(self.memory, raw, allowed, self.mode, anchor="edge")
        return safe or generate.refusal(message) or BILMIYORUM

    def causes_of(self, effect_label):
        """effect'in SEBEPLERİ (etiket). ms: by_value ters indeksi, O(gelen-derece).
        Saf graf yürüyüşü — sinirsel üretim yok."""
        ek = link.resolve(self.memory, effect_label, self.vectors)
        if ek is None:
            return []
        return [link.label_of(self.memory, self.memory.records[k].subject)
                for k in self.memory.by_value.get(ek, ())
                if self.memory.records[k].predicate == self._causes_key]

    def effects_of(self, cause_label):
        """cause'un SONUÇLARI (etiket). ms: by_subject, O(giden-derece)."""
        ck = link.resolve(self.memory, cause_label, self.vectors)
        if ck is None:
            return []
        return [link.label_of(self.memory, self.memory.records[k].value)
                for k in self.memory.by_subject.get(ck, ())
                if self.memory.records[k].predicate == self._causes_key]

    def root_causes(self, effect_label, depth=4):
        """effect'e giden nedensel ZİNCİR — sınırlı-derinlik BFS (döngü-korumalı).
        Kök sebeplere kadar geri yürür. Yol/ara-adımları döndürür; X→Z'yi OTOMATİK
        YAZMAZ (nedensellik her zaman geçişli değil — condition-4 güvenli). ms."""
        ek = link.resolve(self.memory, effect_label, self.vectors)
        if ek is None:
            return []
        seen, frontier, chain = {ek}, [ek], []
        for _ in range(depth):
            nxt = []
            for node in frontier:
                for k in self.memory.by_value.get(node, ()):
                    r = self.memory.records[k]
                    if r.predicate != self._causes_key or r.subject in seen:
                        continue
                    seen.add(r.subject)
                    chain.append((link.label_of(self.memory, r.subject),
                                  link.label_of(self.memory, node)))
                    nxt.append(r.subject)
            if not nxt:
                break
            frontier = nxt
        return chain            # [(sebep, sonuç)] kenarları — kök→yaprak zinciri

    def respond(self, message):
        """Bir mesaja cevap + KONUŞMA BAĞLAMINI günceller. Asıl mantık _respond'da;
        bu sarmalayıcı son N turu `history`'de tutar (sohbet sürekliliği).
        `last_written`: bu turda KAPININ kabul ettiği üçlüler — konsolidasyon
        hasadı için (kapı-onaylı = güvenilir eğitim hedefi; modelin kendi ham
        çıktısı DEĞİL)."""
        self.last_written = []
        said = self._respond(message)
        if message and message.strip():
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": said or ""})
            self.history = self.history[-12:]     # son ~6 tur (kayan pencere)
        return said

    def _respond(self, message):
        """Bir mesaja cevap. Dönen daima metin; asla desteksiz olgu.

        Sağlamlık (kod-denetimi): boş mesaj korunur, tüm akış try/except içinde
        (tek bozuk üretim konuşmayı çökertmesin — güvenli cevaba düşer)."""
        if not message or not message.strip():
            return ""
        # ARAŞTIRMA ONAYI (önce-sor): geçen turda "araştırayım mı?" teklif
        # edildiyse, bu mesaj ONAY mı diye bak. Onaysa çek+öğren; değilse teklifi
        # bırak ve mesajı olağan işle (yeni soru olabilir).
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
            # ARAŞTIRMA ONAYI (mimari — içerik karar verir): geçen tur "araştırayım
            # mı?" teklif edildiyse, bu mesaj YENİ İÇERİK mi (öğretme/soru) yoksa
            # saf onay mı — EXTRACT söyler. Yeni içerik = yeni tur (pending düşer,
            # aşağıda normal işlenir); yalnız içeriksiz-olumlama araştırmayı
            # tetikler. Eskiden körlemesine is_affirmative çağrılıyordu; "biliyor
            # musun balina da memelidir" yanlışlıkla "evet" sanılıp turu kaçırıyordu.
            if self._pending is not None:
                (subj, orig_q), self._pending = self._pending, None
                new_content = bool(op["triples"] and op["triples"][0][0])
                if not new_content:
                    try:
                        if generate.is_affirmative(message):
                            return self._research(subj, orig_q)
                    except Exception:                       # noqa: BLE001
                        pass
                # yeni içerik ya da onay değil → aşağıda normal işlenir
            if op["kind"] == extract.WRITE and op["triples"]:
                # NEDENSEL cümle mi ("X, Y'ye neden olur") — is-a'dan AYRI sakla
                # (#causes yüklemi). Öyleyse learn_cause; değilse normal is-a yazımı.
                caused = self._learn_causal(message)
                if caused:
                    return caused
                said = self._write(op["triples"], message)
                if said:
                    return said
                # Boş yazım (yanlış WRITE sınıflaması / değersiz üçlü — ör. "X
                # nedir" yanlışlıkla WRITE geldi): uydurma yerine SORU gibi ele
                # al → getir/reddet. Düşer.
            subject = op["triples"][0][0] if op["triples"] else None
            subject_key = (link.resolve(self.memory, subject, self.vectors)
                           if subject else None)
            # KİMLİK ROUTE (mimari köprü): özne ÇÖZÜLEMİYORSA (CHAT ya da "seni
            # kim yaptı" gibi extract'ın "sen"i çözemediği durum) bu bir KİMLİK
            # sorusu mu — deterministik sınıflandır. Öyleyse extract'ın kumarını
            # ATLA, kimlik olgusunu _lmm_key'den GARANTİ getiren yola sok. Yalnız
            # özne-çözülemeyende çağrılır (net "kartal nedir"de ekstra çağrı yok).
            if subject_key is None:
                try:
                    if generate.is_identity_question(message):
                        return self._identity_reply(message)
                except Exception:                           # noqa: BLE001
                    pass
            # NEDENSEL SORU mu — YALNIZ özne nedensel kenar taşıyorsa Qwen'e sor
            # (ms ön-kontrol: causes_of/effects_of boş değilse). Boşuna model
            # çağrısı yok; nedensel bilgisi olmayan özne için hiç sorulmaz.
            if subject and (self.causes_of(subject) or self.effects_of(subject)):
                try:
                    cq = generate.is_causal_question(message)
                except Exception:                           # noqa: BLE001
                    cq = None
                if cq:
                    return self._causal_answer(message, cq[0], cq[1])
            if op["kind"] in (extract.WRITE, extract.ASK):
                return self._answer(message, subject)
            return self._chat(message)
        except Exception:                                   # noqa: BLE001
            return BILMIYORUM

    # --- kimlik (deterministik route) ----------------------------------
    def _identity_reply(self, message):
        """KİMLİK sorusunu deterministik cevapla. Kök sebep (denetim): kimlik
        sorusu extract'ta ASK olup 'sen' çözülemeyince olgu HİÇ getirilmiyordu.
        Burada olguyu _lmm_key'den GARANTİ getir; adı + olguları identity_answer'a
        ver; verify anchor='value' (özne öz-referans zamir → None; edge yolu
        kimliği düşürürdü, nesne 'rüzgar' allowed'da çapa)."""
        id_records = retrieve.gather(self.memory, self._lmm_key)
        id_block = "\n".join(
            f"{link.label_of(self.memory, r.subject)} "
            f"{link.label_of(self.memory, r.predicate)} → "
            f"{link.label_of(self.memory, r.value)}" for r in id_records)
        name = link.label_of(self.memory, self._lmm_key)
        raw = generate.identity_answer(message, name, id_block)
        safe = verify.verify(self.memory, raw, self._identity, self.mode,
                             anchor="value")
        return safe or generate.refusal(message) or BILMIYORUM

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
            if not _grounded_in(value, message):
                continue     # değer mesajda yok → Qwen uydurdu ("atom nedir"→
                             # "birleşik"): soruyu WRITE sanma tuzağı, yazma
            sk = link.resolve(self.memory, subject, self.vectors, create=True)
            vk = link.resolve(self.memory, value, self.vectors, create=True)
            # sk == vk: ÖZ-DÖNGÜ ("almanya → almanya") — bir şey kendisi olamaz,
            # anlamsız kayıt. extract ara sıra üretiyordu; kapıdan geçirme.
            if sk is None or vk is None or sk == vk or (sk, vk) in seen:
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
                # konsolidasyon hasadı: kapıdan geçen ADAY üçlünün kendisi
                # (düğüm etiketi DEĞİL — düğüm yanlış-birleşmiş olabilirdi ve
                # extractor'a girdide olmayan metin üretmek öğretilirdi;
                # code-review bulgusu #3). Bunlar extract._clean çıktısı:
                # fold'lu, mesajla topraklanmış.
                self.last_written.append((subject, predicate or "", value))
                # TÜRETME (geçişli akıl): "kartal→kuş, kuş→hayvan ⊢ kartal→
                # hayvan". Geçişlilik veriden öğrenilir (≥2 tanıklı üçgen),
                # çıkarım #inference kaynağıyla düşük güvenle yazılır. Bu,
                # "kendi yorumunu katar" mekanizması — v3'ten sarmalandı.
                self._learn_transitive(pk)
                if pk in self.memory.transitive:
                    self._derive(sk, pk, vk)
        if not wrote:
            return ""
        # YAŞANTI (tasarım notu — denetim "recall/weigh kullanılmıyor" dedi):
        # lmm'de yaşantı BİLEREK yaz-only. v3'te deneyim, aday cevapları sürprizle
        # TARTIYORDU; lmm'de cevap olgudan+kapıdan gelir, aday-tartma YOK — o
        # yüzden recall/weigh burada anlamsız. lived() bir "ne yaptım" günlüğüdür:
        # uyku/damıtım onu budar, provenans için durur. Zorla bağlamak kullanılmayan
        # karmaşa ekler (sulandırır), sağlamlaştırmaz.
        self.memory.lived(str(wrote), outcome=1.0,
                          about=[self.memory.self_key])
        # DİNAMİK DİL: teyit cümlesi ELLE Türkçe değil — Qwen kullanıcının
        # dilinde üretir (öğrenilen olgu + varsa çelişki). Olgular grafa
        # yazıldığı için teyit güvenli.
        said = generate.confirm(wrote, conflicts, message)
        return said or "OK"

    # --- doküman yutma (RAG'siz öğrenme) --------------------------------
    def learn_text(self, text, source="#document"):
        """Bir METNİ (doküman/paragraf) grafa yut — embed/chunk-RAG'in yerine.

        RAG metni saklayıp sorguda benzerlikle PARÇA arar; LMM metni bir kez
        OKUYUP olgulara çevirir, kapıdan yazar — cevap sorguda ms graf-yürüyüşü,
        çok-adımlı türetim bedava (RAG parça-birleştiremez). Cümle cümle:
        reextract (olgu iddiaları; selam/yorum → []) → _write ile aynı korumalar
        (topraklama, öz-döngü) → DOCUMENT güveni (CERTAIN altı → cevapta
        kaynak-etiketli, operatör olgusunu ezemez). Dönen: (yazılan, atlanan)."""
        wrote, skipped = 0, 0
        seen = set()          # _write ile aynı: aynı üçlü iki cümlede geçerse
        sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text)
                     if s.strip()]
        for sent in sentences:
            # NEDENSELLİK de dokümandan öğrenilir ("Yağmur yağarsa bataklık
            # büyür") — benchmark bunu yakaladı: is_causal yalnız sohbet
            # yolundaydı, dokümandaki neden-sonuç hiç yutulmuyordu.
            causal = generate.is_causal(sent)
            if causal and _grounded_in(causal[0], sent) \
                    and _grounded_in(causal[1], sent):
                self.learn_cause(causal[0], causal[1], source=source)
                wrote += 1
                continue
            for subject, predicate, value in extract.reextract(sent):
                if not value or not _grounded_in(value, sent):
                    skipped += 1
                    continue
                sk = link.resolve(self.memory, subject, self.vectors,
                                  create=True)
                vk = link.resolve(self.memory, value, self.vectors,
                                  create=True)
                if sk is None or vk is None or sk == vk or (sk, vk) in seen:
                    skipped += 1
                    continue
                seen.add((sk, vk))
                pk = (link.resolve(self.memory, predicate, self.vectors,
                                   create=True) if predicate else None)
                record, _ = self.gate.admit(sk, pk, vk, source, DOCUMENT)
                if record is None:
                    skipped += 1
                    continue
                wrote += 1
                self._learn_transitive(pk)
                if pk in self.memory.transitive:
                    self._derive(sk, pk, vk)
        if wrote:
            self.memory.lived(f"document:{source}:{wrote}", outcome=1.0,
                              about=[self.memory.self_key])
        return wrote, skipped

    # --- türetme (geçişli akıl — v3'ten) -------------------------------
    def _learn_transitive(self, predicate):
        """Bu yüklem geçişli mi — graf ≥2 tanıklı kapalı üçgen gördü mü.
        Geçişlilik ELLE değil VERİDEN öğrenilir: "tür" is-a örneklerinden
        öğrenir; "sever" asla (sevgi zinciri grafta kapanmaz). Tek tesadüfi
        üçgen yetmez (yanlış çıkarım deliği) — en az iki bağımsız üçgen."""
        WITNESSED = 2
        if predicate is None or predicate in self.memory.transitive:
            return
        # TERS İNDEKS: tüm grafı değil, YALNIZ bu yüklemli kayıtları gez (O(N)→O(derece)).
        edges = [r for r in (self.memory.records[k]
                             for k in self.memory.by_predicate.get(predicate, ()))
                 if r.source != "#inference"]
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
        # TERS İNDEKS: subject'i DEĞER alan kayıtları by_value'dan al (O(N)→O(derece)).
        incoming = [(r.subject, value, [r.key])
                    for r in (self.memory.records[k]
                              for k in self.memory.by_value.get(subject, ()))
                    if r.predicate == predicate and r.subject != value]
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
        # GERÇEK BOŞLUK (gaps sinyali) — grafta bu özne hakkında HİÇ olgu YOKSA,
        # araştırmayı ÜRETİMDEN ÖNCE teklif et. Böylece modelin nazik "bilmiyorum"u
        # (verify'ı geçip safe'i doldurur) teklifi ENGELLEMEZ (Bug #2). Boşluk =
        # merak = araştır. Olgu varken buraya girilmez → bildiğini araştırmaya
        # kaçmaz.
        if not records:
            if subject_label:
                self._pending = (subject_label, question)   # önce-sor
                offer = generate.offer_research(subject_label, question)
                return offer or generate.refusal(question) or BILMIYORUM
            return generate.refusal(question) or BILMIYORUM
        # HEDEFLİ KENAR (çok-adım — benchmark bulgusu): soruda ikinci bir
        # BİLİNEN kavram geçiyorsa ("zilfen bir canlı mıdır" → 'canlı') ve graf
        # o kenarı biliyorsa (türetilmiş dahil), o kaydı ÖNE al — graf türetmişti
        # ama cevap seçici başka olguyu seslendiriyordu. Saf graf, ms, dil yok.
        qwords = {fold(w) for w in re.findall(r"\w+", question) if len(w) >= 3}
        qwords.discard(fold(subject_label or ""))
        targeted = []
        for r in records:
            vlab = fold(link.label_of(self.memory, r.value))
            if vlab and any(w == vlab or vlab.startswith(w) or w.startswith(vlab)
                            for w in qwords):
                targeted.append(r)
        if targeted:
            records = targeted + [r for r in records if r not in targeted]
        # Olgu VAR → grounded cevap + çıkış kapısı
        block = retrieve.facts_block(self.memory, records)
        raw = generate.answer(question, block)
        allowed = verify.allowed_of(self.memory, records)
        safe = verify.verify(self.memory, raw, allowed, self.mode, anchor="edge")
        if not safe:
            # olgu var ama üretim tökezledi (MPS/örnekleme) → bir kez yeniden dene;
            # yine düşerse bildiğini "araştırayım mı" diye SORMAZ, güvenli reddeder.
            safe = verify.verify(self.memory, generate.answer(question, block),
                                 allowed, self.mode, anchor="edge")
            if not safe:
                return generate.refusal(question) or BILMIYORUM
        # KAYNAK-GÜVEN (en sıkı): en zayıf olgu CERTAIN altındaysa kaynak+çekince.
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
        # TEK sade kategori çıkar (taksonomi karmaşası değil özü) → temiz olgu,
        # temiz cevap. Çok değerli/latin terim küçük modeli boğuyordu.
        value = generate.category_from(subject_label, text)
        sk = link.resolve(self.memory, subject_label, self.vectors, create=True)
        vk = (link.resolve(self.memory, value, self.vectors, create=True)
              if value else None)
        if not value or sk is None or vk is None:
            return generate.refusal(question) or BILMIYORUM
        # #web damgası + DOCUMENT (0.6 < CERTAIN) → cevapta kaynak-etiketli.
        self.gate.admit(sk, None, vk, f"#web:{url}", DOCUMENT)
        self.memory.lived(f"web:{subject_label}", outcome=1.0,
                          about=[self.memory.self_key])
        return self._answer(question, subject_label)     # artık grafta → cevap+hedge

    # --- sohbet --------------------------------------------------------
    def _chat(self, message):
        """Sohbet cevabı — kapı, sızan olgu iddiasını yine süzer.

        DELİK KAPATILDI (F2): eskiden `safe or raw` idi — verify tüm cümleleri
        desteksiz bulup düşürürse HAM (denetimsiz) çıktı dönüyordu, kapı komple
        atlanıyordu. Artık boşsa güvenli tarafa düşer, ham uydurma dönmez.
        """
        id_records = retrieve.gather(self.memory, self._lmm_key)
        # KİMLİK bloğu YÜKLEMLİ (üretici gizlenmesin) — "kim yaptı" yanıtlanabilsin.
        id_block = "\n".join(
            f"{link.label_of(self.memory, r.subject)} "
            f"{link.label_of(self.memory, r.predicate)} → "
            f"{link.label_of(self.memory, r.value)}" for r in id_records)
        # KONUŞMA BAĞLAMI: son turları da ver → sohbet sürekliliği ("araştır"ın
        # neyi, "ne yapıyorsun"un bağlamı korunur). Uydurma yine verify'da süzülür.
        raw = generate.chat(message, id_block, history=self.history)
        # Sohbette allowed = yalnız KİMLİK olguları. anchor="value": özne öz-
        # referanslı zamir (ben/beni) çözülemez, NESNE'nin (rüzgar) izinli olması
        # yeter; dış uydurma (Google) yine düşer. Bkz. verify.verify.
        safe = verify.verify(self.memory, raw, self._identity, self.mode,
                             anchor="value")
        return safe or generate.refusal(message) or BILMIYORUM

    # --- LMM gücü: öz-farkındalık (merak + çelişki basıncı) ------------
    def curiosity(self, most=5):
        """Sistem neyi BİLMEDİĞİNİ bilir: hakkında hiç/yalnız-episodik kaydı olan,
        ama konuşmada geçmiş kavramların etiketleri. `dynamics.gaps`'i sarar (o
        organ v3'te vardı, lmm akışına bağlı değildi). Proaktif araştırmanın
        girdisi: 'şunu bilmiyorum, bakayım mı'. Kendi/kimlik düğümleri hariç."""
        # Yüklem düğümlerini (tür/özellik gibi ilişki etiketleri) merak sayma —
        # kavram değiller. Kendi/kimlik düğümleri de hariç.
        predicates = {r.predicate for r in self.memory.records.values()
                      if r.predicate is not None}
        out = []
        for key, _gap in dynamics.gaps(self.memory, most=most * 4):
            if key in self._identity or key in predicates:
                continue
            label = link.label_of(self.memory, key)
            if label and label not in out:
                out.append(label)
            if len(out) >= most:
                break
        return out

    def tension(self, most=5):
        """Sistem çelişkiden RAHATSIZ olur: en yüksek basınçlı çelişkiler
        (özne, [rakip değerler]). `dynamics.pressure`'ı sarar. Kullanıcıya
        'şu konuda çelişkili bilgim var' diye yüzeye çıkarılabilir."""
        from v3.memory import CONTRA
        out, seen = [], set()
        for rkey, _p in dynamics.pressure(self.memory):
            r = self.memory.records.get(rkey)
            if r is None or rkey in seen:
                continue
            rivals = [self.memory.records[k] for kind, k in r.links
                      if kind == CONTRA and k in self.memory.records]
            seen.add(rkey)
            seen.update(x.key for x in rivals)
            subject = link.label_of(self.memory, r.subject)
            values = [link.label_of(self.memory, r.value)] + [
                link.label_of(self.memory, x.value) for x in rivals]
            out.append((subject, values))
            if len(out) >= most:
                break
        return out

    # --- bakım ---------------------------------------------------------
    def save(self):
        if self.path:
            self.memory.save(self.path)
