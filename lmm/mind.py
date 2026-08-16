"""Mind — respond()'un mesajsız, sinyal-güdümlü İKİZİ. Otonom gelişim döngüsü.

respond() TEPKİSELdir: mesaj gelir → cevap. Mind PROAKTİFtir: mesaj gelmese de
iç sinyallerden (çelişki basıncı, bilgi boşluğu) iş çıkarır — çelişkilerini çözer,
türetir, damıtır, (ileride) araştırır. "Kendi kendine düşünen, kendi büyüyen akıl."

İKİ HIZ (net ayrım):
  ms-iç   : türet / çöz / damıt — SAF GRAF, model/web YOK, mikro-saniye
  yavaş-dış: araştır — Qwen+web, saniyeler, bütçeli+onaylı (Adım 4, henüz yok)

Bu YAŞAM DEĞİL: graf üstünde zamanlanmış, sınırlı, devam-ettirilebilir hesap.
Sinyaller boşalınca DİNLENİR — substrat-özgür değil. Sayı ≠ anlayış.

Bu dosya şimdilik yalnız SAF-MS çekirdeği (Adım 0-2): sıfır model/web/thread riski.
Yavaş-dış araştırma + guardrail'lar (bütçe, #auto damgası, frontier dondurma) ayrı
tartışılıp eklenecek.
"""
from v3 import dynamics
from v3.memory import DOCUMENT
from lmm import generate, link
from lmm import research as web


class Mind:
    """Bir Session'ı sarar; iç sinyallerden proaktif iş çıkarır. Session (konuşma)
    dokunulmaz; ikisi aynı `memory`'yi paylaşır."""

    def __init__(self, session):
        self.session = session
        self.memory = session.memory

    # --- tek tik (ms, saf graf) ----------------------------------------
    def step(self):
        """Bir düşünme tiki — bedava (ms) yapısal hamle. Öncelik: ÇELİŞKİ çöz >
        TÜRET (geçişli kapanış) > DAMIT (uyku). Dönen: yapılan eylemin SAYISAL
        raporu (dict). Model/web YOK — deterministik, çevrimdışı.

        (Adım 3'te if-sıra yerine dürtü/maliyet argmax'e yükselecek; ama davranış
        zaten 'ucuz yapısal hamleyi yeğle' — model-yargısı eylem yok.)"""
        # 1) ÇELİŞKİ varsa çöz — en yüksek basınçlı kaydı hakemle
        pres = dynamics.pressure(self.memory)
        if pres:
            record = self.memory.records.get(pres[0][0])
            if record is not None:
                dynamics.arbitrate(self.memory, record)
                return {"action": "resolve", "kalan_celisi": len(pres) - 1}
        # 2) TÜRET — geçişli yüklemlerde eksik #inference kenarlarını doldur
        derived = self._derive_closure()
        if derived:
            return {"action": "derive", "turetildi": derived}
        # 3) DAMIT — uyku (settle/fade/prune). Bir şey değişmezse doygun.
        report = dynamics.sleep(self.memory)
        return {"action": "distill", **report}

    def _derive_closure(self):
        """Geçişli yüklemlerde eksik türetimleri kapat. Önce geçişliliği veriden
        öğren (_learn_transitive), sonra her kenar çevresinde _derive. ms, saf graf.
        Dönen: bu turda türetilen yeni #inference kenar sayısı."""
        for pk in list(self.memory.by_predicate.keys()):
            self.session._learn_transitive(pk)
        new = 0
        for pk in list(self.memory.transitive):
            for key in list(self.memory.by_predicate.get(pk, ())):
                record = self.memory.records.get(key)
                if record is None or record.source == "#inference":
                    continue
                before = len(self.memory.records)
                self.session._derive(record.subject, pk, record.value)
                new += len(self.memory.records) - before
        return new

    # --- döngü (doygunlukta durur) -------------------------------------
    def run(self, max_steps=200):
        """Sinyal ya da bütçe bitene dek düşün. DOYGUNLUKTA durur (dinlenir):
        çelişki yok + türetilecek yok + damıtım hiçbir şey değiştirmiyor. Zamanlayıcı
        durumu geçici; graf'tan yeniden hesaplanır → durup devam bedava."""
        log = []
        for _ in range(max_steps):
            report = self.step()
            log.append(report)
            if report["action"] == "distill" and not any(
                    report.get(k) for k in
                    ("settled", "faded", "judged", "pruned", "distilled")):
                break                       # doygun — dinlen
        return log

    # --- yarı-otonom: merakını YÜZEYE çıkar, onayla doldur ------------
    def wonder(self, most=3):
        """ms-loop doygunlaştıktan sonra mind'ın DOLDURAMADIĞI en büyük boşluklar
        — 'araştırmak istediklerim'. Web'e DOKUNMAZ, yalnız öneri döner. Semi-
        otonom güvenlik: döngü kendi başına web'e çıkmaz; insan onaylar (guardrail).
        curiosity() zaten yüklem/self düğümlerini süzüyor."""
        return self.session.curiosity(most=most)

    def research(self, subject_label):
        """İNSAN ONAYIYLA bir boşluğu web'den doldur — INGEST-ONLY, otonom-güvenli:
        wiki_summary(başlık) → category_from → gate.admit(#web, DÜŞÜK güven). Sentetik
        soru YOK (başlık zaten etiket → condition-5), konuşma üretimi YOK — sadece
        kaynaklı+düşük-güvenli yutma. Provenans (#web:url) denetim için kalır; web
        güvenilmez olduğu için asla CERTAIN, operatör olgusunu asla ezmez."""
        text, url = web.wiki_summary(subject_label)
        if not text:
            return False
        value = generate.category_from(subject_label, text)
        sk = link.resolve(self.memory, subject_label, self.session.vectors,
                          create=True)
        vk = (link.resolve(self.memory, value, self.session.vectors, create=True)
              if value else None)
        if not value or sk is None or vk is None or sk == vk:
            return False
        self.session.gate.admit(sk, None, vk, f"#web:{url}", DOCUMENT)
        self.memory.lived(f"auto-research:{subject_label}", outcome=1.0,
                          about=[self.memory.self_key])
        return True

    def status(self):
        """Gelişim nabzı — sayı, dil değil (projenin etiği). Bebek ne kadar 'büyüdü'."""
        return {
            "olgu": len(self.memory.records),
            "kavram": len(self.memory.identities),
            "acik_bosluk": len(dynamics.gaps(self.memory)),
            "acik_celiski": len(dynamics.pressure(self.memory)),
            "turetilmis": sum(1 for r in self.memory.records.values()
                              if r.source == "#inference"),
        }
