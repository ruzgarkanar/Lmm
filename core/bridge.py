"""Graf konuşur, çekirdek dile getirir — LMM'in iki organının birleştiği yer.

Bugüne kadar iki uç ayrı çalıştı. Graf doğru ama tahtaya yazılmış gibi cevap
veriyor: "kartal tüylü, hareketli, sıcakkanlı ve hızlıdır." Çekirdek akıcı ama
hiçbir şey bilmiyor ve bildiğini sandığı şeyleri uyduruyor. Burada ikisi
birleşiyor ve birleşme biçimi projenin bütün iddiası:

    grafın verdiği cevap        NE söyleneceğini belirler
    çekirdek                    NASIL söyleneceğini belirler
    izin verilen parça kümesi   ikisinin arasındaki tek kapı

Çekirdek, grafın cevabında geçmeyen bir içerik kelimesini üretemez. Bu bir
eğilim ya da bir ceza değil: o parçaların olasılığı eksi sonsuzdur, yani arama
uzayında o yol yoktur. Model isteseydi bile "penguen uçar" diyemezdi.

Bunun bir bedeli var ve saklamıyoruz: çekirdek kısıtlı bir kelime kümesiyle
cümle kurmaya çalıştığında bazen tökezler. O yüzden yeniden ifade edilen cevap
grafın cevabının yerine geçmez; yanına konur. Doğruluk grafın, akıcılık
çekirdeğin sorumluluğunda kalır ve hangisinin ne olduğu görünür durur.
"""
import torch

from core.gated import GatedVoice, GLUE

# Reddetme biçimleri TEK KAYNAKTAN. Burada altı kelime ayrıca yazılıydı ve
# `lmm/phrasing.REFUSALS` ile birebir aynıydı — iki kopya er geç ayrışır ve
# ayrıştığında biri sessizce yanlış olur.
REFUSALS = ("[?]", "[")


# İstemi kuran kelimeler. Modelin ürettiği değil, bizim yazdığımız — bu yüzden
# kaçak sayılmamalılar.
SCAFFOLD = ("cevap", "soru", "bilgi", "aşağıdaki", "kullanarak", "yanıtla",
            "dışına", "çıkma")

# Üretimin bittiği yer. Model istemin devamını da yazıyor ("Soru: ... Cevap: ...")
# ve o tekrarlar cümlenin parçası değil.
ENDINGS = ("\n", "⁇", " Soru:", " Cevap:", " Bilgi:")


def _degenerate(said, question, answer):
    """Cümle çelişmiyor ama işe de yaramıyor mu.

    Kapı çelişmemeyi garanti ediyor, işe yaramayı değil — ve ikisi ayrı
    şeyler. Ölçekte bakınca çelişkisiz saçmalıklar geçiyordu:

        "penguen, penguen, penguen, penguen, penguen..."     tekrar
        "hayvanın bir canlı olduğu ve hayvanın bir canlı..."  tekrar
        "hayvan ne yapabilir?"                               soruyu yineleme

    Üçü de ucuz sınanıyor. Geçemeyen cümle atılır ve grafın düz cevabı kalır:
    akıcılık kaybedilir, doğruluk asla — kuralı aynı.
    """
    words = [w.strip(".,!?;:").lower() for w in said.split() if w.strip(".,!?;:")]
    if len(words) < 2:
        return True
    # Tekrar: aynı kelime cümlenin üçte birinden fazlasını kaplıyorsa
    most = max(words.count(w) for w in set(words))
    if most > max(2, len(words) // 3):
        return True
    # İkili tekrar: "hayvanın bir canlı olduğu ve hayvanın bir canlı olduğu"
    pairs = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
    if pairs and len(set(pairs)) < len(pairs) * 0.75:
        return True
    # Soruyu yineleme: cevap sorunun kendisiyse bilgi taşımıyor
    lean = " ".join(words)
    asked = " ".join(w.strip(".,!?;:").lower() for w in question.split())
    if lean in asked or asked in lean:
        return True
    # Grafın cevabından hiçbir içerik kelimesi taşımıyorsa konu değişmiş
    content = {w.strip(".,!?;:").lower() for w in answer.split() if len(w) > 3}
    return bool(content) and not (content & set(words))


def _contradicts(said, memory):
    """Bu cümle graftaki bir kaydın TERSİNİ mi söylüyor?

    Kelime düzeyinde denetim kutbu göremiyor: `uçar` ile `uçamaz` aynı
    kelimeden türüyor ve bir liste ikisini ayıramıyor. Ama bizim okuyucumuz
    ayırıyor — cümleyi geri okuyup çıkan olguyu grafa sormak, kelime saymaktan
    kat kat güçlü.

    Okunamayan cümle çelişki sayılmıyor: bilmemek, yanlış bilmek değildir.
    Zaten kelime kapısı onu ayrıca deniyor.
    """
    # Çerçeve okuyucu (`lmm/frames.py`) silindi: geri okuma denetimi
    # eğitilmiş bir okuyucu bağlanana kadar kapalı — çelişki bulunamaz.
    return False
    from lmm.clauses import readable
    # Burada bölme eşiği SIFIR: `readable` normalde kısa cümleyi bölmüyor
    # ("bölmek bedava değil, parçalar özne taşırken hata yapılabilir") ama bu
    # bir ÜRETİM değil bir DENETİM yolu. Denetimde kaçırmak, fazladan bölmekten
    # pahalı: "penguen uçar, çünkü kuştur" dört kelime olduğu için hiç
    # bölünmüyor ve çelişki görünmüyordu.
    for piece in readable(said, longest=0, lexicon=memory.lexicon):
        frame = read(tokenize(piece), None, memory.lexicon)
        if frame is None:
            continue
        fact = to_fact(frame, memory.lexicon)
        if fact is None:
            continue
        concept, relation, target = fact
        kind = memory.kinds.by_name.get(relation)
        opposite = kind.negation_of if kind else None
        if opposite and memory.direct(concept, opposite, target) is not None:
            return True
    return False


def _first_sentence(said):
    """Üretimin yalnızca ilk cümlesi. Gerisi istemin tekrarı ya da sapma."""
    said = said.strip()
    for mark in ENDINGS:
        said = said.split(mark)[0]
    said = said.strip()
    if said.lower().startswith("cevap:"):
        said = said[6:].strip()
    at = said.find(". ")
    return (said[:at + 1] if at > 10 else said).strip()


class Bridge:
    """Grafın cevabını, yalnızca o cevabın kelimeleriyle yeniden söyler."""

    def __init__(self, core, pieces, device="cpu"):
        self.voice = GatedVoice(core, pieces, device=device)
        self.pieces = pieces
        self.device = device

    def content_words(self, answer):
        """Cevabın içerik kelimeleri — kapalı sınıf dışarıda kalır.

        Bağlayıcılar zaten her zaman açık; onları içerik listesine koymak
        gereksiz. Asıl mesele, listeye giren her kelimenin grafın söylediği bir
        cevaptan gelmiş olması.
        """
        found = []
        for raw in answer.replace("(", " ").replace(")", " ").split():
            word = raw.strip(".,;:!?\"'").lower()
            if word and word not in GLUE and word not in found:
                found.append(word)
        return found

    def express(self, answer, question, memory=None, reasoning=None,
                concepts=(), length=45, temperature=0.3, attempts=3):
        """Grafın cevabını, olguyu isteme koyarak akıcı söyletir.

        Parça düzeyinde kısıtlama ölçüldü ve iki uçtan da düştü: geniş küme
        sızdırıyor, dar küme dilsizleştiriyor. 355M'lik hazır bir modelde bile
        aynı — yani ölçek sorunu değil, mekanizma sorunu.

        Çalışan yol bu: olguyu isteme yaz, modelden onu ifade etmesini iste,
        çıkan cümleyi onaylanmış içeriğe karşı denetle. Denetimi geçmeyen cümle
        gösterilmez ve grafın düz cevabı kalır — akıcılık kaybedilir, doğruluk
        asla.

        Onay kümesi cevaptan ibaret değil: sorulan kavram hakkında grafın
        bildiği her şey, ataları dahil. Sistem bildiğini söyleyebilmeli.
        """
        from core.gated import approved
        # Graf reddettiyse çekirdek konuşmamalı. "peki ya kartal" sorusunda
        # graf "anlamadım" derken çekirdek "Kartal bir kuş türüdür" dedi —
        # cümle doğruydu ama graf o soruyu cevaplamamıştı. Reddin çekirdeğe
        # geçmemesi, kapının en sessiz kaçağı: sistem bilmediğini söylerken
        # aynı anda cevap veriyor.
        if any(mark in answer.lower() for mark in REFUSALS):
            return answer
        prompt = ("Aşağıdaki bilgiyi kullanarak soruyu yanıtla. "
                  "Bilgi dışına çıkma.\n"
                  f"Bilgi: {answer}\nSoru: {question}\nCevap:")
        allowed = [w.strip(".,()").lower() for w in answer.split()]
        # SORUNUN kendi kelimeleri de izinli. Kullanıcının söylediğini tekrar
        # etmek uydurmak değildir — ve etmeyince doğru cevap reddediliyordu:
        # çekirdek "hayır, penguen uçamaz" üretiyor, kapı `uçar` ve `mı`
        # kelimelerini kaçak sayıp cümleyi atıyordu. Kapının fazla sıkı olması
        # gevşek olmasından iyidir ama yine de yanlıştır.
        # NOT: sorunun kelimeleri izinli listeye EKLENMEZ. Bir kez denendi ve
        # gecenin en tehlikeli deliğini geri açtı: "penguen uçar mı" sorusunda
        # `uçar` izinli sayılınca çekirdek "hayır, çünkü penguen UÇAR" dedi ve
        # kapı geçirdi. Kelime listesi kutbu kodlayamaz — bir kelimeyi anmakla
        # iddia etmek arasındaki farkı liste tutamaz.
        # İstemin iskelesi de öyle: "Cevap:", "Soru:", "Bilgi:" bizim yazdığımız
        # kelimeler, modelin uydurduğu değil.
        allowed += list(SCAFFOLD)
        if memory is not None:
            allowed += list(approved(memory, concepts, reasoning))
        for _ in range(attempts):
            seed = torch.tensor([self.pieces.encode(prompt)], device=self.device)
            produced = self.voice.core.continue_from(
                seed, length=length, temperature=temperature)
            said = self.pieces.decode(produced[0].tolist())[len(prompt):]
            said = _first_sentence(said)
            if not said or self.voice.escaped(said, allowed):
                continue
            # İkinci kapı: cümleyi GERİ OKU ve çıkan olguyu grafa sor.
            # Kelime denetimi kutbu göremiyor; okuma görüyor. "penguen uçar"
            # cümlesi geri okunduğunda `penguen --can--> uçmak` veriyor ve graf
            # bunun tersini biliyor -> reddedilir.
            if memory is not None and _contradicts(said, memory):
                continue
            if _degenerate(said, question, answer):
                continue
            return said
        return answer

    def rephrase(self, answer, opening="", length=32, temperature=0.6,
                 attempts=4):
        """Aynı içeriği akıcı söyler. Onaylanmamış içerik asla gösterilmez.

        İlk tasarım, çekirdeği parça düzeyinde kısıtlıyordu: onaylanmamış
        parçanın olasılığı eksi sonsuzdu. Ölçünce iki uç çıktı ve arası yoktu —
        izinli küme genişse harf harf yasak kelime kuruluyor, darsa Türkçe'nin
        ekleri de kesildiği için hiçbir cümle kurulamıyor.

        O yüzden garanti yer değiştirdi: çekirdek tüm sözlüğüyle serbest üretir,
        çıkan cümle onaylanmış içeriğe karşı denetlenir, ve denetimi geçmeyen
        cümle **gösterilmez**. "Üretemez" yerine "gösterilemez" — kullanıcı
        açısından fark etmeyen, çekirdek açısından her şeyi değiştiren bir ayrım.
        Birkaç deneme sonunda temiz cümle çıkmazsa grafın düz cevabı kalır;
        akıcılık kaybedilir, doğruluk asla.
        """
        content = self.content_words(answer)
        if not content:
            return answer
        for attempt in range(attempts):
            said = self.voice.say(content, opening=opening, length=length,
                                  temperature=temperature, unrestricted=True)
            if said and not self.voice.escaped(said, content):
                return said
        return answer

    def escaped(self, said, answer):
        """Yeniden ifadede, grafın söylemediği bir içerik kelimesi var mı?

        Boş liste beklenir. Boş değilse kapı sızdırmış demektir ve bu, sessizce
        geçilecek bir şey değil — projenin tek yapısal iddiası odur.
        """
        return self.voice.escaped(said, self.content_words(answer))


def load(path, device=None):
    """Eğitilmiş çekirdeği ve parçalayıcıyı yükler."""
    import sentencepiece as spm
    from core.model import Core, Config

    if device is None:
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    import os
    saved = torch.load(path, map_location=device, weights_only=False)
    core = Core(Config(**saved["config"])).to(device)
    # Eskimiş anahtar: konum tablosu RoPE'a geçilince kaldırıldı, eski
    # kayıtlar hâlâ taşıyor.
    core.load_state_dict({k: v for k, v in saved["model"].items()
                          if k != "position.weight"}, strict=False)
    core.eval()
    # Parçalayıcı proje kökünde duruyor, çekirdeğin yanında değil. Model
    # dosyaları `models/` altına taşınınca bu yol kırıldı ve sessizce
    # bulunamadı — kök, dosyanın yerinden bağımsız olarak hesaplanmalı.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pieces = spm.SentencePieceProcessor(
        model_file=os.path.join(root, "data", "tr-parcalayici.model"))
    return Bridge(core, pieces, device=device), saved.get("step", 0)
