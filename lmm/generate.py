"""FACTS + soru → Qwen cevap. Katman-1 topraklama: system prompt "yalnız verilen
olgulardan konuş" der. Bu TEK BAŞINA kapı DEĞİL (LLM parametrik bilgiyi
sızdırabilir) — asıl kapı verify.py'nin üretim-sonrası geri-okumasıdır.
"""
from lmm import prompts, runtime


def answer(question, facts_block, warmth=0.2):
    """Soruya, yalnız verilen olgularla, akıcı Türkçe cevap. Olgu yoksa model
    'bilmiyorum' demeye yönlendirilir (system prompt)."""
    if facts_block.strip():
        user = f"OLGULAR:\n{facts_block}\n\nSORU: {question}"
    else:
        user = (f"SORU: {question}\n\n"
                "(Belleğinde bu konuda kayıtlı olgu YOK. Uydurma; "
                "bilmediğini söyle.)")
    return runtime.generate(user, system=prompts.ANSWER_SYSTEM,
                            max_tokens=200, temperature=warmth)


def chat(message, identity_block="", warmth=0.7):
    """Sohbet cevabı (selam, teşekkür, küçük konuşma). Olgu iddiası taşırsa
    verify süzer — yani doğal konuş, ama uydurulan olgu yine çıkışta düşer.

    `identity_block`: kimlik olguları (graftan, ör. lmm→üretici→rüzgar).
    ENJEKTE edilir ki "seni kim yaptı" gibi sorular personaya değil GRAFA
    dayansın — dil-bağımsız (İngilizce persona Türkçe'de zayıf kalıyordu;
    enjekte olgu her dilde çalışır). Olgu graftan geldiği için verify'dan geçer.
    """
    system = prompts.CHAT_SYSTEM
    if identity_block.strip():
        system += (
            "\n\nFACTS about yourself, as [entity relation value] rows from your "
            "memory:\n" + identity_block +
            "\n\nIf the user asks who/what you are or who made/created you, answer "
            "using these facts, but write a natural sentence IN THE USER'S "
            "LANGUAGE — never copy the raw rows or the arrow. If a fact is not "
            "listed, do not invent it.")
        # Kimlik olgusaldır — ılımlı düşük sıcaklık: model dodge/ramble yerine
        # olguyu daha tutarlı söylesin (0.7 örneklemesi bazen sapıyordu, 0.2
        # ise fazla tutuk kalıp reddediyordu).
        warmth = 0.4
    return runtime.generate(message, system=system,
                            max_tokens=120, temperature=warmth)


def are_rivals(a, b):
    """İki değer AYNI şey hakkında söylendi: bir arada var olabilir mi, yoksa
    birbirini DIŞLAYAN alternatif mi? Çelişki tespiti için — dil-bağımsız (kuralı
    biz yazmıyoruz, Qwen yargılar). Dönen: True = rakip (çelişki), False = bir arada.

    "kuş"/"yırtıcı" → bir arada (kartal ikisi de) → False.
    "kuş"/"balık"   → dışlayan → True.  "paris"/"berlin" (tek başkent) → True."""
    system = (
        "Two labels were each stated about the SAME single entity. Decide if "
        "they can BOTH hold at once, or are mutually EXCLUSIVE.\n"
        "Key idea: labels on DIFFERENT dimensions coexist (a category + a trait; "
        "a color + a shape). Two labels filling the SAME dimension (two species, "
        "two cities as the one capital, two opposite sizes) are exclusive.\n"
        "Answer ONE word: COEXIST or EXCLUSIVE.\n"
        "bird + predator -> COEXIST\n"
        "organ + muscle -> COEXIST\n"
        "red + round -> COEXIST\n"
        "bird + fish -> EXCLUSIVE\n"
        "big + small -> EXCLUSIVE\n"
        "Paris + Berlin -> EXCLUSIVE")
    out = runtime.generate(f"{a} + {b}", system=system,
                           max_tokens=4, temperature=0.0)
    return "EXCLUSIVE" in out.upper()


# --- DİNAMİK SİSTEM SÖZLERİ (dil-bağımsız) --------------------------------
# "Bunu bilmiyorum", "Öğrendim" gibi sistem cümleleri ELLE Türkçe yazılmaz —
# Qwen kullanıcının dilinde üretir. Böylece tez (tüm diller, elle dil yok)
# sistemin kendi ağzında da tutar; İngilizce derse İngilizce, Almanca derse
# Almanca teyit/ret alır.

def hedge_note(source_label, message):
    """Kullanıcının dilinde KISA bir çekince NOTU üretir (olgu YOK; yalnız 'bu
    bilgi şu kaynaktan, emin değilim' anlamı). Doğrulanmış cevaba EKLENİR; cevabın
    kendisi DEĞİŞMEZ — böylece hedge adımı asla uydurma ekleyemez (yalnız not,
    üstelik olgu taşımadığı için verify'dan da geçer). condition-5: notu Qwen
    kurar, elle kalıp yok."""
    system = ("In the SAME LANGUAGE as the user's message, write ONE very short "
              "caveat, in parentheses, meaning: the statement is not certain and "
              f"comes from this source: {source_label}. Contain NO facts — only "
              "the caveat and the source. Keep it under 8 words.")
    return runtime.generate(message, system=system, max_tokens=32, temperature=0.3)


def is_affirmative(message):
    """Kullanıcı mesajı ONAY/evet/'devam et' mi? (araştırma teklifine yanıt).
    Dil-bağımsız (Qwen). Dönen: True=onay."""
    system = ("Does the user's message mean YES / go ahead / approval, as opposed "
              "to no or a different request? Answer exactly ONE word: YES or NO.")
    out = runtime.generate(message, system=system, max_tokens=3, temperature=0.0)
    return "YES" in out.upper()


def offer_research(subject, message):
    """Kullanıcının dilinde: 'bunu bilmiyorum, araştırayım mı?' (önce-sor)."""
    system = (f"You do NOT have information about '{subject}' in your memory. In "
              "the SAME LANGUAGE as the user's message, briefly say you don't know "
              "it yet and ASK whether you should look it up. One short sentence, "
              "phrased as an offer/question. Invent no facts.")
    return runtime.generate(message, system=system, max_tokens=40, temperature=0.3)


def refusal(message):
    """Kullanıcının dilinde: 'bu bilgi bende yok'. Uydurma yasak, kısa."""
    system = ("The user asked about something that is NOT in your memory. Reply "
              "ONLY in the SAME LANGUAGE as their message (never switch to "
              "another language), briefly and honestly saying you don't have "
              "that information yet. Do NOT invent any fact. One short sentence.")
    return runtime.generate(message, system=system, max_tokens=50,
                            temperature=0.3)


def confirm(learned, conflicts, message):
    """Kullanıcının dilinde: öğrenilen olguyu kısaca teyit et (+ çelişki notu).

    `learned`: [(özne, değer)] · `conflicts`: [(özne, eski, yeni)]. Olgular
    zaten grafa yazıldı (desteklenmiş) — teyit güvenli.
    """
    facts = "; ".join(f"{s} = {v}" for s, v in learned)
    system = (f"The user taught you a new fact and you have stored it in your "
              f"memory: {facts}. In the SAME LANGUAGE as their message, give a "
              f"short, positive acknowledgement that you learned and remembered "
              f"it. Do NOT apologize. Do NOT say you forgot.")
    if conflicts:
        clash = "; ".join(f"{s}: previously '{o}', now '{n}'"
                          for s, o, n in conflicts)
        system += (f" Note: this conflicts with what you already knew: {clash}. "
                   f"Gently mention the conflict.")
    system += (" One or two short, natural sentences, ONLY in the same language "
               "as the user — never mix languages. Do NOT add any other facts.")
    return runtime.generate(message, system=system, max_tokens=90,
                            temperature=0.3)
