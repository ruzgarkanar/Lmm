"""FACTS + soru → Qwen cevap. Katman-1 topraklama: system prompt "yalnız verilen
olgulardan konuş" der. Bu TEK BAŞINA kapı DEĞİL (LLM parametrik bilgiyi
sızdırabilir) — asıl kapı verify.py'nin üretim-sonrası geri-okumasıdır.
"""
import re

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


def category_from(subject, text):
    """Metinden `subject`'in NE OLDUĞUNU TEK sade kavramla (isim) çıkarır —
    araştırma olgusu için. Latin/teknik terim değil, günlük kategori ister
    (Wikipedia ilk cümlesi taksonomi karmaşasıyla dolu; onu değil özü al)."""
    system = (f"From the text, what kind of thing is '{subject}'? Reply with ONE "
              "short everyday common-noun category, a single word. Use the SAME "
              "LANGUAGE as the text (do not translate to English). Not a "
              "latin/scientific name, not a sentence — just the one noun.")
    out = runtime.generate(text[:400], system=system, max_tokens=12,
                           temperature=0.0).strip()
    return out.strip(" .\"'")


def is_causal(message):
    """Mesaj NEDENSELLİK iddia ediyor mu (X, Y'ye neden olur)? Öyleyse (sebep,
    sonuç) döner, değilse None. YÖN kritik — few-shot ile pekiştirilir; ayrıca
    session tarafında _grounded_in ile iki varlık da mesajda mı denetlenir. Dil
    kodda değil (Qwen yargılar); is-a/soru/sohbet → None."""
    system = ("Decide if the message asserts a CAUSAL relation (X causes / leads "
              "to / results in Y). If yes, output exactly 'CAUSE: <cause> -> "
              "EFFECT: <effect>' using the head nouns, CAUSE first. If it is NOT "
              "causal (a definition, a question, small talk), output 'NONE'.\n"
              "sigara kansere neden olur -> CAUSE: sigara -> EFFECT: kanser\n"
              "aşırı stres kalp krizine yol açar -> CAUSE: stres -> EFFECT: kalp krizi\n"
              "smoking causes cancer -> CAUSE: smoking -> EFFECT: cancer\n"
              "kartal bir kuştur -> NONE\n"
              "kanser nedir -> NONE\n"
              "merhaba -> NONE")
    out = runtime.generate(message, system=system, max_tokens=30, temperature=0.0)
    m = re.search(r"CAUSE:\s*(.+?)\s*->\s*EFFECT:\s*(.+)", out, re.I)
    if not m:
        return None
    cause = m.group(1).strip(" .'\"\n")
    effect = m.group(2).strip(" .'\"\n")
    return (cause, effect) if cause and effect else None


def is_causal_question(message):
    """Nedensel soru mu, hangi YÖN? 'CAUSES: <konu>' (neyin sebebi) / 'EFFECTS:
    <konu>' (neye yol açar) / None. few-shot, dil Qwen'de. Yalnız öznenin nedensel
    kenarı varsa çağrılır (ms ön-kontrol) — boşuna model çağrısı olmasın."""
    system = ("Is this asking about the CAUSES or the EFFECTS of something?\n"
              "- what CAUSES X (why X, X'in sebebi ne, X neden olur) -> 'CAUSES: X'\n"
              "- what X CAUSES (X neye yol açar, X'in sonucu, what happens if X) "
              "-> 'EFFECTS: X'\n- otherwise -> 'NONE'\n"
              "kanserin sebebi ne -> CAUSES: kanser\n"
              "sigara neye yol açar -> EFFECTS: sigara\n"
              "stresin sonucu nedir -> EFFECTS: stres\n"
              "kanser nedir -> NONE\nmerhaba -> NONE")
    out = runtime.generate(message, system=system, max_tokens=20, temperature=0.0)
    m = re.search(r"(CAUSES|EFFECTS):\s*(.+)", out, re.I)
    if not m:
        return None
    subject = m.group(2).strip(" .'\"\n")
    return (m.group(1).lower(), subject) if subject else None


def confirm_cause(cause, effect, message):
    """Kullanıcının dilinde: nedensel olguyu öğrendiğini teyit et. Olgu grafa
    (kaynaklı) yazıldı — teyit güvenli. condition-5: cümleyi Qwen kurar."""
    system = (f"The user taught you a CAUSE→EFFECT fact and you stored it: "
              f"'{cause}' causes '{effect}'. In the SAME LANGUAGE as their message, "
              "give a short, positive acknowledgement that you learned this causal "
              "relation. One short sentence, only in that language. No extra facts.")
    return runtime.generate(message, system=system, max_tokens=50, temperature=0.3)


def is_identity_question(message):
    """Mesaj ASİSTANIN KENDİ kimliğini mi soruyor (kimsin / adın / seni kim yaptı /
    who are you / who made you)? Dış bir şeyi soruyorsa NO. Dil-bağımsız (Qwen).
    Kimlik sorularını oynak persona yerine graftan deterministik cevaplamak için."""
    # Few-shot: düz talimat Qwen-3B'ye "sana mı soruyor" kavramını veremiyordu;
    # örneklerle güvenilir. Yalnız CHAT dalında çağrılır (ASK "X nedir" buraya
    # gelmez), o yüzden "nedir"deki yanlış-pozitif akışı etkilemez.
    system = ("Classify if the message asks the responder ABOUT ITSELF — its "
              "identity, name, nature, or who made/created it. A question about "
              "some OTHER thing ('what is X') is NO. Output ONLY yes/no.\n"
              "sen kimsin -> yes\nseni kim yaptı -> yes\nadın ne -> yes\n"
              "who are you -> yes\nwho made you -> yes\n"
              "kartal nedir -> no\npangolin nedir -> no\nwhat is a dog -> no\n"
              "merhaba -> no\nteşekkürler -> no\nhava nasıl -> no")
    out = runtime.generate(message, system=system, max_tokens=3, temperature=0.0)
    return "yes" in out.strip().lower()


def identity_answer(question, name, id_block):
    """Kimlik sorusunu GRAFTAN cevaplar. Köprü ROUTE'ta kurulur (özne=self); burada
    talimat modele ADINI ('name', graftan) ve kendine dair OLGULARI verir — böylece
    'sen kimsin'→ad, 'seni kim yaptı'→üretici. 'sen/seni' zamirini çözmeye gerek
    yok. Çıktı Qwen'den (elle kalıp yok, condition-5); olgudan sapamaz."""
    system = (f"You are the assistant, and your name is '{name}'. Facts about "
              f"yourself:\n{id_block}\n\nThe user is asking about you. Reply in the "
              "SAME LANGUAGE as the question, in ONE short natural sentence: use "
              "your name for who/what you are, and these facts for who made you. "
              "Write a real, natural sentence — NEVER copy the raw fact rows or "
              "the → arrow. Use ONLY this; if something isn't covered, say you "
              "don't know. Never invent, never switch language.")
    return runtime.generate(question, system=system, max_tokens=60,
                            temperature=0.2)


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
