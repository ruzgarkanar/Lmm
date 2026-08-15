"""FACTS + soru → Qwen cevap. Katman-1 topraklama: system prompt "yalnız verilen
olgulardan konuş" der. Bu TEK BAŞINA kapı DEĞİL (LLM parametrik bilgiyi
sızdırabilir) — asıl kapı verify.py'nin üretim-sonrası geri-okumasıdır.
"""
from lmm import prompts, runtime


def answer(question, facts_block, warmth=0.4):
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


def chat(message, warmth=0.7):
    """Sohbet cevabı (selam, teşekkür, küçük konuşma). Olgu iddiası taşırsa
    verify süzer — yani doğal konuş, ama uydurulan olgu yine çıkışta düşer."""
    return runtime.generate(message, system=prompts.CHAT_SYSTEM,
                            max_tokens=120, temperature=warmth)
