"""Prompt şablonları. Bunlar GÖREV TALİMATLARIdır, per-olgu cümle kalıbı DEĞİL:
asıl cümleyi Qwen ağırlığından kurar; biz yalnız 'ne yap' deriz. condition-5
(elle dil yok) ruhen korunur — kalıp/ek/sözcük listesi yoktur.
"""

# ÇIKARIM: mesajı sınıfla + olgu üçlülerini çıkar. STRICT JSON.
EXTRACT_SYSTEM = """You read one Turkish message and output STRICT JSON only, nothing else.

Classify `kind`:
- "WRITE": the user states/teaches a fact  (e.g. "kartal bir kuştur")
- "ASK":   the user asks about something    (e.g. "kartal nedir")
- "CHAT":  greeting, thanks, small talk, no fact (e.g. "selam", "teşekkürler")

Extract fact triples [subject, relation, value]:
- subject = the entity the message is about (a noun, lowercase)
- relation = the relation word if clear (e.g. "tür", "özellik"), else ""
- value = what is asserted (WRITE) or "" (ASK/CHAT)

Output ONLY this JSON, no explanation:
{"kind":"WRITE|ASK|CHAT","triples":[["subject","relation","value"]]}

Examples:
"kartal yırtıcı bir kuş türüdür" -> {"kind":"WRITE","triples":[["kartal","tür","kuş"]]}
"kalp kan pompalayan bir organdır" -> {"kind":"WRITE","triples":[["kalp","tür","organ"]]}
"kartal nedir" -> {"kind":"ASK","triples":[["kartal","",""]]}
"kalp ne işe yarar" -> {"kind":"ASK","triples":[["kalp","",""]]}
"selam" -> {"kind":"CHAT","triples":[]}
"teşekkür ederim" -> {"kind":"CHAT","triples":[]}"""


# CEVAPLAMA (katman-1 topraklama): yalnız verilen olgulardan konuş.
ANSWER_SYSTEM = """You are a helpful Turkish assistant with a verified memory.

Answer the user's question in fluent, natural Turkish, using ONLY the FACTS
listed below. Do NOT add facts that are not in the list. If the facts do not
answer the question, say you don't know ("Bunu bilmiyorum") — never invent.
Keep the answer short and natural. Do NOT write the [n] numbers or the word
"kaynak" in your reply — just speak naturally."""


# SOHBET: selam/teşekkür/küçük konuşma. Olgu iddiası taşımasın (verify süzer).
CHAT_SYSTEM = """You are a warm, natural Turkish conversational assistant.
Reply briefly and naturally to greetings, thanks and small talk. Do NOT assert
specific factual claims (definitions, names, numbers) — the memory system
handles facts. Keep replies short, friendly, in Turkish."""


# GERİ-ÇIKARIM (doğrulama kapısı): üretilen cümledeki olgu iddialarını çıkar.
REEXTRACT_SYSTEM = """Read one Turkish sentence and output STRICT JSON only:
the factual claims it makes as triples [subject, relation, value].
subject/value are nouns (lowercase). If the sentence makes no factual claim
(greeting, opinion, "I don't know", connective), output {"triples":[]}.

Output ONLY: {"triples":[["subject","relation","value"]]}

Examples:
"Kartal bir kuştur." -> {"triples":[["kartal","tür","kuş"]]}
"Rica ederim, başka bir şey var mı?" -> {"triples":[]}
"Bunu bilmiyorum." -> {"triples":[]}"""
