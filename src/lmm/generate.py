"""FACTS + question → Qwen answer. Layer-1 grounding: the system prompt says
"speak only from the given facts". This ALONE is NOT the gate (the LLM can leak
parametric knowledge) — the real gate is verify.py's post-generation read-back.
"""
import re

from lmm import prompts, runtime


def answer_prompt(question, facts_block):
    """The user turn an answering call is built from — ONE definition.

    STRUCTURAL LABELS, NOT LANGUAGE. The two words framing this prompt used to
    be Turkish ("OLGULAR"/"SORU"), which made every answer in the system pass
    through a Turkish skeleton — a question in German was answered from a
    Turkish-labelled frame. They are field names for the engine, so they are in
    the one language the whole codebase already speaks. The ANSWER LANGUAGE is
    not affected by them: ANSWER_SYSTEM binds the reply to the question's own
    language, and the no-facts branch repeats it, because that branch has no
    facts to borrow a language from.

    It is a function rather than two f-strings because the LoRA data builder
    has to reproduce this frame EXACTLY — it was a hand-copied duplicate, and
    when the frame changed here the training data kept teaching the old one.
    """
    if facts_block.strip():
        return f"FACTS:\n{facts_block}\n\nQUESTION: {question}"
    return (f"QUESTION: {question}\n\n"
            "(There is NO recorded fact about this in your memory. Do not "
            "invent anything; say that you do not know — writing that in "
            "the SAME LANGUAGE as the question above.)")


def answer(question, facts_block, warmth=0.2):
    """Answer the question fluently, using only the given facts. If there is no
    fact, the model is steered to say 'I don't know' (system prompt)."""
    return runtime.generate(answer_prompt(question, facts_block),
                            system=prompts.ANSWER_SYSTEM,
                            max_tokens=200, temperature=warmth)


def chat(message, identity_block="", warmth=0.7, history=None):
    """Chat reply (greeting, thanks, small talk). If it carries a fact claim,
    verify filters it — so speak naturally, but a fabricated fact still drops at
    the exit.

    `history`: recent turns [{role,content}] — conversational CONTINUITY (what
    we just talked about). So "what are you doing" and follow-up questions are
    answered with context.

    `identity_block`: identity facts (from the graph, e.g. lmm→creator→rüzgar).
    It is INJECTED so questions like "who made you" rest on the GRAPH, not on a
    persona — language-independent (an English persona stayed weak in Turkish;
    an injected fact works in every language). Since the fact comes from the
    graph, it passes verify.
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
        # Identity is factual — moderately low temperature: let the model state
        # the fact more consistently instead of dodging/rambling (0.7 sampling
        # sometimes drifted, while 0.2 stayed too stiff and refused).
        warmth = 0.4
    # CONVERSATION CONTEXT: fold recent turns into messages, current message last.
    messages = list(history or [])
    messages.append({"role": "user", "content": message})
    return runtime.generate(messages, system=system,
                            max_tokens=120, temperature=warmth)


def are_rivals(a, b):
    """Two values were stated about the SAME thing: can they coexist, or are
    they mutually EXCLUSIVE alternatives? For contradiction detection —
    language-independent (we don't write the rule, Qwen judges). Returns:
    True = rivals (contradiction), False = coexist.

    "bird"/"predator" → coexist (an eagle is both) → False.
    "bird"/"fish"     → exclusive → True.  "paris"/"berlin" (one capital) → True."""
    system = (
        "Two labels were each stated about the SAME single entity. Decide if "
        "they can BOTH hold at once, or are mutually EXCLUSIVE.\n"
        "Key idea: labels on DIFFERENT dimensions coexist (a category + a trait; "
        "a color + a shape). Two labels filling the SAME dimension (two species, "
        "two cities as the one capital, two opposite sizes) are exclusive.\n"
        "Answer ONE word: COEXIST or EXCLUSIVE.\n"
        # The examples are deliberately in SEVERAL languages: the judgement is
        # about meaning, not about English, and a single-language example set
        # teaches the pattern in that language only.
        "bird + predator -> COEXIST\n"
        "órgano + músculo -> COEXIST\n"
        "kırmızı + yuvarlak -> COEXIST\n"
        "bird + fish -> EXCLUSIVE\n"
        "groß + klein -> EXCLUSIVE\n"
        "Paris + Berlin -> EXCLUSIVE")
    out = runtime.generate(f"{a} + {b}", system=system,
                           max_tokens=4, temperature=0.0)
    return "EXCLUSIVE" in out.upper()


# --- DYNAMIC SYSTEM UTTERANCES (language-independent) ---------------------
# System sentences like "I don't know this", "Learned" are NOT hand-written —
# Qwen produces them in the user's language. So the thesis (all languages, no
# hand-written language) holds in the system's own mouth too: speak English,
# get an English confirmation/refusal; German, German.
#
# WHICH ONLY HOLDS IF THE ENGINE ACTUALLY MATCHES. Measured on the English
# corpus: asked "what is melvarit", the research offer came back in DUTCH, and
# before the answer prompt was tightened the English absence questions were
# answered "Ich weiß es nicht." These utterances carry no evidence sentence to
# take their language from — the user's message is the only cue, and a rule
# buried at the end of an instruction was not enough. _MATCH_LANGUAGE goes
# FIRST in every one of them, and says what to do rather than what to be: look
# at the message, identify its language, write in that one.
_MATCH_LANGUAGE = (
    "FIRST look at the user's message below and identify what language it is "
    "written in. Write your entire reply in THAT language and no other — not "
    "English unless the message is English, not a language that merely "
    "resembles it. ")

def supported(answer, block):
    """SECOND-TIER support check: does the evidence block REALLY say this answer
    — called when the coverage gate trips on innocent narrative words (the
    connectives and reporting verbs any language sprinkles through a sentence).
    The digit discipline (digits_ok) comes BEFORE this check
    and is non-negotiable; this only judges word-level residue. Strict: when in
    doubt, no → the answer drops (false-negative is safe, false-positive is
    dangerous)."""
    out = runtime.generate(f"EVIDENCE:\n{block}\n\nCLAIM: {answer}",
                           system=prompts.SUPPORT_SYSTEM, max_tokens=4,
                           temperature=0.0)
    return out.strip().lower().startswith("yes")


def answers_asked(question, answer, block):
    """THE RELATION READ-BACK: does the evidence state the relation THE QUESTION
    ASKS — not merely everything the answer happens to assert.

    `supported` judges a CLAIM, and a claim carries only what it chooses to say.
    An answer that voices a true fact from the evidence passes it whatever was
    asked, so the one remaining way to a confident WRONG answer stayed open:
    answer a different relation with real material (measured — asked who SIGNED
    the document, the answer gave the PREPARER field, coverage 1.0, read-back
    yes). Here the proposition comes from the QUESTION, so an answer can no
    longer pass by leaving out what was asked for. Strict: unsure = no, and no
    means the turn abstains."""
    out = runtime.generate(
        f"EVIDENCE:\n{block}\n\nQUESTION: {question}\n\nANSWER: {answer}",
        system=prompts.RELATION_SYSTEM, max_tokens=4, temperature=0.0)
    return out.strip().lower().startswith("yes")


def hedge_note(source_label, message):
    """Produces a SHORT caveat NOTE in the user's language (NO facts; only the
    meaning 'this information is from this source, I'm not certain'). It is
    APPENDED to the verified answer; the answer itself does NOT change — so the
    hedge step can never add fabrication (only a note, and since it carries no
    fact it also passes verify). condition-5: Qwen composes the note, no
    hand-written template."""
    system = (_MATCH_LANGUAGE +
              "Write ONE very short "
              "caveat, in parentheses, meaning: the statement is not certain and "
              f"comes from this source: {source_label}. Contain NO facts — only "
              "the caveat and the source. Keep it under 8 words.")
    return runtime.generate(message, system=system, max_tokens=32, temperature=0.3)


def category_from(subject, text):
    """Extracts WHAT `subject` IS from the text as ONE plain concept (noun) —
    for a research fact. It wants an everyday category, not a Latin/technical
    term (Wikipedia's first sentence is full of taxonomy clutter; take the
    essence, not that)."""
    system = (f"From the text, what kind of thing is '{subject}'? Reply with ONE "
              "short everyday common-noun category, a single word. Use the SAME "
              "LANGUAGE as the text (do not translate to English). Not a "
              "latin/scientific name, not a sentence — just the one noun.")
    out = runtime.generate(text[:400], system=system, max_tokens=12,
                           temperature=0.0).strip()
    return out.strip(" .\"'")


def is_causal(message):
    """Does the message assert CAUSALITY (X causes Y)? If so, returns (cause,
    effect), else None. DIRECTION is critical — reinforced with few-shot; on the
    session side _grounded_in additionally checks that both entities are in the
    message. Language is not in the code (Qwen judges); is-a/question/chat → None."""
    system = ("Decide if the message asserts a CAUSAL relation (X causes / leads "
              "to / results in Y). If yes, output exactly 'CAUSE: <cause> -> "
              "EFFECT: <effect>' using the head nouns, CAUSE first. If it is NOT "
              "causal (a definition, a question, small talk), output 'NONE'.\n"
              # MULTILINGUAL BY DESIGN: the examples must not teach the pattern
              # in one language, or the classifier learns that language's cue
              # words instead of the relation. Every sentence here is invented
              # and belongs to no document this system is measured on.
              "smoking causes cancer -> CAUSE: smoking -> EFFECT: cancer\n"
              "la sequía provoca hambruna -> CAUSE: sequía -> EFFECT: hambruna\n"
              "Frost führt zu Rissen im Beton -> CAUSE: Frost -> EFFECT: Risse\n"
              "aşırı gürültü uykusuzluğa yol açar -> CAUSE: gürültü -> EFFECT: uykusuzluk\n"
              "an eagle is a bird -> NONE\n"
              "was ist Beton -> NONE\n"
              "merhaba -> NONE")
    out = runtime.generate(message, system=system, max_tokens=30, temperature=0.0)
    m = re.search(r"CAUSE:\s*(.+?)\s*->\s*EFFECT:\s*(.+)", out, re.I)
    if not m:
        return None
    cause = m.group(1).strip(" .'\"\n")
    effect = m.group(2).strip(" .'\"\n")
    return (cause, effect) if cause and effect else None


def is_causal_question(message):
    """Is it a causal question, and which DIRECTION? 'CAUSES: <topic>' (what
    causes it) / 'EFFECTS: <topic>' (what it leads to) / None. few-shot, the
    language lives in Qwen. Called only if the subject has a causal edge (an ms
    pre-check) — no wasted model call."""
    system = ("Is this asking about the CAUSES or the EFFECTS of something?\n"
              "- what CAUSES X (why X, what X comes from) -> 'CAUSES: X'\n"
              "- what X CAUSES (what X leads to, what happens if X) "
              "-> 'EFFECTS: X'\n- otherwise -> 'NONE'\n"
              # Invented examples, several languages — the direction of a
              # causal question is carried by grammar, and each language
              # carries it differently.
              "what causes rust -> CAUSES: rust\n"
              "¿qué provoca la hambruna? -> CAUSES: hambruna\n"
              "was bewirkt Frost -> EFFECTS: Frost\n"
              "gürültü neye yol açar -> EFFECTS: gürültü\n"
              "what is rust -> NONE\nhola -> NONE")
    out = runtime.generate(message, system=system, max_tokens=20, temperature=0.0)
    m = re.search(r"(CAUSES|EFFECTS):\s*(.+)", out, re.I)
    if not m:
        return None
    subject = m.group(2).strip(" .'\"\n")
    return (m.group(1).lower(), subject) if subject else None


def confirm_cause(cause, effect, message):
    """In the user's language: confirm that the causal fact was learned. The
    fact was written to the graph (with a source) — confirming is safe.
    condition-5: Qwen composes the sentence."""
    system = (_MATCH_LANGUAGE +
              f"The user taught you a CAUSE→EFFECT fact and you stored it: "
              f"'{cause}' causes '{effect}'. Give a short, positive "
              "acknowledgement that you learned this causal relation. One short "
              "sentence. No extra facts.")
    return runtime.generate(message, system=system, max_tokens=50, temperature=0.3)


def is_identity_question(message):
    """Is the message asking about the ASSISTANT'S OWN identity (who are you /
    your name / who made you)? If it asks about something external, NO.
    Language-independent (Qwen). To answer identity questions deterministically
    from the graph instead of a wobbly persona."""
    # Few-shot: a plain instruction could not give Qwen-3B the "is it asking
    # about YOU" concept; with examples it's reliable. Called only on the CHAT
    # branch (ASK "what is X" never reaches here), so the false-positive on
    # "what is" does not affect the flow.
    system = ("Classify if the message asks the responder ABOUT ITSELF — its "
              "identity, name, nature, or who made/created it. A question about "
              "some OTHER thing ('what is X') is NO. Output ONLY yes/no.\n"
              # Several languages on BOTH sides of the boundary, so that
              # "asking about you" is learned as a meaning and not as a set of
              # English pronouns.
              "who are you -> yes\nwho made you -> yes\nwhat is your name -> yes\n"
              "wer hat dich erschaffen -> yes\n¿cómo te llamas? -> yes\n"
              "sen kimsin -> yes\n"
              "what is a dog -> no\nwas ist Beton -> no\n"
              "¿qué es una brújula? -> no\nkartal nedir -> no\n"
              "how are you -> no\nwie geht es dir -> no\ngracias -> no\n"
              "merhaba -> no\nnaber -> no")
    out = runtime.generate(message, system=system, max_tokens=3, temperature=0.0)
    return "yes" in out.strip().lower()


def identity_answer(question, name, id_block):
    """Answers the identity question FROM THE GRAPH. The bridge is built in
    ROUTE (subject=self); here the instruction gives the model its NAME ('name',
    from the graph) and the FACTS about itself — so 'who are you'→name, 'who
    made you'→creator. No need to resolve the 'sen/seni' pronoun. The output is
    from Qwen (no hand-written template, condition-5); it cannot stray from the
    facts."""
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
    """Is the user's message APPROVAL/yes/'go ahead'? (a reply to the research
    offer). Language-independent (Qwen). Returns: True=approval."""
    system = ("Does the user's message mean YES / go ahead / approval, as opposed "
              "to no or a different request? Answer exactly ONE word: YES or NO.")
    out = runtime.generate(message, system=system, max_tokens=3, temperature=0.0)
    return "YES" in out.upper()


def offer_research(subject, message):
    """In the user's language: 'I don't know this, shall I look it up?' (ask-first)."""
    system = (_MATCH_LANGUAGE +
              f"You do NOT have information about '{subject}' in your memory. "
              "Briefly say you don't know it yet and ASK whether you should look "
              "it up. One short sentence, phrased as an offer/question. Invent "
              "no facts.")
    return runtime.generate(message, system=system, max_tokens=40, temperature=0.3)


def refusal(message):
    """In the user's language: 'I don't have this information'. No fabrication, short."""
    system = (_MATCH_LANGUAGE +
              "The user asked about something that is NOT in your memory. Reply "
              "briefly and honestly, saying you don't have that information yet. "
              "Do NOT invent any fact. One short sentence.")
    return runtime.generate(message, system=system, max_tokens=50,
                            temperature=0.3)


def confirm(learned, conflicts, message):
    """In the user's language: briefly confirm the learned fact (+ conflict note).

    `learned`: [(subject, value)] · `conflicts`: [(subject, old, new)]. The
    facts were already written to the graph (supported) — confirming is safe.
    """
    facts = "; ".join(f"{s} = {v}" for s, v in learned)
    system = (_MATCH_LANGUAGE +
              f"The user taught you a new fact and you have stored it in your "
              f"memory: {facts}. Give a short, positive acknowledgement that you "
              f"learned and remembered it. Do NOT apologize. Do NOT say you "
              f"forgot.")
    if conflicts:
        clash = "; ".join(f"{s}: previously '{o}', now '{n}'"
                          for s, o, n in conflicts)
        system += (f" Note: this conflicts with what you already knew: {clash}. "
                   f"Gently mention the conflict.")
    system += (" One or two short, natural sentences, in the user's language "
               "— never mix languages. Do NOT add any other facts.")
    return runtime.generate(message, system=system, max_tokens=90,
                            temperature=0.3)
