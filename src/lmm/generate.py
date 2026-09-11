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


def _voiced(system, persona):
    """The persona rides in front of a PHRASING prompt — and only there.

    In a prompt-only system the system prompt is tone and safety at once, so
    exposing it exposes everything. Here safety is code: the gates read the
    OUTPUT, never the prompt, so an operator's persona can colour the voice
    of every spoken turn while being structurally unable to loosen what may
    be spoken. It is prepended, the way _MATCH_LANGUAGE is, so the identity
    reads first and the operational rules keep the last word. The judge
    prompts (support, re-extraction, relation) never pass through here."""
    return f"{persona.strip()}\n\n{system}" if persona and persona.strip() \
        else system


def answer(question, facts_block, warmth=0.2, persona="", max_tokens=None,
           field=""):
    """Answer the question fluently, using only the given facts. If there is no
    fact, the model is steered to say 'I don't know' (system prompt).

    `field` is the head the corpus writes for what the question asked about
    in other words — the bridge's finding. It steers VOCABULARY, never
    content: nothing is added to the facts, and the gates read the facts as
    before. Measured on a hardware corpus: asked which machine carries the
    most RAM, the block held "MEMORY — largest: Falcon Workstation: 64 GB"
    and the writer answered "carries the most RAM", which the jury rightly
    refused because RAM appears nowhere in the evidence. The answer was on
    the table and the user got nothing. Naming the field as the documents
    name it is the difference between an attested sentence and a silence.
    """
    system = prompts.ANSWER_SYSTEM
    if field:
        system += ("\n\nTHE FACTS CALL THIS FIELD \"%s\". The question uses "
                   "another word for it. Answer with the facts' word and the "
                   "facts' value — your word is not attested and the answer "
                   "will be refused." % field)
    return runtime.generate(answer_prompt(question, facts_block),
                            system=_voiced(system, persona),
                            max_tokens=max_tokens or 200, temperature=warmth)


def compose(brief, material_block, warmth=0.2, persona="", max_tokens=None,
            style=""):
    """Draft a long-form document from labelled material — the long-form
    counterpart of `answer`, under the same contract: the engine phrases, it
    does not know. The token budget is the one thing that differs, because a
    draft is not a sentence; the verification that makes the budget safe to
    raise lives in the caller (`Session.compose`), which re-reads the draft
    line by line against this same material."""
    fmt = f"\n\nFORMAT (the operator's sheet \u2014 follow it):\n{style}" \
        if style else ""
    return runtime.generate(
        f"REQUEST:\n{brief}{fmt}\n\nMATERIAL:\n{material_block}",
        system=_voiced(prompts.COMPOSE_SYSTEM, persona),
        max_tokens=max_tokens or 900, temperature=warmth)


def compose_stream(brief, material_block, warmth=0.2, persona="",
                   max_tokens=None, style=""):
    """`compose` as a stream of chunks — same prompt, same contract; the
    caller (Session.compose) judges each completed line as it arrives."""
    fmt = f"\n\nFORMAT (the operator's sheet \u2014 follow it):\n{style}" \
        if style else ""
    yield from runtime.generate_stream(
        f"REQUEST:\n{brief}{fmt}\n\nMATERIAL:\n{material_block}",
        system=_voiced(prompts.COMPOSE_SYSTEM, persona),
        max_tokens=max_tokens or 900, temperature=warmth)


def chat_stream(message, identity_block="", warmth=0.7, history=None,
                persona="", max_tokens=None):
    """`chat` as a stream of chunks — same prompt built the same way (the
    identity addendum and its steadier warmth included), so the streamed
    voice and the blocking voice are one voice. The caller judges each
    completed sentence as its full stop arrives."""
    system = _voiced(prompts.CHAT_SYSTEM, persona)
    if identity_block.strip():
        system += (
            "\n\nFACTS about yourself, as [entity relation value] rows from your "
            "memory:\n" + identity_block +
            "\n\nIf the user asks who/what you are or who made/created you, answer "
            "using these facts, but write a natural sentence IN THE USER'S "
            "LANGUAGE \u2014 never copy the raw rows or the arrow. If a fact is not "
            "listed, do not invent it.")
        warmth = 0.4
    messages = list(history or [])
    messages.append({"role": "user", "content": message})
    yield from runtime.generate_stream(messages, system=system,
                                       max_tokens=max_tokens or 120,
                                       temperature=warmth)


def chat(message, identity_block="", warmth=0.7, history=None, persona="",
         max_tokens=None):
    """Chat reply (greeting, thanks, small talk). If it carries a fact claim,
    verify filters it — so speak naturally, but a fabricated fact still drops at
    the exit.

    `history`: recent turns [{role,content}] — conversational CONTINUITY (what
    we just talked about). So "what are you doing" and follow-up questions are
    answered with context.

    `identity_block`: identity facts (from the graph, e.g. lmm→creator→<operator>).
    It is INJECTED so questions like "who made you" rest on the GRAPH, not on a
    persona — language-independent (an English persona stayed weak in Turkish;
    an injected fact works in every language). Since the fact comes from the
    graph, it passes verify.
    """
    system = _voiced(prompts.CHAT_SYSTEM, persona)
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
                            max_tokens=max_tokens or 120, temperature=warmth)


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
                           max_tokens=4, temperature=0.0, small=True)
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
                           temperature=0.0, small=True)
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
        system=prompts.RELATION_SYSTEM, max_tokens=4, temperature=0.0, small=True)
    return out.strip().lower().startswith("yes")


# `hedge_note` STOOD HERE, AND IS GONE. It asked the engine for a short caveat
# and `Session` concatenated the result onto an answer that had already passed
# verify / digits_ok / coverage / read-back — so the one property this
# architecture rests on ("what is not in the evidence does not get said") was
# enforced, and then the string it was enforced on was mutated. Measured on
# twelve public documents: four of six wrong answers were this note, e.g.
# "I do not know." followed by an invented HTTP status code, in a language
# nobody had asked for. The mark that replaced it is built from the stored
# source stamp with no model call at all (see `session.UNCERTAIN`).


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
    out = runtime.generate(message, system=system, max_tokens=30, temperature=0.0, small=True)
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
    out = runtime.generate(message, system=system, max_tokens=3, temperature=0.0, small=True)
    return "yes" in out.strip().lower()


def phrasings(message, sample=()):
    """How might the thing this question asks about be WRITTEN in a
    document? Words and short phrases, nothing else.

    THE SECOND ASK. Retrieval here is lexical, which is what makes it
    auditable — a passage arrives because a word arrived, and both can be
    shown. The cost is paraphrase: a novel says a character "oturur" in a
    house and the reader asks where she "yaşıyor", and the memory
    abstains with the sentence in its hands. The frameworks that do not
    have this problem embed their text in a vector space where meaning is
    geometry; they also cannot say why a passage was chosen.

    So the engine is asked for the WORDS, not for the answer, and only
    after the first attempt found nothing. What comes back is filtered
    against the store's own index before it can widen a search, so a word
    nobody wrote cannot enter — the same rule the field bridge keeps.
    """
    listing = ("\n\nSome words this collection uses:\n" + ", ".join(sample)
               if sample else "")
    # ROOTS, NOT INFLECTIONS. The store matches a query word to its
    # relatives by a short shared opening, so a fully inflected proposal
    # reaches nothing: measured, the engine offered "oturuyor" for a book
    # that writes "oturur", and four letters of ending put them out of
    # each other's reach. The shortest form of a word reaches all of its
    # forms; asking for that costs nothing and is the same instruction in
    # every language.
    system = ("The user asked a question. Answer with WORDS ONLY: the words "
              "a written document would likely use for what is being asked "
              "about — synonyms and the plainer noun or verb. Give the "
              "SHORTEST, most basic form of each word (a stem, not an "
              "inflected form). Do NOT answer the question. Do NOT invent "
              "facts. At most six items, comma-separated, in the SAME "
              "LANGUAGE as the question."
              # BALANCED ACROSS LANGUAGES, like every other few-shot in
              # this file: the examples teach the SHAPE of the answer
              # (stems, not inflections; words, not facts), and a set
              # leaning on one language teaches that language's habits
              # with it.
              "\n\nwhere does she live -> live, resid, home, lodging, room"
              "\nwie viel kostet das -> Preis, Kost, Gebühr, Betrag"
              "\n¿de qué está hecho? -> material, hecho, composición"
              "\nnerede yaşıyor -> otur, yaşa, ikamet, ev, konut"
              "\ncombien de personnes -> personne, participant, nombre"
              + listing)
    out = runtime.generate(message, system=system, max_tokens=40,
                           temperature=0.0, small=True)
    words = [w.strip(" .;:\"'") for w in re.split(r"[,\n]", out or "")]
    return [w for w in words if w and len(w) > 1][:6]


def field_for(message, heads):
    """Which of the documents' OWN field names does this question ask
    about? One of them, written exactly, or NONE.

    Not a synonym dictionary — a mapping onto vocabulary that provably
    exists in the store. The engine is never asked what a word means in
    general; it is shown the heads the corpus repeats and asked which one
    the question is reaching for. A wrong pick costs a retrieval, never a
    claim: the gates read the evidence, not this."""
    listing = "\n".join(heads)
    system = ("Below are the FIELD NAMES used in a collection of documents. "
              "Decide which single field the question is asking about. Reply "
              "with that field name copied EXACTLY as written, or with NONE "
              "if no field fits. Output nothing else.\n\nFIELDS:\n"
              + listing)
    out = runtime.generate(message, system=system, max_tokens=16,
                           temperature=0.0, small=True)
    return (out or "").strip().strip('".')


def asked_words(head, value=""):
    """The words a reader might use to ask for this FIELD — one call per
    head, paid once at the operator's request, owned by the store.

    The mirror of `field_for`: that one runs at question time and maps a
    question onto the store's vocabulary; this one runs at learning time
    and maps the store's vocabulary onto the reader's, so that at question
    time no engine is needed at all. The example value is shown because a
    head is often an abbreviation the value disambiguates. The reply is
    WORDS, not sentences — nothing produced here can ever be spoken, it
    can only nominate a head for the record path, where every existing
    rule (one head only, one source only, the gates) still stands."""
    shown = ("FIELD NAME: %s\nEXAMPLE VALUE: %s" % (head, value) if value
             else "FIELD NAME: %s" % head)
    system = ("A document stores a field. List the words a reader would "
              "likely use when ASKING for this field, in the language(s) "
              "of the field name itself. Include plain everyday words "
              "(units, question words a reader would pair with it). Output "
              "ONLY the words, comma-separated, no sentences.")
    out = runtime.generate(shown, system=system, max_tokens=60,
                           temperature=0.0, small=True)
    return [w.strip() for w in (out or "").replace("\n", ",").split(",")
            if w.strip()]


def items_of(question, block):
    """The distinct items in this evidence that answer the question — a
    LIST, never a number. The counting organ's one engine call
    (`Session._count_answer`): the engine reads the seated lines and
    names the items; the STORE then verifies every name against the
    block, and the count is the length of what survives. A hallucinated
    item fails verification and does not count, which is why the number
    is never asked for directly — a model asked "how many" answers from
    plausibility, a model asked "which ones" hands over claims the store
    can check one by one."""
    system = ("The EVIDENCE lines below mention zero or more distinct "
              "items of the kind the question asks about. List the NAMES "
              "of those items, exactly as the evidence writes them, "
              "comma-separated. Name nothing the evidence does not "
              "contain. If there are none, output NONE.")
    out = runtime.generate("EVIDENCE:\n%s\n\nQUESTION: %s" % (block, question),
                           system=system, max_tokens=120, temperature=0.0)
    out = (out or "").strip()
    if not out or out.upper().startswith("NONE"):
        return []
    return [x.strip() for x in out.replace("\n", ",").split(",") if x.strip()]


def turn_shape(message):
    """What KIND of turn this message asks for — one reading, at the
    door. Replaces three separate yes/no classifiers (material, count,
    order) asked one after another in the rescue seats: the shape is
    read ONCE, before any chain runs, so a count-shaped question can go
    to the counting organ the way a record-shaped one goes to the
    record path — never gambled on the factual chain first (W88).

    Returns "material", "count", "order" or "none". Few-shot in four
    languages on both sides of each boundary — calibration, not rules;
    no word of any language is matched in code. Small-road eligible."""
    system = ("Classify the message into EXACTLY one word:\n"
              "material - asks to PRODUCE or RECOMMEND content now "
              "(a plan, programme, draft, catalogue, proposal; states a "
              "need and asks what fits it)\n"
              "count - asks HOW MANY of something\n"
              "sum - asks HOW MUCH IN TOTAL of an amount (money, hours, "
              "distance) accumulated over time\n"
              "order - asks which of two things came FIRST or LATER in "
              "time\n"
              "when - asks WHEN something happened, or the first/last "
              "time it did\n"
              "none - anything else (facts, durations, greetings, "
              "context, thanks, brakes)\n"
              "give me your best three-hour plan -> material\n"
              "yeni terfi edenler i\u00e7in ne verelim -> material\n"
              "was empfiehlst du f\u00fcr unser Team -> material\n"
              "how many projects am I leading -> count\n"
              "ka\u00e7 tane liderlik e\u011fitiminiz var -> count\n"
              "\u00bfcu\u00e1ntos restaurantes he probado -> count\n"
              "how much money have I spent on bike gear in total -> sum\n"
              "how many days did my breaks take in total -> sum\n"
              "how many weeks did it take to finish the series -> sum\n"
              "toplam ka\u00e7 saat yol gittim -> sum\n"
              "wie viel habe ich insgesamt ausgegeben -> sum\n"
              "which did I attend first, the workshop or the webinar -> order\n"
              "hangisine \u00f6nce kat\u0131ld\u0131m -> order\n"
              "when did I last go hiking -> when\n"
              "en son ne zaman y\u00fcr\u00fcy\u00fc\u015fe \u00e7\u0131kt\u0131m -> when\n"
              "wann war ich zuletzt wandern -> when\n"
              "what is the duration of the Alpha module -> none\n"
              "Empatik Liderlik ka\u00e7 g\u00fcn -> none\n"
              "wie lange dauert das Training -> none\n"
              "dur biraz, hemen \u00f6nerme -> none\n"
              "we are in banking, my team is ten people -> none\n"
              "thanks, that helps -> none")
    out = runtime.generate(message, system=system, max_tokens=4,
                           temperature=0.0, small=True)
    word = (out or "").strip().lower()
    for shape in ("material", "count", "order", "sum", "when"):
        if shape in word:
            return shape
    return "none"


def event_date(phrase, rows, question=""):
    """WHEN did this event happen — read off the candidate lines.

    `rows`: [(stamp "YYYY/MM/DD", line text), ...]. People tell events
    days later; the date may be written in the LINE ("back on March 22
    the GPS died") or the line's stamp may be the event's own day. The
    engine reads — the one thing it does reliably with the sentence in
    front of it — and answers one date; the CALLER's arithmetic decides
    whether the reading may stand (`Session._event_anchor`): a proposal
    is a proposal here like everywhere else."""
    shown = "\n".join("[stamped %s] %s" % (stamp, text[:300])
                      for stamp, text in rows[:8])
    system = ("Each line below is a chat message with the date it was "
              "SENT. When did the event in question actually happen? If "
              "a line's text states the date (\"on March 22\", \"last "
              "Tuesday\" resolved against its stamp), use that; "
              "otherwise use the stamp of the line that reports it. "
              "Output ONLY the date as YYYY/MM/DD.")
    # THE QUESTION RIDES ALONG: a recurring event ("the Sunday mass")
    # has many true dates, and WHICH one is meant lives in the question
    # — measured, the reading without it picked the wrong week and the
    # arithmetic could not object, because that week was written too.
    asked = ("\n\nQUESTION THIS SERVES: %s" % question) if question else ""
    out = runtime.generate("LINES:\n%s\n\nEVENT: %s%s"
                           % (shown, phrase, asked),
                           system=system, max_tokens=12, temperature=0.0)
    out = (out or "").strip()
    match = re.search(r"(\d{4})\D(\d{1,2})\D(\d{1,2})", out)
    return "%04d/%02d/%02d" % tuple(int(g) for g in match.groups()) \
        if match else ""


PLAN_OPS = (
    "anchor: PHRASE -> the earliest dated line carrying the phrase\n"
    "latest: PHRASE -> the latest dated line carrying the phrase\n"
    "lines: PHRASE -> every dated line carrying the phrase\n"
    "span: A, B -> days between two anchors\n"
    "before: A, B -> whether anchor A predates anchor B\n"
    "before_lines: L, A -> only the lines dated before anchor A\n"
    "after_lines: L, A -> only the lines dated after anchor A\n"
    "count: L -> how many lines\n"
    "month_tally: L -> the month with the most lines")


def plan_of(question):
    """A PLAN over the verified primitives — the composer's one call.

    The end of the hand-written organ queue (W94): instead of one organ
    per question shape, the engine proposes how to COMPOSE the shapes
    it sees from the store's own primitives. Each step is
    `name = op: args`; the interpreter executes only known operations
    on store-anchored phrases, and the spoken sentence is built from
    the final step's typed value by our template — the engine
    contributes operation names and phrases, never an output word. A
    plan that misreads the question costs a retrieval, never a claim.
    """
    system = ("Decompose the question into steps over ONLY these "
              "operations:\n" + PLAN_OPS + "\n"
              "One step per line, exactly `name = op: arg` (args comma-"
              "separated; PHRASE args copied from the question's own "
              "words; the final step must be named out). If the "
              "question does not fit these operations, output NONE.\n"
              "Example:\n"
              "Q: did I adopt the cat before I moved house?\n"
              "a = anchor: adopt the cat\n"
              "b = anchor: moved house\n"
              "out = before: a, b\n"
              "Example:\n"
              "Q: how many weeks passed between the recital and the gala?\n"
              "a = anchor: the recital\n"
              "b = anchor: the gala\n"
              "out = span: a, b\n"
              "Example:\n"
              "Q: how many rehearsals did we hold before the premiere?\n"
              "l = lines: rehearsals\n"
              "a = anchor: the premiere\n"
              "f = before_lines: l, a\n"
              "out = count: f\n"
              "Example:\n"
              "Q: when did I last water the orchids?\n"
              "out = latest: water the orchids\n"
              "Example:\n"
              "Q: when was the first shipment?\n"
              "out = anchor: the shipment")
    raw = runtime.generate("Q: %s" % question, system=system,
                           max_tokens=120, temperature=0.0)
    raw = (raw or "").strip()
    if not raw or raw.upper().startswith("NONE"):
        return []
    steps = []
    for line in raw.splitlines():
        if "=" not in line:
            continue
        name, _eq, rest = line.partition("=")
        op, _colon, args = rest.partition(":")
        parts = [a.strip() for a in args.split(",") if a.strip()]
        steps.append(tuple([name.strip(), op.strip()] + parts))
    return steps[:6]


def things_of(question):
    """The two things a comparison/order question weighs — phrases, not
    an answer. The ordering organ's one call (`Session._order_answer`):
    the engine reads the question and names what is being compared; the
    STORE then anchors each phrase to a dated line or refuses, so a
    misreading costs a retrieval, never a claim."""
    # Measured miss: "how many days passed between A and B" was read
    # as not-a-comparison and answered NONE, so the span opener never
    # engaged. The reading is about REFERRING to two things — compared,
    # ordered, or spanned between — not about comparison grammar.
    system = ("The question refers to TWO distinct things or events — "
              "compared, ordered in time, or with a span between them. "
              "Output those two, one per line, copied as closely as "
              "possible from the question's own words. Nothing else. "
              "If the question does not involve two distinct things, "
              "output NONE.\n"
              "which came first, the audit or the launch ->\n"
              "the audit\nthe launch\n"
              "how many days passed between the marathon and the gala ->\n"
              "the marathon\nthe gala\n"
              "iki etkinlik aras\u0131nda ka\u00e7 g\u00fcn ge\u00e7ti, "
              "konser ile sergi ->\nkonser\nsergi\n"
              "how many books did I read -> NONE\n"
              "what is the duration of the module -> NONE")
    out = runtime.generate(question, system=system, max_tokens=48,
                           temperature=0.0, small=True)
    out = (out or "").strip()
    if not out or out.upper().startswith("NONE"):
        return []
    things = [x.strip(" -•\t") for x in out.splitlines() if x.strip()]
    return things[:2]


def wants_order(message):
    """Does this message ask WHICH CAME FIRST / the order of events?
    Language-independent, few-shot both sides; called only on abstained
    turns, after the count question — one tiny call where the turn
    would otherwise end in a shrug."""
    system = ("Classify if the message asks about the ORDER of events in "
              "time — which happened first, earlier, later, before or "
              "after another. Output ONLY yes/no.\n"
              "which did I attend first, the workshop or the webinar -> yes\n"
              "hangisine \u00f6nce kat\u0131ld\u0131m, atölyeye mi seminere mi -> yes\n"
              "was kam zuerst, das Seminar oder der Kurs -> yes\n"
              "did the launch happen before the audit -> yes\n"
              "how many workshops did I attend -> no\n"
              "what is the duration of the workshop -> no\n"
              "when is the next session -> no\n"
              "thanks, that helps -> no")
    out = runtime.generate(message, system=system, max_tokens=3,
                           temperature=0.0, small=True)
    return "yes" in (out or "").strip().lower()


def amounts_of(question, block):
    """Item and amount pairs the evidence states for this question — a
    LIST of pairs, never a total. The summing organ's one engine call
    (`Session._sum_answer`), under the counting organ's law: the engine
    reads and names, the store verifies each amount beside its item,
    and the TOTAL is arithmetic over what survives — asked for a sum a
    model estimates, asked for the addends it hands over claims that
    can be checked one by one."""
    system = ("The EVIDENCE lines mention zero or more amounts of the "
              "kind the question asks about. List each as "
              "ITEM :: NUMBER, one per line, the number copied exactly "
              "as the evidence writes it, no currency signs, no totals. "
              "Name nothing the evidence does not contain. If none, "
              "output NONE.")
    out = runtime.generate("EVIDENCE:\n%s\n\nQUESTION: %s" % (block, question),
                           system=system, max_tokens=200, temperature=0.0)
    out = (out or "").strip()
    if not out or out.upper().startswith("NONE"):
        return []
    pairs = []
    for line in out.splitlines():
        if "::" in line:
            item, _sep, amount = line.partition("::")
            if item.strip() and amount.strip():
                pairs.append((item.strip(), amount.strip()))
    return pairs


def wants_count(message):
    """Does this message ask HOW MANY of something — a count over what
    the memory holds? Language-independent, few-shot on both sides,
    called only on abstained turns (the rescue seat): one tiny call
    exactly where the turn would otherwise end in a shrug."""
    system = ("Classify if the message asks for a COUNT — how many of "
              "something. Output ONLY yes/no.\n"
              "how many projects am I leading -> yes\n"
              "ka\u00e7 e\u011fitim ald\u0131m bug\u00fcne kadar -> yes\n"
              "wie viele Kurse haben wir gebucht -> yes\n"
              "\u00bfcu\u00e1ntos restaurantes he probado -> yes\n"
              "what is the duration of the Alpha module -> no\n"
              "hangi e\u011fitimler yar\u0131m g\u00fcn s\u00fcr\u00fcyor -> no\n"
              "list my projects -> no\n"
              "thanks a lot -> no")
    out = runtime.generate(message, system=system, max_tokens=3,
                           temperature=0.0, small=True)
    return "yes" in (out or "").strip().lower()


def wants_material(message):
    """Is the message asking the responder to PRODUCE a deliverable NOW — a
    plan, programme, draft, catalogue, recommendation — rather than sharing
    context, asking about a fact, or making small talk? Language-independent,
    few-shot on both sides of the boundary (the is_identity_question
    pattern). Called only on abstained consultation turns — one tiny call,
    exactly where the conversation would otherwise end in a shrug."""
    system = ("Classify if the message asks the responder to PRODUCE or "
              "DELIVER content now (a plan, programme, draft, catalogue, "
              "proposal, recommendation). Sharing context, factual "
              "questions, greetings and thanks are NO. Output ONLY yes/no.\n"
              "give me your best three-hour plan -> yes\n"
              "put together a programme for my team -> yes\n"
              "you decide everything, I don't know -> yes\n"
              "stell mir bitte einen Katalog zusammen -> yes\n"
              "prepara una propuesta para nosotros -> yes\n"
              "propose-moi une formation adapt\u00e9e -> yes\n"
              # A NEED WITH "WHAT SHOULD WE...?" IS AN ORDER TO PRODUCE,
              # measured live: a team's situation plus "what do we give
              # them?" was read as a factual question, so the delivery
              # seat never opened and the factual chain answered with
              # whichever row shared the question's words. Stating a need
              # and asking what fits it IS asking for a recommendation.
              "my new hires struggle with clients, what should we give them -> yes\n"
              "yeni terfi edenler i\u00e7in ne verelim -> yes\n"
              "was empfiehlst du f\u00fcr unser Vertriebsteam -> yes\n"
              "\u00bfqu\u00e9 nos recomiendas para los gerentes nuevos -> yes\n"
              # the boundary, measured live: an OFFER OF HELP and a BRAKE
              # both wear request grammar and are not orders to produce —
              # "can you help us" opened a catalogue, and "wait, you
              # suggested too fast" opened ANOTHER one.
              "we want to run trainings, can you help us -> no\n"
              "kannst du uns dabei helfen -> no\n"
              "wait, hold on \u2014 I did not ask for anything yet -> no\n"
              "dur biraz, hemen önerme -> no\n"
              "un momento, espera un poco \u2014 a\u00fan no ped\u00ed nada -> no\n"
              "we are in banking, my team is ten people -> no\n"
              "what is the duration of the Alpha module -> no\n"
              "wie lange dauert das Training -> no\n"
              "risk is our main focus -> no\n"
              "thanks, that helps -> no\nmerhaba -> no\nhola -> no")
    out = runtime.generate(message, system=system, max_tokens=3,
                           temperature=0.0, small=True)
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
              "don't know. Never invent, never switch language.\n\n"
              # A MAKER NOBODY LISTED IS A FABRICATION, and it costs the
              # whole sentence: told only its name, the engine wrote "I am
              # X, built by OpenAI", the gate rightly dropped the claim,
              # and the name went down with it — the operator's own
              # declaration lost to an invented one. If the rows name no
              # maker, the answer is the name alone.
              "If the facts above do not name who made you, DO NOT NAME "
              "ANYONE — not a company, not a person, not a model. Say who "
              "you are and stop there.")
    # IDENTITY IS A FACT, NOT A PLACE FOR SAMPLING. At 0.2 the same
    # question answered "I am Vale Coach" once and something the gate had to
    # drop the next time — and a dropped identity falls through to the
    # chat voice, which says something pleasant and nameless. The rows are
    # fixed; the sentence should be too.
    return runtime.generate(question, system=system, max_tokens=60,
                            temperature=0.0, small=True)


def is_affirmative(message):
    """Is the user's message APPROVAL/yes/'go ahead'? (a reply to the research
    offer). Language-independent (Qwen). Returns: True=approval."""
    system = ("Does the user's message mean YES / go ahead / approval, as opposed "
              "to no or a different request? Answer exactly ONE word: YES or NO.")
    out = runtime.generate(message, system=system, max_tokens=3, temperature=0.0, small=True)
    return "YES" in out.upper()


def language_of(text):
    """The language a sentence is written in, named in English, one word.

    THE JUDGE THAT COMPARED TWO SENTENCES DID NOT WORK. It was asked
    "is the second in the same language as the first?" with examples on
    both sides, and measured against the live engine it answered "no" for
    an English question answered in English — the verdict carried no
    information at all, so every refusal was rewritten once and then kept
    whatever came back. Naming ONE sentence's language is a smaller
    question and the engine answers it reliably; two names can then be
    compared here, where the comparison is arithmetic rather than
    judgement.
    """
    system = ("Name the language this sentence is written in. Answer with "
              "the English name of the language, ONE word, nothing else.\n"
              "What is the price? -> English\n"
              "Fiyat nedir? -> Turkish\n"
              "Ik heb die informatie niet. -> Dutch\n"
              "Quel est le prix ? -> French\n"
              "Wie hoch ist der Preis? -> German")
    # THE TEXT IS QUOTED MATERIAL, NOT A MESSAGE TO OBEY (W89's root).
    # Handed bare, a question-shaped sentence was ANSWERED — "I'm
    # sorry, but" — and the token cap clipped that to "Im", which the
    # caller then compared as a language name. Between markers the
    # sentence is data, and the classifier names its language reliably.
    out = runtime.generate("TEXT: <<%s>> LANGUAGE:" % text,
                           system=system, max_tokens=4, temperature=0.0,
                           small=True)
    return re.sub(r"[^A-Za-z]", "", (out or "").strip().split()[:1][0]
                  if (out or "").strip() else "")


def same_language(question, reply):
    """Are these two written in the same language? Two names, compared."""
    return bool(language_of(question)) and (
        language_of(question).lower() == language_of(reply).lower())


def _spoken_to(message, system, persona="", warmth=0.3, max_tokens=50):
    """Say something in the user's own language — and CHECK that we did.

    A REFUSAL IS SPOKEN IN THE LANGUAGE IT WAS ASKED IN, and this had been
    hiding behind a Turkish corpus: six English questions with no answer in
    the store were refused in Dutch, French and Spanish, never once in
    English. Two structural moves, no phrase in any language written by us.
    First the question is handed back as a LANGUAGE SAMPLE inside the
    instruction — the anchoring the composer already uses for its material —
    and the rule is repeated as the LAST thing read before writing, because
    the end of a prompt is where an instruction survives. That carried five
    of six. The sixth is the reason for the second move: a memory that
    audits every claim it speaks can audit the tongue it speaks them in, so
    the sentence is read back by the same kind of small judge the gates use,
    and on a no it is written once more. A second no is kept — silence is
    worse than a sentence in the wrong language, and nothing here writes a
    canned phrase. The memory speaks in the engine's voice, or not at all.
    """
    # THE LANGUAGE IS NAMED, NOT ONLY SHOWN. The sample anchor alone left
    # a stubborn residue — measured, "Which documents state a MEMORY?" was
    # refused in Dutch about half the time — and the comparing judge that
    # was supposed to catch it did not work at all. So the engine names
    # the language of the question (a small question it answers well),
    # writes in that named language, and the reply's own language is named
    # too: if the two names differ the sentence is written once more, with
    # the name in front of it. Still no canned phrase in any language.
    tongue = ""
    try:
        tongue = language_of(message)
    except Exception:                                        # noqa: BLE001
        tongue = ""
    # THE NAME IS FOR COMPARING, NEVER FOR COMMANDING (W89). Naming the
    # tongue inside the instruction turned one clipped classifier reply
    # ("Im") into an order to write in a language that does not exist,
    # and the retry repeated the order. The sample carries the language;
    # the name only lets two readings be compared by arithmetic below.
    anchored = (system + "\n\nLANGUAGE SAMPLE — the user wrote this, and your "
                "reply must be in the SAME language as this sentence:\n"
                + message.strip() +
                "\n\nWrite one short sentence, in the language of the "
                "sample above.")
    said = runtime.generate(message, system=_voiced(anchored, persona),
                            max_tokens=max_tokens, temperature=warmth)
    if said and tongue:
        try:
            spoken = language_of(said)
        except Exception:                                    # noqa: BLE001
            spoken = tongue
        if spoken and spoken.lower() != tongue.lower():
            # THE RETRY IS THE LAST WORD, SO IT DOES NOT GAMBLE. Kept at
            # the first attempt's warmth, the rewrite was another sample
            # of the same dice — observed live, one refusal came out as
            # letters that were no language at all, and the second roll
            # was what the user read. The retry is the turn's final
            # sentence; it is written at temperature zero, the way every
            # candidate the gates read is.
            # THE RETRY IS SHOWN ITS OWN MISTAKE, AND — WHEN TWO
            # INDEPENDENT READINGS AGREE — THE SAMPLE'S NAME. Measured:
            # the engine writes an English question's refusal in Dutch
            # deterministically, instruction or no instruction; only
            # naming the language moves it. W89's law stands — a bogus
            # name must never command — so the name may enter the
            # instruction only when a SECOND naming, differently
            # phrased, returns the same word: a clipped chat reply does
            # not survive two prompt shapes. One extra small call, paid
            # only on the mismatch turns.
            second = ""
            try:
                second = re.sub(r"[^A-Za-z]", "", (runtime.generate(
                    "<<%s>>" % message,
                    system=("Identify the language of the text between "
                            "the markers. Output only the language name "
                            "in English."),
                    max_tokens=4, temperature=0.0, small=True)
                    or "").strip().split()[0] if True else "")
            except Exception:                                # noqa: BLE001
                second = ""
            named = (("\nThe sample's language is %s. Write strictly in "
                      "%s." % (tongue, tongue))
                     if second and second.lower() == tongue.lower() else "")
            said = runtime.generate(
                message,
                system=_voiced(anchored + "\n\nYour previous reply was:\n"
                               + said + "\nThat is NOT the language of the "
                               "sample. Write one short sentence, strictly "
                               "in the language of the sample above."
                               + named,
                               persona),
                max_tokens=max_tokens, temperature=0.0)
    return said


def offer_research(subject, message):
    """In the user's language: 'I don't know this, shall I look it up?' (ask-first)."""
    return _spoken_to(message, _MATCH_LANGUAGE +
                      f"You do NOT have information about '{subject}' in your "
                      "memory. Briefly say you don't know it yet and ASK whether "
                      "you should look it up. One short sentence, phrased as an "
                      "offer/question. Invent no facts.", max_tokens=40)


def refusal(message, persona="", warmth=0.0, max_tokens=None):
    """In the user's language: 'I don't have this information'. No fabrication, short."""
    return _spoken_to(message, _MATCH_LANGUAGE +
                      "The user asked about something that is NOT in your memory. "
                      "Reply briefly and honestly, saying you don't have that "
                      "information yet. Do NOT invent any fact. One short sentence.",
                      persona=persona, warmth=warmth, max_tokens=max_tokens or 50)


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


# HOW MANY QUESTIONS TO ASK FOR PER LINE. This is a GENERATION BUDGET, not a
# decision threshold — it buys candidates, and which of them survive is decided
# by the document itself (`evidence.SentenceStore.learn_expansions`). It is
# stated once, overridable with `LMM_EXPAND_K`, and nothing downstream depends
# on its value: a larger K costs more completion tokens and hands the same
# filter a longer list.
EXPANSIONS = 3


def expansions(sentence):
    """The questions this line answers, asked in other words — index material
    for the offline expansion (doc2query--).

    Returns a list of strings, at most `EXPANSIONS` of them. NOTHING here is
    speakable: the caller writes it into a separate index and the answer block
    is still built from the document's own sentences. See
    `evidence.SentenceStore.__init__` for the wall and why it is absolute.

    Two structural filters, and neither is a language rule: a line that comes
    back as itself is not an expansion (the engine echoed instead of rewording),
    and an empty output is the prompt's own way of saying this line asks for
    nothing.
    """
    import os
    count = int(os.environ.get("LMM_EXPAND_K", EXPANSIONS))
    if count <= 0:
        return []
    # THE LANGUAGE RULE GOES FIRST, and it is the one this codebase already
    # measured: a language instruction buried inside a list of rules is not
    # enough (see `_MATCH_LANGUAGE`). Measured here too, and expensively — the
    # first full run of this pass expanded an ENGLISH corpus and an English NIST
    # publication entirely in SPANISH, 4,766 units of it, because the rule sat
    # third in a list while three few-shot examples sat under it in three
    # languages. Every generated word was then novel to the document by virtue
    # of being in the wrong language, which is index bloat wearing the shape of
    # a synonym.
    out = runtime.generate(sentence,
                           system=_MATCH_LANGUAGE + prompts.EXPAND_SYSTEM,
                           max_tokens=40 * count, temperature=0.0, small=True)
    seen, kept = {_flat(sentence)}, []
    for line in (out or "").splitlines():
        # Leading list marks are FORMAT the prompt asked not to produce; the
        # engine produces them anyway, and stripping punctuation off the front
        # of a line is not a word list.
        line = line.strip().lstrip("-*•").strip().strip("\"'").strip()
        if not line or line == "->":
            continue
        key = _flat(line)
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)
        if len(kept) >= count:
            break
    return kept


def _flat(text):
    """Case- and punctuation-blind form of a line — only used to notice that
    two generated lines are the same line."""
    return " ".join(re.findall(r"\w+", text.lower()))
