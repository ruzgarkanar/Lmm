"""Nearness between words — used to suggest, never to answer.

This is where an embedding would go, and the reason there isn't one is worth
stating. An embedding decides that two words are alike by putting them near each
other in a space nobody can inspect. When it is right you get a fluent answer;
when it is wrong you get a fluent wrong answer, and the only repair is to
retrain. LMM states likeness instead: "otomobil bir arabadır" is one line a
person can write, read, and delete.

What an embedding would genuinely buy us is the smaller thing: recognising that
someone typing "otomobil" probably means the "araba" we already know. That help
is worth having, so it is offered as a question rather than taken as an answer —
the system still says it does not know, and asks whether you meant something
else. A guess that has to be confirmed cannot become a false belief.

Character bigrams are enough for that job and need nothing but the standard
library. If a vector model ever replaces the scoring here, the rule above is
what must not change: suggestions may come from a guess, answers may not.
"""

THRESHOLD = 0.45
SUGGESTIONS = 3


def _bigrams(word):
    padded = f" {word} "
    return {padded[i:i + 2] for i in range(len(padded) - 1)}


def distance(first, second, cap=2):
    """Edit distance, given up on once it passes the cap."""
    if abs(len(first) - len(second)) > cap:
        return cap + 1
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, start=1):
        current = [i]
        for j, right in enumerate(second, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (left != right)))
        if min(current) > cap:
            return cap + 1
        previous = current
    return previous[-1]


def closeness(first, second):
    """0..1 — how much two words look alike.

    Bigram overlap carries longer words; a short word needs edit distance, since
    "kuş" and "kus" share almost no bigrams while being one keystroke apart.

    Neither catches synonyms. "araba" and "otomobil" look nothing alike and no
    amount of string comparison will connect them — that is a fact about the
    world, and facts about the world belong in memory: "otomobil bir arabadır".
    """
    if first == second:
        return 1.0
    typo = distance(first, second)
    if typo <= 1 or (typo == 2 and min(len(first), len(second)) >= 6):
        return 0.9
    left, right = _bigrams(first), _bigrams(second)
    shared = left & right
    if not shared:
        return 0.0
    return len(shared) / len(left | right)


def nearest(word, candidates, limit=SUGGESTIONS, threshold=THRESHOLD):
    """The known words a person might have meant, best first."""
    scored = []
    for candidate in candidates:
        if candidate == word:
            continue
        score = closeness(word, candidate)
        if score >= threshold:
            scored.append((score, candidate))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [candidate for _, candidate in scored[:limit]]
