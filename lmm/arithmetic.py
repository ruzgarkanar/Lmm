"""Arithmetic: computed, never recalled, never guessed.

A language model predicts the tokens of an answer, which is why it can be
confidently wrong about 17 × 43. Storing the answer as a fact would be no better
— there are infinitely many sums and a memory can hold only the ones someone
wrote down.

So this is a third kind of source, next to what was taught and what was worked
out: a result the system produced itself, with the working available as its
justification. It is deterministic, so it cannot be wrong; where it cannot
compute, it declines, exactly as the gate declines what memory does not hold.

Nothing here evaluates arbitrary text. Tokens are numbers and operators or the
expression is refused.
"""
OPERATORS = {
    "+": ("topla", 1), "-": ("çıkar", 1),
    "*": ("çarp", 2), "×": ("çarp", 2), "x": ("çarp", 2),
    "/": ("böl", 2), "÷": ("böl", 2),
    "^": ("üs", 3),
}

WORDS = {
    "artı": "+", "topla": "+", "eksi": "-", "çıkar": "-",
    "çarpı": "*", "kere": "*", "bölü": "/", "üzeri": "^",
}

ENDINGS = ("kaç", "kaçtır", "eder", "kaçeder", "=", "?")


class Undecidable(Exception):
    """The expression is not one this can settle. Refuse rather than guess."""


def _number(token):
    cleaned = token.replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def tokenize(text):
    """Numbers and operators only; anything else makes the whole thing refuse."""
    raw = text.replace("=", " ").split()
    tokens = []
    for word in raw:
        lowered = word.lower()
        if lowered in ENDINGS:
            continue
        if lowered in WORDS:
            tokens.append(WORDS[lowered])
            continue
        if lowered in OPERATORS:
            tokens.append(lowered)
            continue
        value = _number(word)
        if value is None:
            raise Undecidable(word)
        tokens.append(value)
    return tokens


def looks_like_a_sum(text):
    """Cheap enough to ask before anything else, and never a false positive."""
    try:
        tokens = tokenize(text)
    except Undecidable:
        return False
    return (len(tokens) >= 3
            and any(isinstance(t, str) for t in tokens)
            and isinstance(tokens[0], (int, float)))


def normalise(text):
    """The expression as the system read it, so the answer echoes that and not
    the question's trailing words."""
    return " ".join(str(token) for token in tokenize(text))


def evaluate(text):
    """The value of an expression, by precedence, with no eval anywhere."""
    tokens = tokenize(text)
    if not tokens or not isinstance(tokens[0], (int, float)):
        raise Undecidable(text)
    values, operators = [tokens[0]], []
    index = 1
    while index < len(tokens):
        operator = tokens[index]
        if not isinstance(operator, str) or operator not in OPERATORS:
            raise Undecidable(text)
        if index + 1 >= len(tokens) or isinstance(tokens[index + 1], str):
            raise Undecidable(text)
        while operators and OPERATORS[operators[-1]][1] >= OPERATORS[operator][1]:
            _reduce(values, operators)
        operators.append(operator)
        values.append(tokens[index + 1])
        index += 2
    while operators:
        _reduce(values, operators)
    return values[0]


def _reduce(values, operators):
    operator = operators.pop()
    right, left = values.pop(), values.pop()
    values.append(_apply(operator, left, right))


def _apply(operator, left, right):
    if operator in ("+",):
        return left + right
    if operator == "-":
        return left - right
    if operator in ("*", "×", "x"):
        return left * right
    if operator in ("/", "÷"):
        if right == 0:
            raise Undecidable("sıfıra bölme")
        result = left / right
        return int(result) if float(result).is_integer() else result
    if operator == "^":
        return left ** right
    raise Undecidable(operator)
