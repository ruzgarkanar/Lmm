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


SHORT = 2               # bu uzunluğa kadar olan adlar indekse güvenilemez


def _postings(candidates):
    """İkiliden adlara indeks. Kurulabildiği yere iliştirilir, bir kez kurulur.

    Aday listesine yazılamıyorsa (düz liste, demet) indeks kurulmuyor:
    tek bir çağrı için indeks kurmak, taramaktan pahalıdır. İndeks ancak
    tekrar tekrar sorulan bir listede kazanç.
    """
    found = getattr(candidates, "near_index", None)
    if found is not None:
        return found
    if not hasattr(candidates, "__dict__"):
        return None         # düz liste, demet: indeksin asılacağı yer yok, ve
        # ömrü tek çağrı olan bir indeks kurmak taramadan pahalı. Bu denetim
        # KURMADAN ÖNCE: sonra bakılsaydı, indekslenemeyen çağrı hem indeksi
        # kurup hem de taramak zorunda kalırdı.
    postings, short = {}, []
    for name in candidates:
        if len(name) <= SHORT:
            short.append(name)
        for gram in _bigrams(name):
            postings.setdefault(gram, []).append(name)
    candidates.near_index = found = (postings, short)
    return found


def _worth_scoring(word, candidates):
    """Puanı sıfırdan büyük OLABİLECEK adlar. Gerisini hesaplamaya gerek yok.

    NEDEN eksiksiz: `closeness` üç yoldan puan verir ve üçü de ortak ikili
    ister. Yazım yakınlığı (uzaklık <= 1) iki harften uzun iki sözcükte en az
    bir ikiliyi hep sağ bırakır — tek bir düzeltme hem baştaki hem sondaki
    ikiliyi birden bozamaz. Uzaklık 2 dalı yalnız altı harften uzun sözcükler
    için açık ve iki düzeltme yedi ikilinin en fazla dördünü götürür. Geriye
    ikili örtüşmesi kalıyor, o da tanımı gereği ortak ikili ister. Tek istisna
    iki harfe kadar olan adlar ("a" ile "b" bir tuş uzaklıkta, ortak ikilileri
    yok); onlar ayrı tutuluyor ve her zaman puanlanıyor.

    Yani indeks dışında kalan her ad tam olarak 0,0 alır. Eşik sıfır ya da
    altındaysa 0,0 da geçerli bir cevaptır — o durumda indeks kullanılmıyor.
    """
    found = _postings(candidates)
    if found is None:
        return candidates
    postings, short = found
    pool = dict.fromkeys(short)
    # Sıralı gezinme: ikililer sıralı, gönderi listeleri aday sırasında.
    # Küme üzerinde gezinmek `PYTHONHASHSEED`e bağlı bir sıra demek olurdu ve
    # bu projede o bir kez aynı soruya iki farklı cevap ürettirdi.
    for gram in sorted(_bigrams(word)):
        for name in postings.get(gram, ()):
            pool[name] = None
    return pool


def nearest(word, candidates, limit=SUGGESTIONS, threshold=THRESHOLD):
    """The known words a person might have meant, best first."""
    if threshold > 0:
        candidates = _worth_scoring(word, candidates)
    scored = []
    for candidate in candidates:
        if candidate == word:
            continue
        score = closeness(word, candidate)
        if score >= threshold:
            scored.append((score, candidate))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [candidate for _, candidate in scored[:limit]]
