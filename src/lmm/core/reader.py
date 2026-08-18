"""The reader: turns a sentence into an operation memory can UNDERSTAND — trained, rule-free.

In the old system reading was split into three pieces: first classify the
sentence, then find the concept, then ask the graph. The three were separate
organs, making separate mistakes, and the class label in the middle was a
way station serving no purpose.

Here there is a single step, and it is the same work a language model does:
text goes in, STRUCTURE comes out. The difference is that the emerging
structure is not buried in the weights but is an operation to be written into
memory.

    "kartal bir kuştur"   ->  WRITE(kartal, tür, kuş)
    "kartal nedir"        ->  ASK(kartal, ?, ?)
    "selam"               ->  PASS

Three operations, three numbers. Not language — their counterparts in the
architecture:

    PASS   carries no knowledge — an utterance keeping the conversation going
    ASK    something is being requested from memory
    WRITE  something is being put into memory (provided it passes the gate)

In this file the network is NOT TRAINED, only LOADED and run. Training is a
separate workbench's job; the only responsibility here is wiring the trained
network to memory's language. If there is no network, `ready` is False and
the system stays silent — a convenience, not a dependency: `v3/` must not
depend on torch.

Nothing belonging to language exists: no suffix, no pattern, no word list.
The network works at the letter level and finds its alphabet by counting it
from its own data.
"""
import os

PASS, ASK, WRITE = 0, 1, 2


class Operation:
    """The reader's output: a single operation to apply to memory.

    `subject`, `predicate`, `value` are text pieces — not identities yet.
    Turning them into identities is geometry's job (the same spelling can be
    more than one concept, and context says which one it is).
    """

    __slots__ = ("kind", "subject", "predicate", "value", "confidence")

    def __init__(self, kind, subject=None, predicate=None, value=None,
                 confidence=0.0):
        self.kind = kind
        self.subject = subject
        self.predicate = predicate
        self.value = value
        self.confidence = confidence

    def __repr__(self):
        return (f"Operation({self.kind}, {self.subject!r}, "
                f"{self.predicate!r}, {self.value!r}, {self.confidence:.2f})")


class Reader:
    """The trained reader. If there is no model, `ready` is False."""

    def __init__(self, folder="models/v3", name="reader.pt"):
        self.ready = False
        self.model = None
        # The path is relative to the repo root: when it was relative to the
        # cwd, a session opened outside the root SILENTLY failed to find the
        # network (round-3 observation) — errorless but blind.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, folder, name)
        if not os.path.exists(path):
            return
        try:
            self._load(path)
        except Exception:                                   # noqa: BLE001
            self.ready = False

    def _load(self, path):
        import torch
        from torch import nn

        held = torch.load(path, map_location="cpu", weights_only=False)
        self.torch = torch
        self.letters = held["letters"]
        width = self.width = held.get("width", 256)
        size = held.get("size", 256)
        layers = held.get("layers", 6)
        heads = held.get("heads", 8)

        class Net(nn.Module):
            """Letter sequence -> (operation kind, role per letter).

            Two heads share one body: one says what the sentence wants to do,
            the other which letters are subject/predicate/value. One body,
            because the two are two faces of the same reading.

            The position table's length is read FROM THE CHECKPOINT, not
            assumed here. In the first draft it was fixed at 4096 and
            training had saved with 257; the model could not load and the
            reader silently stayed not-ready. Writing one number in two
            places is, sooner or later, writing two different numbers.
            """

            def __init__(self):
                super().__init__()
                self.token = nn.Embedding(len(held["letters"]) + 1, size)
                self.place = nn.Embedding(width + 1, size)
                block = nn.TransformerEncoderLayer(
                    size, heads, size * 4, batch_first=True,
                    norm_first=True, dropout=0.1)
                self.body = nn.TransformerEncoder(block, layers)
                self.final = nn.LayerNorm(size)
                self.kind_head = nn.Linear(size, 3)
                self.role_head = nn.Linear(size, 4)

            def forward(self, ids):
                steps = torch.arange(ids.shape[1],
                                     device=ids.device).unsqueeze(0)
                x = self.token(ids) + self.place(steps)
                x = self.final(self.body(x))
                return self.kind_head(x.mean(dim=1)), self.role_head(x)

        self.model = Net()
        self.model.load_state_dict(held["model"])
        self.model.eval()
        self.score = held.get("score", 0.0)
        self.ready = True

    def read(self, sentence):
        """Sentence -> Operation. Without a network returns PASS (silence beats confabulation)."""
        if not self.ready or not sentence:
            return Operation(PASS)
        torch = self.torch
        # FOLD: upper/lower case must NOT CHANGE the operation kind.
        # Measured — while "kalp nedir" read as ASK, "Kalp nedir" was mistaken
        # for WRITE and the garbage "kalp→nedir" was written into the graph;
        # since the facts in training always began with a capital, the
        # network had taken capital-initial as a "fact" marker. Folding
        # preserves length (indexes stay valid) and is language-independent;
        # since training also saw lowercase, it does not fall out of
        # distribution.
        from lmm.core.dataset import fold
        sentence = fold(sentence)
        ids = torch.tensor([[self.letters.get(ch, 0)
                             for ch in sentence[:self.width]]])
        with torch.no_grad():
            kind_out, role_out = self.model(ids)
            kind = int(kind_out.argmax())
            confidence = float(torch.softmax(kind_out, dim=-1)[0].max())
            roles = role_out.argmax(-1)[0].tolist()
        subject, predicate, value = spans(sentence, roles)
        return Operation(kind, subject, predicate, value, confidence)


def spans(text, roles):
    """Reads the three pieces back from the letter roles.

    Roles: 0 outside · 1 subject · 2 predicate · 3 value. For each role the
    longest unbroken run is taken and widened to the word boundary — so the
    piece is still found if the network misses a letter in the middle. The
    word boundary is not a LANGUAGE RULE: knowing that whitespace separates
    words does not require knowing Turkish.
    """
    found = {}
    for role in (1, 2, 3):
        best, run, start = (0, 0), 0, None
        for at, one in enumerate(list(roles[:len(text)]) + [0]):
            if one == role:
                if start is None:
                    start = at
                run += 1
            else:
                if start is not None and run > best[1] - best[0]:
                    best = (start, at)
                start, run = None, 0
        if not best[1]:
            found[role] = None
            continue
        low, high = best
        while low > 0 and not text[low - 1].isspace():
            low -= 1
        while high < len(text) and not text[high].isspace():
            high += 1
        piece = text[low:high]
        # Instead of a hand-written punctuation set, Unicode itself: from the
        # front and back, everything that is not a letter/digit is trimmed.
        # Writing a set would have smuggled an ASCII assumption into a
        # pipeline that had counted its own alphabet.
        start, stop = 0, len(piece)
        while start < stop and not piece[start].isalnum():
            start += 1
        while stop > start and not piece[stop - 1].isalnum():
            stop -= 1
        piece = piece[start:stop]
        from lmm.core.dataset import fold
        found[role] = fold(piece) if piece else None
    return found[1], found[2], found[3]
