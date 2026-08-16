"""The speaker: builds sentences from records — trained, template-free.

In the old system the programmer had written the speaking: "{kavram} bir
{hedef}dır". A language model has no counterpart of that; there the sentence
is produced word by word, by probability, and every sentence is new. This
file is the counterpart of that mechanism here.

    input    [ (kartal, tür, kuş) ‹hayvanlar.txt› , (kartal, özellik, hızlı) ]
    output   "Kartal, hızlı bir kuştur."

The difference is where the input comes from: a language model derives the
sentence from its weights, and what it rests on is uncomputable. Here
generation is CONDITIONED ON RECORDS and record keys stand behind every
sentence.

TWO GATES. Generation is not free:

    FIRST    the network sees only the records handed to it — it has no
             material to fabricate something it does not know
    SECOND   the built sentence is read back (`v3/reader.py`) and the
             resulting operations are compared with the records; a sentence
             that does not hold FALLS

Without the second gate, fluency opens itself to confabulation. In this
architecture there is no trade between fluency and correctness: the unfluent
sentence is discarded, and so is the wrong one.

In this file the network is NOT TRAINED, only loaded. If there is no network,
`ready` is False and the system dumps raw records — it cannot speak, but it
does not lie either.

Nothing belonging to language exists: no template, no suffix, no word list.
"""
import os

# The most candidates produced in one answer. The weighing picks among them;
# producing a single candidate is speaking without thinking.
CANDIDATES = 4

# The length at which generation stops — letters. The network learns the stop
# marker itself; this is only a safeguard against an infinite loop.
LONGEST = 400


class Speaker:
    """The trained speaker. If there is no model, `ready` is False."""

    def __init__(self, folder="models/v3", name="speaker.pt"):
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
        self.inverse = {value: key for key, value in self.letters.items()}
        size = held.get("size", 256)
        layers = held.get("layers", 6)
        heads = held.get("heads", 8)
        # The stop marker comes from the checkpoint and CANNOT be 0: 0 is the
        # padding/unknown-letter identity — if it collides with the stop,
        # generation cuts off at the first unrecognized letter. The training
        # script must reserve the stop as len(letters)+1.
        self.stop = held.get("stop")
        width = held.get("width", 2048)
        self.width = width

        class Net(nn.Module):
            """Record sequence + the letters so far -> the next letter.

            Causal: while generating, the future is not seen. The reader was
            bidirectional because understanding requires seeing the end;
            generating is the opposite.
            """

            def __init__(self):
                super().__init__()
                # +2: padding (0) and the STOP identity (len+1). The notebook
                # trained it this way; a loader building +1 silently fell
                # over with a size mismatch.
                self.token = nn.Embedding(len(held["letters"]) + 2, size)
                self.place = nn.Embedding(width + 1, size)
                block = nn.TransformerEncoderLayer(
                    size, heads, size * 4, batch_first=True,
                    norm_first=True, dropout=0.1)
                self.body = nn.TransformerEncoder(block, layers)
                self.final = nn.LayerNorm(size)
                self.head = nn.Linear(size, len(held["letters"]) + 2)

            def forward(self, ids, mask):
                import torch as _torch
                steps = _torch.arange(ids.shape[1],
                                      device=ids.device).unsqueeze(0)
                x = self.token(ids) + self.place(steps)
                x = self.body(x, mask=mask)
                return self.head(self.final(x))

        self.model = Net()
        self.model.load_state_dict(held["model"])
        self.model.eval()
        self.score = held.get("score", 0.0)
        self.ready = True

    def say(self, prompt, count=CANDIDATES, warmth=0.8):
        """Produces candidate sentences from a record sequence.

        `prompt`: the records' plain-text layout — the only thing the network
        sees.
        Returns: [sentence]. Multiple candidates come from the same input and
        the weighing does the picking; this is the counterpart of the
        "should I say it this way or that" step.
        """
        if not self.ready or not prompt:
            return []
        torch = self.torch
        found = []
        for at in range(count):
            ids = [self.letters.get(ch, 0) for ch in prompt]
            start = len(ids)
            made = []
            with torch.no_grad():
                for step in range(LONGEST):
                    window = torch.tensor([ids[-self.width:]])
                    mask = torch.nn.Transformer.generate_square_subsequent_mask(
                        window.shape[1])
                    logits = self.model(window, mask)[0, -1]
                    # LOOP BLOCK (no-repeat n-gram): if an n-gram starting
                    # with the last n-1 letters has ALREADY occurred in the
                    # generated part, ban the letter that would complete it.
                    # An undertrained network locked up in greedy generation
                    # into "bir kaç tane bir kaç tane"; that is not a model
                    # error but a decoding error — it is broken here.
                    for ban in self._loops(ids, start):
                        logits[ban] = float("-inf")
                    # The first candidate is heated little (stable), let the
                    # later ones diversify.
                    heat = 0.5 if at == 0 else max(warmth, 0.6)
                    probs = torch.softmax(logits / heat, dim=-1)
                    # TOP-P (nucleus sampling): pick from the smallest set
                    # whose probability exceeds a total of p. It skips both
                    # greedy's loop and plain heating's noise tail — fluent
                    # but unstuck generation.
                    pick = self._nucleus(probs, 0.92)
                    if self.stop is not None and pick == self.stop:
                        break
                    ids.append(pick)
                    made.append(self.inverse.get(pick, ""))
            text = "".join(made).strip()
            if text and text not in found:
                found.append(text)
        return found

    def _loops(self, ids, start, size=10):
        """If an n-gram starting with the last (size-1) letters repeats in
        the generated sequence, returns the letters that would complete it —
        an exact loop block. Looks only at the GENERATED part (the
        input/prompt does not count)."""
        made = ids[start:]
        if len(made) < size:
            return set()
        prefix = made[-(size - 1):]
        banned = set()
        for at in range(len(made) - (size - 1)):
            if made[at:at + size - 1] == prefix:
                banned.add(made[at + size - 1])
        return banned

    def _nucleus(self, probs, p):
        """Nucleus (top-p) sampling: starting from the most probable,
        accumulate until the total probability exceeds p, sample from that
        set."""
        torch = self.torch
        order = torch.argsort(probs, descending=True)
        keep, cum = [], 0.0
        for idx in order.tolist():
            keep.append(idx)
            cum += float(probs[idx])
            if cum >= p:
                break
        weights = torch.tensor([float(probs[i]) for i in keep])
        return keep[int(torch.multinomial(weights, 1))]


def prompt_of(records, memory):
    """Turns the records into the plain text the network will see.

    This is NOT A SENTENCE TEMPLATE, it is a data layout: the network's
    input, not something a human will read. The separators are
    language-independent markers; the network decides which word goes where.
    """
    lines = []
    for record in records:
        subject = _label(memory, record.subject)
        predicate = _label(memory, record.predicate)
        value = _label(memory, record.value)
        lines.append(f"{subject}\t{predicate}\t{value}")
    return "\n".join(lines) + "\n\n"


def _label(memory, key):
    """From an identity key to its first label. If not an identity, the value itself."""
    # A None predicate must be EMPTY, not "None": garbage like
    # "kalp\tNone\t..." was entering the network's input and "none" was
    # leaking into the output (measured).
    if key is None:
        return ""
    held = memory.identities.get(key) if isinstance(key, int) else None
    if held is not None and held.labels:
        return held.labels[0]
    return str(key)
