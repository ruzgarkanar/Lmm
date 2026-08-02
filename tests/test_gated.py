"""Kapı testi: çekirdek, belleğin onaylamadığı bir şeyi söyleyebilir mi?

Bu testler torch ister ve torch yoksa atlanır. `lmm/` paketinin kendisi hiçbir
zaman torch'a bağımlı değildir; çekirdek isteğe bağlı bir organdır ve testleri de
öyle davranmalıdır.

Sınanan şey modelin kalitesi değil — model burada eğitilmemiştir ve rastgele
konuşur. Sınanan şey, rastgele konuşurken bile yasak bölgeye giremediğidir. Bir
eğilim ölçmüyoruz; erişilemezlik ölçüyoruz.
"""
import unittest

try:
    import torch
    from core.model import Core, Config
    from core.gated import GatedVoice, GLUE, content_of
    HAVE_TORCH = True
except ImportError:                                     # pragma: no cover
    HAVE_TORCH = False


class Pieces:
    """Küçük, öngörülebilir bir parçalayıcı — gerçeğinin yerine geçer.

    Gerçek bir sentencepiece modeli eğitmek bu testin sorusuyla ilgisiz; sorulan
    şey kısıtlamanın tutup tutmadığı, parçalamanın iyi olup olmadığı değil.
    """

    WORDS = ["<pad>", "<unk>", "<s>", "</s>", "penguen", "kartal", "uçar",
             "uçamaz", "kuş", "bir", "ve", "çünkü", "değil", ".", "a", "b"]

    def __init__(self):
        self.index = {w: i for i, w in enumerate(self.WORDS)}

    def get_piece_size(self):
        return len(self.WORDS)

    def id_to_piece(self, identifier):
        return self.WORDS[identifier]

    def eos_id(self):
        return 3

    def bos_id(self):
        return 2

    def encode(self, text):
        return [self.index[w] for w in text.split() if w in self.index]

    def decode(self, identifiers):
        return " ".join(self.WORDS[i] for i in identifiers
                        if i not in (0, 2, 3))


@unittest.skipUnless(HAVE_TORCH, "torch yok — çekirdek isteğe bağlı")
class TestTheGateHolds(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        pieces = Pieces()
        core = Core(Config(vocabulary=pieces.get_piece_size(), dimensions=32,
                           layers=2, heads=2, context=32))
        self.voice = GatedVoice(core, pieces)
        self.pieces = pieces

    def test_an_unapproved_word_is_unreachable_not_merely_unlikely(self):
        """Asıl iddia. 'uçar' izinli değilse üretilemez — bir kez bile."""
        for _ in range(25):
            said = self.voice.say(["penguen", "uçamaz"], opening="penguen",
                                  length=12, temperature=2.0)
            self.assertNotIn("uçar", said.split())
            self.assertNotIn("kartal", said.split())

    def test_what_was_approved_can_still_be_said(self):
        """Kapı yalnızca kapatmamalı; onaylananın yolu açık kalmalı."""
        allowed = self.voice.permit(["penguen", "uçamaz"])
        self.assertIn(self.pieces.index["penguen"], allowed)
        self.assertIn(self.pieces.index["uçamaz"], allowed)
        self.assertNotIn(self.pieces.index["uçar"], allowed)

    def test_glue_is_always_open_or_no_sentence_could_be_built(self):
        allowed = self.voice.permit(["penguen"])
        for word in ("bir", "ve", "çünkü"):
            self.assertIn(self.pieces.index[word], allowed,
                          f"bağlayıcı kapalı: {word}")

    def test_approving_nothing_still_cannot_produce_content(self):
        """Bellek sessizse çekirdek de sessiz kalmalı — boşluğu doldurmamalı."""
        for _ in range(15):
            said = self.voice.say([], length=10, temperature=2.0)
            for forbidden in ("penguen", "kartal", "uçar", "uçamaz", "kuş"):
                self.assertNotIn(forbidden, said.split())

    def test_widening_the_approval_widens_what_can_be_said(self):
        narrow = set(self.voice.permit(["penguen"]))
        wide = set(self.voice.permit(["penguen", "kartal"]))
        self.assertTrue(narrow < wide)
        self.assertIn(self.pieces.index["kartal"], wide - narrow)


@unittest.skipUnless(HAVE_TORCH, "torch yok — çekirdek isteğe bağlı")
class TestReadingTheChain(unittest.TestCase):
    def test_content_comes_from_the_reasoning_chain_not_from_the_model(self):
        chain = ["penguen bir kuş", "kuş uçar (kaynak: hayvanlar.txt)"]
        content = content_of(chain)
        self.assertIn("penguen", content)
        self.assertIn("kuş", content)
        self.assertNotIn("kartal", content)

    def test_the_same_word_is_not_listed_twice(self):
        self.assertEqual(content_of(["kuş kuş kuş"]), ["kuş"])


if __name__ == "__main__":
    unittest.main()
