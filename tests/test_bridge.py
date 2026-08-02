"""Köprü: grafın cevabı akıcılaşırken içerik değişmemeli.

Model burada eğitilmemiştir ve saçmalar. Test edilen şey saçmalarken bile
grafın söylemediği bir şeyi söyleyememesi — yani kalitenin değil, sınırın
sınanması.
"""
import unittest

try:
    import torch
    from core.model import Core, Config
    from core.bridge import Bridge
    from tests.test_gated import Pieces
    HAVE_TORCH = True
except ImportError:                                     # pragma: no cover
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch yok — çekirdek isteğe bağlı")
class TestTheBridgeCarriesContentOnly(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        pieces = Pieces()
        core = Core(Config(vocabulary=pieces.get_piece_size(), dimensions=32,
                           layers=2, heads=2, context=32))
        self.bridge = Bridge(core, pieces)

    def test_glue_is_not_treated_as_content(self):
        content = self.bridge.content_words("penguen bir kuştur ve uçamaz")
        self.assertIn("penguen", content)
        self.assertNotIn("bir", content)
        self.assertNotIn("ve", content)

    def test_the_source_note_is_content_too_not_decoration(self):
        """Kaynak da cevabın parçasıdır; kapıdan o da geçmelidir."""
        content = self.bridge.content_words("penguen uçamaz (kaynak: kuslar)")
        self.assertIn("kuslar", content)

    def test_a_word_the_graph_never_said_cannot_appear(self):
        """Asıl iddia, akıcılaştırma katmanında da geçerli mi?"""
        answer = "penguen uçamaz"
        for _ in range(20):
            said = self.bridge.rephrase(answer, temperature=2.0, length=12)
            self.assertNotIn("uçar", said.split())
            self.assertNotIn("kartal", said.split())

    def test_an_empty_answer_is_returned_untouched(self):
        self.assertEqual(self.bridge.rephrase(""), "")

    def test_the_escape_check_finds_nothing_on_a_gated_answer(self):
        answer = "penguen uçamaz"
        said = self.bridge.rephrase(answer, temperature=1.5, length=12)
        self.assertEqual(self.bridge.escaped(said, answer), [])


if __name__ == "__main__":
    unittest.main()
