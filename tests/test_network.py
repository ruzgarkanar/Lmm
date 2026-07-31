"""The network recognises sentence shapes, not the words that fill them."""
import unittest

from lmm.network import MiniNetwork, training_data, features, CLASSES
from lmm.intuition import tokenize, Intuition, TEACH, UNKNOWN
from lmm.lexicon import Lexicon


class TestFeatures(unittest.TestCase):
    def test_concepts_are_generalised(self):
        """Two sentences of the same shape must look identical to the network."""
        self.assertEqual(features(tokenize("kuşlar uçar")),
                         features(tokenize("robotlar uçar")))

    def test_position_and_length_are_kept(self):
        """Same tags, different arrangement — these must not collide."""
        self.assertNotEqual(features(tokenize("kar beyaz mı")),
                            features(tokenize("penguen bir kuş değildir")))

    def test_a_learned_verb_is_tagged_as_a_verb(self):
        lexicon = Lexicon()
        lexicon.learn_verb("çalışmak", "çalışır", "çalışamaz")
        self.assertEqual(features(tokenize("robotlar çalışır"), lexicon),
                         features(tokenize("kuşlar uçar")))


class TestMiniNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = MiniNetwork.default()

    def test_training_data_covers_every_class(self):
        self.assertEqual({label for _, label in training_data()}, set(CLASSES))

    def test_recognises_learned_shapes(self):
        for sentence, expected in [("kedi bir hayvandır", "TEACH_TYPE"),
                                   ("balık yüzer mi", "ASK_ABILITY"),
                                   ("kimler uçar", "ASK_WHO"),
                                   ("kar beyazdır", "TEACH_PROPERTY"),
                                   ("penguen neden uçamaz", "ASK_WHY")]:
            label, confidence = self.network.predict(tokenize(sentence))
            self.assertEqual(label, expected, sentence)
            self.assertGreater(confidence, 0.6, sentence)

    def test_shape_recognition_survives_an_unknown_subject(self):
        """A brand new concept must not cost confidence — meeting one is free."""
        known = self.network.predict(tokenize("kuşlar uçar"))
        novel = self.network.predict(tokenize("zebralar uçar"))
        self.assertEqual(known, novel)

    def test_no_evidence_yields_uniform_distribution(self):
        """Regression: a bias term once made this answer gibberish at 92%.

        With scores built only from evidence, zero evidence must leave every
        class equally likely — the network's own form of "bilmiyorum".
        """
        _, confidence = self.network.predict([])
        self.assertAlmostEqual(confidence, 1.0 / len(self.network.classes))


class TestIntuitionUsesTheNetworkAsAHint(unittest.TestCase):
    def setUp(self):
        self.intuition = Intuition(network=MiniNetwork.default())

    def test_a_parsed_sentence_is_not_second_guessed(self):
        """The parser is the authority: a match is the evidence."""
        intent = self.intuition.understand("robotlar uçar")
        self.assertEqual(intent.kind, TEACH)
        self.assertEqual(intent.confidence, 1.0)
        self.assertIsNone(intent.resembles)

    def test_an_unparsed_sentence_gets_a_guess(self):
        intent = self.intuition.understand("zırf zurf zarf qqq")
        self.assertEqual(intent.kind, UNKNOWN)
        self.assertIsNotNone(intent.resembles)


if __name__ == "__main__":
    unittest.main()
