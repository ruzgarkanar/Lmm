import unittest

from lmm.network import MiniNetwork, training_data
from lmm.intuition import tokenize, Intuition, TEACH


class TestMiniNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = MiniNetwork.default()   # trained on the seed data

    def test_training_data_covers_four_classes(self):
        classes = {label for _, label in training_data()}
        self.assertEqual(classes, {"TEACH_TYPE", "TEACH_ABILITY",
                                   "ASK_ABILITY", "ASK_DEFINITION"})

    def test_recognises_learned_patterns(self):
        label, confidence = self.network.predict(tokenize("kedi bir hayvandır"))
        self.assertEqual(label, "TEACH_TYPE")
        self.assertGreater(confidence, 0.6)
        label, confidence = self.network.predict(tokenize("balık yüzer mi"))
        self.assertEqual(label, "ASK_ABILITY")
        self.assertGreater(confidence, 0.6)

    def test_unseen_input_yields_low_confidence(self):
        _, confidence = self.network.predict(["florp", "glorp", "zzz"])
        self.assertLess(confidence, 0.5)   # cannot be assertive about the unseen

    def test_no_evidence_yields_uniform_distribution(self):
        """Regression: a bias term once made this answer gibberish at 92%.

        With scores built only from evidence, zero evidence must leave every
        class equally likely — the network's own form of "bilmiyorum".
        """
        _, confidence = self.network.predict(["florp", "glorp", "zzz"])
        self.assertAlmostEqual(confidence, 1.0 / len(self.network.classes))
        _, empty_confidence = self.network.predict([])
        self.assertAlmostEqual(empty_confidence, 1.0 / len(self.network.classes))

    def test_intuition_integration(self):
        intuition = Intuition(network=self.network)
        intent = intuition.understand("Penguen bir kuştur.")
        self.assertEqual(intent.kind, TEACH)
        self.assertGreater(intent.confidence, 0.6)


if __name__ == "__main__":
    unittest.main()
