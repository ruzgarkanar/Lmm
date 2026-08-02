"""Counted meaning: geometry from co-occurrence, with no gradient anywhere."""
import unittest

from lmm.vectors import Vectors, cosine, _orthonormalise


def sentences(*lines):
    return [line.split() for line in lines]


CORPUS = sentences(
    "kuşlar uçar", "kuşlar öter", "kuşlar tüylüdür",
    "serçeler uçar", "serçeler öter", "serçeler tüylüdür",
    "kartallar uçar", "kartallar öter", "kartallar tüylüdür",
    "balıklar yüzer", "balıklar pullu", "balıklar suda",
    "somonlar yüzer", "somonlar pullu", "somonlar suda",
    "sazanlar yüzer", "sazanlar pullu", "sazanlar suda",
)


class TestCounting(unittest.TestCase):
    def test_it_learns_without_being_told_anything_about_the_language(self):
        vectors = Vectors(dimensions=6, minimum=1).learn(CORPUS)
        self.assertIn("uçar", vectors.vectors)
        # Asking for six directions out of a corpus that only varies in four
        # gives four. Padding it out with directions the data does not support
        # would be inventing structure, so the collapsed ones are dropped.
        width = len(vectors.vector("uçar"))
        self.assertTrue(0 < width <= 6)
        self.assertTrue(all(len(v) == width for v in vectors.vectors.values()))

    def test_words_that_keep_the_same_company_end_up_together(self):
        """The whole claim, in one assertion: distribution carries class."""
        vectors = Vectors(dimensions=6, minimum=1).learn(CORPUS)
        birds = vectors.nearest("serçeler", ["kartallar", "somonlar"], count=1)
        self.assertEqual(birds[0][0], "kartallar")
        fish = vectors.nearest("somonlar", ["sazanlar", "kartallar"], count=1)
        self.assertEqual(fish[0][0], "sazanlar")

    def test_a_word_it_never_saw_has_no_vector_rather_than_a_guess(self):
        vectors = Vectors(dimensions=6, minimum=1).learn(CORPUS)
        self.assertIsNone(vectors.vector("ejderha"))
        self.assertEqual(vectors.similar("ejderha"), [])

    def test_rare_words_are_left_out_instead_of_being_modelled_on_one_sighting(self):
        corpus = CORPUS + sentences("devekuşu koşar")
        vectors = Vectors(dimensions=6, minimum=2).learn(corpus)
        self.assertIsNone(vectors.vector("devekuşu"))

    def test_the_same_corpus_always_gives_the_same_vectors(self):
        """No seed anywhere, so a regression is a regression and not a reroll."""
        first = Vectors(dimensions=6, minimum=1).learn(CORPUS)
        second = Vectors(dimensions=6, minimum=1).learn(CORPUS)
        self.assertEqual(first.vector("uçar"), second.vector("uçar"))

    def test_an_empty_corpus_produces_nothing_and_does_not_fall_over(self):
        vectors = Vectors(dimensions=6, minimum=1).learn([])
        self.assertEqual(vectors.vectors, {})


class TestArithmetic(unittest.TestCase):
    def test_cosine_of_a_vector_with_itself_is_one(self):
        self.assertAlmostEqual(cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_cosine_of_perpendicular_vectors_is_zero(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_cosine_with_nothing_is_zero_rather_than_an_error(self):
        self.assertEqual(cosine([], [1.0]), 0.0)

    def test_orthonormalising_gives_unit_length_and_no_overlap(self):
        basis = _orthonormalise([[3.0, 0.0], [1.0, 2.0]])
        self.assertEqual(len(basis), 2)
        for vector in basis:
            self.assertAlmostEqual(sum(a * a for a in vector), 1.0)
        self.assertAlmostEqual(cosine(basis[0], basis[1]), 0.0)

    def test_a_direction_already_covered_is_dropped_not_kept_as_noise(self):
        basis = _orthonormalise([[1.0, 0.0], [2.0, 0.0]])
        self.assertEqual(len(basis), 1)


class TestCarryingItAround(unittest.TestCase):
    def test_vectors_survive_being_written_out_and_read_back(self):
        vectors = Vectors(dimensions=6, minimum=1).learn(CORPUS)
        restored = Vectors.from_dict(vectors.to_dict())
        self.assertEqual(sorted(restored.vectors), sorted(vectors.vectors))
        for before, after in zip(vectors.vector("uçar"), restored.vector("uçar")):
            self.assertAlmostEqual(before, after, places=4)


if __name__ == "__main__":
    unittest.main()
