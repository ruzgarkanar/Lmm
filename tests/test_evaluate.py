"""The scorecard itself has to be trustworthy before its numbers mean anything.

A benchmark that passes because it asks nothing, or that varies between runs, is
worse than none — it turns a regression into a green tick.
"""
import unittest

from lmm.evaluate import (run, coverage, retrieval, scope, grounding, synthesis,
                          measurement, _taught, COVERAGE, DERIVED, UNKNOWN)
import os
import tempfile


class TestTheScorecard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run()

    def test_every_heading_is_scored(self):
        names = {result.name for result in self.results}
        self.assertEqual(names, {"kapsam", "çıkarım", "sınır", "temellendirme",
                                 "revizyon", "sıra", "sentez", "ölçüm"})

    def test_nothing_scores_out_of_zero(self):
        """An empty category would pass silently and mean nothing."""
        for result in self.results:
            self.assertGreater(result.total, 0, result.name)

    def test_the_system_passes_its_own_scorecard(self):
        for result in self.results:
            self.assertEqual(result.passed, result.total,
                             f"{result.name}: {result.passed}/{result.total}")

    def test_the_run_repeats_exactly(self):
        again = run()
        self.assertEqual([(r.name, r.passed, r.total) for r in self.results],
                         [(r.name, r.passed, r.total) for r in again])


class TestTheScorecardCanFail(unittest.TestCase):
    """A benchmark that cannot fail is decoration."""

    def setUp(self):
        self.session = _taught(os.path.join(tempfile.mkdtemp(), "bos.lmm"))

    def test_an_untaught_memory_scores_badly_on_coverage(self):
        from lmm.cli import Session
        empty = Session(os.path.join(tempfile.mkdtemp(), "hic.lmm"))
        self.assertEqual(coverage(empty).passed, 0)
        self.assertEqual(retrieval(empty).passed, 0)

    def test_an_untaught_memory_still_scores_full_on_scope(self):
        """Knowing nothing is not an excuse to invent — it is the easy case."""
        from lmm.cli import Session
        empty = Session(os.path.join(tempfile.mkdtemp(), "hic2.lmm"))
        self.assertEqual(scope(empty).passed, len(UNKNOWN))

    def test_the_questions_are_not_trivially_the_same(self):
        asked = {q for q, _ in COVERAGE} | {q for q, _ in DERIVED} | set(UNKNOWN)
        self.assertEqual(len(asked), len(COVERAGE) + len(DERIVED) + len(UNKNOWN))


if __name__ == "__main__":
    unittest.main()
