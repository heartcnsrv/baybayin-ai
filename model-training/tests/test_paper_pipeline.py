import unittest

import cv2
import numpy as np

from baybayin_preprocess import Params, normalize_glyph, segment
from evaluate import edit_ops, label_from_filename, reading_matches
from translate_pipeline import (BigramLM, Cand, GlyphObs, Lexicon, TParams, baybayin_unicode, most_probable_reading,
                                rank_sequence, search_word, syllable_options, word_to_glyphs)


class ReadingsTest(unittest.TestCase):
    def test_mark_readings(self):
        self.assertEqual(syllable_options(GlyphObs("ka", "none")), ["ka"])
        self.assertEqual(syllable_options(GlyphObs("ka", "above")), ["ke", "ki"])
        self.assertEqual(syllable_options(GlyphObs("ka", "below")), ["ko", "ku"])
        self.assertEqual(syllable_options(GlyphObs("ka", "virama")), ["k"])

    def test_da_also_reads_ra(self):
        self.assertEqual(syllable_options(GlyphObs("dara", "none")), ["da", "ra"])

    def test_vowels(self):
        self.assertEqual(syllable_options(GlyphObs("ei")), ["e", "i"])
        self.assertEqual(syllable_options(GlyphObs("ou")), ["o", "u"])


class CandidatesTest(unittest.TestCase):
    def setUp(self):
        self.lex = Lexicon(["puro", "puso", "bahay"])
        self.p = TParams()

    def test_only_word_list_candidates_survive(self):
        obs = [GlyphObs("pa", "below"), GlyphObs("dara", "below")]  # pu/po + du/do/ru/ro
        self.assertEqual([c.word for c in search_word(obs, self.lex, self.p)], ["puro"])

    def test_no_match_keeps_most_probable_reading_unverified(self):
        obs = [GlyphObs("ka", "above"), GlyphObs("dara", "below")]
        cands = search_word(obs, self.lex, self.p)
        self.assertEqual(len(cands), 1)
        self.assertTrue(cands[0].unverified)
        self.assertEqual(cands[0].word, "kidu")
        self.assertEqual(most_probable_reading(obs), "kidu")

    def test_ranking_uses_corpus_frequency_then_neighbours(self):
        lm = BigramLM({"puso", "puro", "ang"})
        lm.fit_texts(["puso puso puso", "ang puro"])
        single = rank_sequence([[Cand("puro"), Cand("puso")]], lm, self.p)
        self.assertEqual(single[0][1][0].word, "puso")  # more frequent alone
        pair = rank_sequence([[Cand("ang")], [Cand("puro"), Cand("puso")]], lm, self.p)
        self.assertEqual(pair[0][1][1].word, "puro")   # fits after "ang"

    def test_spelling_and_unicode(self):
        self.assertEqual(word_to_glyphs("bahay"), [("ba", "none"), ("ha", "none"), ("ya", "virama")])
        self.assertEqual(word_to_glyphs("pangarap"),
                         [("pa", "none"), ("nga", "none"), ("dara", "none"), ("pa", "virama")])
        self.assertIsNone(word_to_glyphs("fiesta"))
        obs = [GlyphObs(c, m) for c, m in word_to_glyphs("bahay")]
        self.assertEqual(baybayin_unicode(obs), "ᜊᜑᜌ᜔")


class SegmentationTest(unittest.TestCase):
    def test_normalize_is_28x28_binary(self):
        g = normalize_glyph(np.pad(np.ones((10, 30), np.uint8) * 255, 5))
        self.assertEqual(g.shape, (28, 28))
        self.assertTrue(set(np.unique(g)) <= {0, 255})

    def test_kudlit_attached_to_overlapping_character(self):
        bw = np.zeros((80, 200), np.uint8)
        cv2.rectangle(bw, (20, 30), (60, 60), 255, -1)     # character 1
        cv2.rectangle(bw, (120, 30), (160, 60), 255, -1)   # character 2
        cv2.circle(bw, (140, 18), 4, 255, -1)              # kudlit over character 2
        res = segment(bw, Params(word_gap_ratio=5.0))
        glyphs = [g for line in res.lines for w in line.words for g in w.glyphs]
        self.assertEqual(len(glyphs), 2)
        self.assertEqual([len(g.marks) for g in glyphs], [0, 1])


class EvaluationHelpersTest(unittest.TestCase):
    def test_label_and_edit_ops(self):
        self.assertEqual(label_from_filename("1 Gwapo.PNG"), "gwapo")
        self.assertEqual(edit_ops(list("abc"), list("abd")), (1, 2))

    def test_ambiguity_is_not_an_error(self):
        obs = [GlyphObs("pa", "below"), GlyphObs("dara", "below")]
        self.assertTrue(reading_matches(obs, "puro"))
        self.assertTrue(reading_matches(obs, "podo"))
        self.assertFalse(reading_matches(obs, "pura"))


if __name__ == "__main__":
    unittest.main()
