import unittest
from grading import compute_transcript_score


class TestGrading(unittest.TestCase):
    def test_example_score(self):
        score = compute_transcript_score(wer=0.15, multilingual_accuracy=0.85, speaker_f1=0.70)
        self.assertAlmostEqual(score, 835, delta=0.1)

    def test_multilingual_gate(self):
        score = compute_transcript_score(wer=0.1, multilingual_accuracy=0.4, speaker_f1=0.9)
        self.assertLessEqual(score, 500)

    def test_no_diarisation(self):
        score = compute_transcript_score(wer=0.0, multilingual_accuracy=1.0, speaker_f1=None)
        self.assertEqual(score, 1000)


if __name__ == '__main__':
    unittest.main()
