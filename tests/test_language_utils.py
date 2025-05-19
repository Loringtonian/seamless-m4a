import unittest
from language_utils import detect_language, identify_spanish_lines

class TestLanguageUtils(unittest.TestCase):
    def test_detect_language_spanish(self):
        self.assertEqual(detect_language('que tenia que hacer'), 'es')

    def test_detect_language_other(self):
        self.assertEqual(detect_language('Ich habe Deutsch gelernt'), 'other')

    def test_identify_spanish_lines(self):
        lines = ['hola como estas', 'Ich habe Deutsch gelernt', 'que pasa']
        idx = identify_spanish_lines(lines)
        self.assertEqual(idx, [0, 2])

    def test_detect_language_with_accent(self):
        self.assertEqual(detect_language('¿Dónde estás?'), 'es')

    def test_reference_spanish_lines(self):
        from pathlib import Path

        ref_path = Path(__file__).resolve().parents[1] / 'corrected transcript.txt'
        lines = [l.strip() for l in ref_path.read_text(encoding='utf-8').splitlines() if l.strip()]
        idx = identify_spanish_lines(lines)
        self.assertEqual(idx, [18, 20])

if __name__ == '__main__':
    unittest.main()
