import unittest
from Compresser import Compresser

class TestCompresser(unittest.TestCase):
    def test_summarize_returns_non_empty_string(self):
        # Call the summarize method with some text
        text = "Some long piece of text to summarize..."
        compresser = Compresser(text)
        compresser.summarize()

        # Assert that the summary is a non-empty string
        self.assertIsInstance(compresser.summary, str)
        self.assertGreater(len(compresser.summary), 0)

if __name__ == '__main__':
    unittest.main()
