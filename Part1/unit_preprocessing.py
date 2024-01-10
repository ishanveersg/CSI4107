from preprocessing import *
import unittest
import re
import string

class TestExtractText(unittest.TestCase):
    def test_extract_text(self):
        document = '<DOC>\n<TEXT>This is some text. It contains punctuation!</TEXT>\n</DOC>'
        expected_output = 'This is some text It contains punctuation'
        self.assertEqual(extract_text(document), expected_output)

    def test_extract_multiple_text(self):
        document = '<DOC>\n<TEXT>This is some text. It contains punctuation!</TEXT>\n<TEXT>This is more text.</TEXT>\n</DOC>'
        expected_output = 'This is some text It contains punctuation This is more text'
        self.assertEqual(extract_text(document), expected_output)

    def test_extract_empty_text(self):
        document = '<DOC>\n<TEXT></TEXT>\n</DOC>'
        expected_output = ''
        self.assertEqual(extract_text(document), expected_output)

    def test_extract_no_text(self):
        document = '<DOC></DOC>'
        expected_output = ''
        self.assertEqual(extract_text(document), expected_output)

class TestTokenizeString(unittest.TestCase):
    def test_tokenize_string(self):
        input_string = 'This is a test string'
        expected_output = ['This', 'is', 'a', 'test', 'string']
        self.assertEqual(tokenize_string(input_string), expected_output)

    def test_tokenize_string_with_punctuation(self):
        input_string = 'This, is another test! string?'
        expected_output = ['This,', 'is', 'another', 'test!', 'string?']
        self.assertEqual(tokenize_string(input_string), expected_output)

    def test_tokenize_empty_string(self):
        input_string = ''
        expected_output = []
        self.assertEqual(tokenize_string(input_string), expected_output)

    def test_tokenize_single_word_string(self):
        input_string = 'word'
        expected_output = ['word']
        self.assertEqual(tokenize_string(input_string), expected_output)


if __name__ == '__main__':
    unittest.main()