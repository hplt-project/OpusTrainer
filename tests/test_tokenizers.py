import unittest

from opustrainer.tokenizers import make_tokenizer, make_detokenizer

@unittest.skip("requires installing pyicu")
class TestTokenizers(unittest.TestCase):

  def test_tokenize_detokenize_icu_en(self):
    """
    Tests lossless text reconstruction by the ICU tokenizer for English.
    Requires installation with the steps specified in https://pypi.org/project/PyICU/
    """
    tokenizer = make_tokenizer('icu:en')
    detokenizer = make_detokenizer('icu:en')
    text = '“This is,” a simple test statement 🤣.'

    tokens, _ = tokenizer.tokenize(text)
    detokenized, _ = detokenizer.detokenize(tokens)

    self.assertEqual(text, detokenized)
    self.assertEqual("“ This ▁ is , ” ▁ a ▁ simple ▁ test ▁ statement ▁ 🤣.", " ".join(tokens))


  def test_tokenize_detokenize_icu_zh(self):
    """
    Tests lossless text reconstruction by the ICU tokenizer for Chinese.
    Requires installation with the steps specified in https://pypi.org/project/PyICU/
    """
    tokenizer = make_tokenizer('icu:zh')
    detokenizer = make_detokenizer('icu:zh')
    text = '这是一个简单的测试语句 🤣 。'

    tokens, _ = tokenizer.tokenize(text)
    detokenized, _ = detokenizer.detokenize(tokens)

    self.assertEqual(text, detokenized)
    self.assertEqual("这 是 一个 简单 的 测试 语 句 ▁ 🤣▁ 。", " ".join(tokens))
