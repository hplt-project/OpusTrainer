import random
import unittest
from opustrainer.modifiers.retokenize import RetokenizeModifier


VOCAB = 'contrib/test-data/vocab.zhen.spm'


def first(it):
  return next(iter(it))


class TestRetokenizer(unittest.TestCase):
  maxDiff = None

  def test_identity(self):
    """The default tokenizer and detokenizer together should have no impact on
    the output."""
    tokenizer = RetokenizeModifier(
      src=dict(detokenize='spaces', tokenize=f'spaces'),
      trg=dict(detokenize='spaces', tokenize=f'spaces'))

    test_pair = [
      'This is a sentence split by spaces .',
      'Is this a sentence that is split by spaces ?',
      '0-1 1-0 2-2 3-3 4-6 5-7 6-8 7-9',
    ]

    self.assertEqual(first(tokenizer(['\t'.join(test_pair)])).split('\t'), test_pair)

  def test_multiple_spaces(self):
    """Spaces tokenizer should have no impact on the output, but it will 
    normalize spaces due to how it is implemented."""
    tokenizer = RetokenizeModifier(
      src=dict(detokenize='spaces', tokenize=f'spaces'),
      trg=dict(detokenize='spaces', tokenize=f'spaces'))

    test_pair = [
      'This is a    sentence split by spaces .',
      'Is this a sentence    that is split by spaces ?',
      '0-1 1-0 2-2 3-3 4-6 5-7 6-8 7-9',
    ]

    ref_pair = [
      'This is a sentence split by spaces .',
      'Is this a sentence that is split by spaces ?',
      '0-1 1-0 2-2 3-3 4-6 5-7 6-8 7-9',
    ]

    self.assertEqual(first(tokenizer(['\t'.join(test_pair)])).split('\t'), ref_pair)

  def test_retokenize(self):
    tokenizer = RetokenizeModifier(
      src=dict(detokenize='moses:en', tokenize=f'spm:{VOCAB}'),
      trg=dict(detokenize='moses:zh', tokenize=f'spm:{VOCAB}'))

    out = tokenizer(['\t'.join([
      'This is a simple test statement 🤣 .',
      #^0   ^1 ^2 ^3    ^4   ^5        ^6 ^7
      '这 是 一个 简单 的 测试 语 句 🤣 。',
      #^0 ^1 ^2  ^3   ^4 ^5   ^6 ^7 ^8 ^9
      '0-0 1-1 2-2 3-3 3-4 4-5 5-6 5-7 6-8 7-9',
    ])])
    self.assertEqual(first(out), '\t'.join([
      'This is a simple test statement 🤣.',
      #[This][ is][ a][ simple][ test][ statement][ ][] [] [] [🤣][.]
      #^0    ^1   ^2  ^3       ^4     ^5          ^6 ^7 ^8 ^9 ^10 ^11 
      '这是一个简单的测试语句 🤣 。',
      #[这][是][一][个][简][单][的][测][试][语][句] [ ] []  []  []  [🤣][ 。]
      #^0  ^1  ^2  ^3 ^4  ^5  ^6  ^7  ^8  ^9  ^10 ^11 ^12 ^13 ^14 ^15  ^16
      '0-0 1-1 2-2 2-3 3-4 3-5 3-6 4-7 4-8 5-9 5-10 10-15 11-16',
      # 0-0 [This]      [这]    0-0
      # 1-1 [is]        [是]    1-1
      # 2-2 [a]         [一个]  2-2 2-3
      # 3-3 [simple]    [简单]  3-4 3-5
      # 3-4 [simple]    [的]    3-6
      # 4-5 [test]      [测试]  4-7 4-8
      # 5-6 [statement] [语]    5-9
      # 5-7 [statement] [句]    5-10 (6-11)
      # 6-8 [🤣]        [🤣]   (7-12 8-13 9-14) 10-15
      # 7-9 [.]         [。]    11-16
    ]))

  def test_retokenize_icu(self):
    tokenizer = RetokenizeModifier(
      src=dict(detokenize='icu:en', tokenize=f'spm:{VOCAB}'),
      trg=dict(detokenize='icu:zh', tokenize=f'spm:{VOCAB}'))

    out = tokenizer(['\t'.join([
      'This ▁ is ▁ a ▁ simple ▁ test ▁ statement ▁ 🤣 .',
      #^0   ^1^2 ^3^4^5^6     ^7^8   ^9^10       ^11^12^13
      '这 是 一个 简单 的 测试 语 句 ▁ 🤣 ▁ 。',
      #^0 ^1 ^2  ^3  ^4 ^5  ^6 ^7^8 ^9^10^11
      '0-0 2-1 4-2 6-3 6-4 8-5 10-6 10-7 12-9 13-11',
    ])])

    self.assertEqual(first(out), '\t'.join([
      'This is a simple test statement 🤣.',
      #[This][ is][ a][ simple][ test][ statement][ ][] [] [] [🤣][.]
      #^0    ^1   ^2  ^3       ^4     ^5          ^6 ^7 ^8 ^9 ^10 ^11
      '这是一个简单的测试语句 🤣 。',
      #[这][是][一][个][简][单][的][测][试][语][句] [ ] []  []  []  [🤣][ 。]
      #^0  ^1  ^2  ^3 ^4  ^5  ^6  ^7  ^8  ^9  ^10 ^11 ^12 ^13 ^14 ^15  ^16
      '0-0 1-1 2-2 2-3 3-4 3-5 3-6 4-7 4-8 5-9 5-10 10-15 11-16',
      # 0-0 [This]      [这]    0-0
      # 1-1 [is]        [是]    1-1
      # 2-2 [a]         [一个]  2-2 2-3
      # 3-3 [simple]    [简单]  3-4 3-5
      # 3-4 [simple]    [的]    3-6
      # 4-5 [test]      [测试]  4-7 4-8
      # 5-6 [statement] [语]    5-9
      # 5-7 [statement] [句]    5-10 (6-11)
      # 6-8 [🤣]        [🤣]   (7-12 8-13 9-14) 10-15
      # 7-9 [.]         [。]    11-16
    ]))

