import unittest

from parameterized import parameterized

from opustrainer.modifiers.punctuation import RemoveEndPunctuationModifier


class TestMerge(unittest.TestCase):

    @parameterized.expand([
        ('.', '。'),
        ('?', '？'),
        ('!', '!')
    ])
    def test_remove_end_punct(self, src_punct, trg_punct):
        ex = '\t'.join([
          f'This ▁ is ▁ a ▁ simple ▁ test ▁ statement ▁ 🤣 {src_punct}',
          f'这 是 一个 简单 的 测试 语 句 ▁ 🤣 ▁ {trg_punct}',
          '0-0 2-1 4-2 6-3 6-4 8-5 10-6 10-7 12-9 13-11',
        ])
        mod = RemoveEndPunctuationModifier(1.0)
        out = list(mod([ex]))[0]
        src, trg, aln = out.split('\t')
        self.assertEqual(src, 'This ▁ is ▁ a ▁ simple ▁ test ▁ statement ▁ 🤣')
        self.assertEqual(trg, '这 是 一个 简单 的 测试 语 句 ▁ 🤣 ▁')
        self.assertEqual(aln, '0-0 2-1 4-2 6-3 6-4 8-5 10-6 10-7 12-9')

    def test_do_not_remove_end_punct_if_different(self):
        ex = '\t'.join([
          'This ▁ is ▁ a ▁ simple ▁ test ▁ statement ▁ 🤣 ▁ ?',
          '这 是 一个 简单 的 测试 语 句 ▁ 🤣 ▁ 。',
          '0-0 2-1 4-2 6-3 6-4 8-5 10-6 10-7 12-9 13-11',
        ])
        mod = RemoveEndPunctuationModifier(1.0)
        out = list(mod([ex]))[0]
        self.assertEqual(out, ex)

    def test_remove_punct_no_alignment(self):
        ex = '\t'.join([
          'This ▁ is ▁ a ▁ simple ▁ test ▁ statement ▁ 🤣 .',
          '这 是 一个 简单 的 测试 语 句 ▁ 🤣 ▁ 。'
        ])
        mod = RemoveEndPunctuationModifier(1.0)
        out = list(mod([ex]))[0]
        src, trg = out.split('\t')
        self.assertEqual(src, 'This ▁ is ▁ a ▁ simple ▁ test ▁ statement ▁ 🤣')
        self.assertEqual(trg, '这 是 一个 简单 的 测试 语 句 ▁ 🤣 ▁')
