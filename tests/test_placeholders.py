import random
import unittest

from textwrap import dedent

from opustrainer.modifiers.placeholders import PlaceholderTagModifier
from opustrainer.trainer import CurriculumLoader


def first(it):
  return next(iter(it))


class TestTagger(unittest.TestCase):
  def setUp(self):
    random.seed(1)

  def test_tagger_tagging(self):
    """Default mode is tagging, and will hint the target word in the source input"""
    tagger = PlaceholderTagModifier(probability=1)
    tagger.print_alignments = True
    output = tagger(['Hello world\tHallo Welt\t0-0 1-1'])
    self.assertEqual(first(output), '__source__ Hello __target__ Hallo __done__ __source__ world __target__ Welt __done__\tHallo Welt\t1-0 3-0 6-1 8-1')
    #                                ^0         ^1    ^2         ^3    ^4       ^5         ^6    ^7         ^8   ^9        ^0    ^1

  def test_tagger_replace(self):
    """Replace mode is the same as tagging mode, except that the target word
    will be random noise, teaching the model to just copy it as is."""
    tagger = PlaceholderTagModifier(probability=0.25, replace=1)
    tagger.print_alignments = True
    output = tagger(['Hello world\tHallo Welt\t0-0 1-1'])
    self.assertEqual(first(output), '__source__ Hello __target__ িৡহ __done__ world\tিৡহ Welt\t1-0 3-0 5-1')
    #                                ^0         ^1    ^2         ^3   ^4       ^5      ^0   ^1

  def test_tagger_augment(self):
    """Augment mode will add random noise without tags to both source and target
    sentence, teaching the model to copy strings it doesn't understand."""
    tagger = PlaceholderTagModifier(probability=1, augment=1)
    tagger.print_alignments = True
    output = tagger(['Hello world\tHallo Welt\t0-0 1-1'])
    self.assertEqual(first(output), 'Hello িৡহ world ЇӤӕѣѮ қӃӄЀҲ\tHallo িৡহ Welt ЇӤӕѣѮ қӃӄЀҲ\t0-0 1-1 2-2 3-3 4-4')

  def test_tagger_augment_icu(self):
    """Augment mode will add random noise without tags to both source and target
    sentence, teaching the model to copy strings it doesn't understand."""
    tagger = PlaceholderTagModifier(probability=1, augment=1, tag=0, custom_detok_src='icu:en', custom_detok_trg='icu:de')
    tagger.print_alignments = True
    output = tagger(['Hello ▁ world\tHallo ▁ Welt\t0-0 1-1 2-2'])
    self.assertEqual(first(output), 'Hello িৡহ world ټ؇ۤە٣ٮڛۃ \tHallo িৡহ Welt ټ؇ۤە٣ٮڛۃ \t0-0 1-1 2-2 3-3')


  def test_retokenize(self):
    """Pass the spm vocab to the placeholder tag generator so that it can
    retokenize the input, and update the alignments accordingly."""
    tagger = PlaceholderTagModifier(
      probability=0.25,
      custom_detok_src='en',
      custom_detok_trg='zh',
      spm_vocab='contrib/test-data/vocab.zhen.spm') # type: ignore Path vs String type issue
    
    output = tagger(['\t'.join([
      'This is a simple test statement 🤣 .',
      #^0   ^1 ^2 ^3    ^4   ^5        ^6 ^7
      '这 是 一个 简单 的 测试 语 句 🤣 。',
      #^0 ^1 ^2  ^3   ^4 ^5   ^6 ^7 ^8 ^9
      '0-0 1-1 2-2 3-3 3-4 4-5 5-6 5-7 6-8 7-9',
    ])])
    self.assertEqual(first(output).split('\t'), [
      '__source__ This __target__ 这 __done__ is a simple test statement 🤣.',
      # [][__source__][This][ ][__target__][这][ ][__done__][ is][ a][ simple][ test][ statement][ ] []  []  []  [🤣][.]
      #^0 ^1          ^2    ^3 ^4          ^5  ^6 ^7        ^8   ^9  ^10       ^11    ^12         ^13 ^14 ^15 ^16 ^17 ^18 
      # Note the empty [] tokens before the special tokens: these are the spaces
      # that are not part of the special marker tokens. It depends on how the
      # spm vocab is trained.
      '这是一个简单的测试语句 🤣 。',
      #[这][是][一][个][简][单][的][测][试][语][句] [ ] []  []  []  [🤣][ 。]
      #^0  ^1  ^2  ^3 ^4  ^5  ^6  ^7  ^8  ^9  ^10 ^11 ^12 ^13 ^14 ^15  ^16
      '2-0 5-0 8-1 9-2 9-3 10-4 10-5 10-6 11-7 11-8 12-9 12-10 17-15 18-16',
      # 0-0 [This]      [这]    2-0
      #     [这]        [这]    5-0
      # 1-1 [is]        [是]    8-1
      # 2-2 [a]         [一个]  9-2 9-3
      # 3-3 [simple]    [简单]  10-4 10-5
      # 3-4 [simple]    [的]    10-6
      # 4-5 [test]      [测试]  11-7 11-8
      # 5-6 [statement] [语]    12-9
      # 5-7 [statement] [句]    12-10 (13-11)
      # 6-8 [🤣]        [🤣]   (14-12 15-13 16-14) 17-15
      # 7-9 [.]         [。]    18-16
    ])

  def test_augment_icu(self):
    """Pass the spm vocab to the placeholder tag generator so that it can
    retokenize the input, and update the alignments accordingly."""
    tagger = PlaceholderTagModifier(
      probability=0.2,
      augment=1,
      tag=0,
      custom_detok_src='icu:en',
      custom_detok_trg='icu:zh',
      spm_vocab='contrib/test-data/vocab.zhen.spm')  # type: ignore Path vs String type issue

    output = tagger(['\t'.join([
      'This ▁ is ▁ a ▁ simple ▁ test ▁ statement ▁ 🤣 .',
      #^0   ^1^2 ^3^4^5^6     ^7^8   ^9^10       ^11^12^13
      '这 是 一个 简单 的 测试 语 句 ▁ 🤣 ▁ 。',
      #^0 ^1 ^2  ^3  ^4 ^5  ^6 ^7^8 ^9^10^11
      '0-0 2-1 4-2 6-3 6-4 8-5 10-6 10-7 12-9 13-11',
    ])])

    self.assertEqual(first(output).split('\t'), [
      'This িৡহ is a simple test statement 🤣.',
      # ['This', ' ', '', '', 'ি', '', '', 'ৡ', '', '', 'হ', ' is', ' a', ' simple', ' test', ' statement', ' ', '', '', '', '🤣', '.']
      '这 িৡহ 是一个简单的测试语句 🤣 。',
      # ['这', ' ', '', '', 'ি', '', '', 'ৡ', '', '', 'হ', ' 是', '一', '个', '简', '单', '的', '测', '试', '语', '句', ' ', '', '', '', '🤣', ' 。']
      '0-0 4-4 4-5 4-6 4-7 4-8 4-9 4-10 5-4 5-5 5-6 5-7 5-8 5-9 5-10 6-4 6-5 6-6 '
      '6-7 6-8 6-9 6-10 7-4 7-5 7-6 7-7 7-8 7-9 7-10 8-4 8-5 8-6 8-7 8-8 8-9 8-10 '
      '9-4 9-5 9-6 9-7 9-8 9-9 9-10 10-4 10-5 10-6 10-7 10-8 10-9 10-10 11-11 12-12 '
      '12-13 13-14 13-15 13-16 14-17 14-18 15-19 15-20 20-25 21-26'
    ])

  def test_retokenize_on_non_trigger(self):
    """Pass the spm vocab to the placeholder tag generator so that it can
    retokenize the input, even if probability is 0."""
    tagger = PlaceholderTagModifier(
      probability=0.0,
      custom_detok_src='en',
      custom_detok_trg='zh',
      spm_vocab='contrib/test-data/vocab.zhen.spm') # type: ignore Path vs String type issue
    
    output = tagger(['\t'.join([
      'This is a simple test statement 🤣 .',
      '这 是 一个 简单 的 测试 语 句 🤣 。',
      '0-0 1-1 2-2 3-3 3-4 4-5 5-6 5-7 6-8 7-9',
    ])])
    self.assertEqual(first(output).split('\t'), [
      'This is a simple test statement 🤣.',
      #[This][ is][ a][ simple][ test][ statement][ ] [] [] [] [🤣][.]
      #^0    ^1   ^2  ^3       ^4     ^5          ^6  ^7 ^8 ^9 ^10 ^11 
      '这是一个简单的测试语句 🤣 。',
      #[这][是][一][个][简][单][的][测][试][语][句] [ ] []  []  []  [🤣][ 。]
      #^0  ^1  ^2  ^3 ^4  ^5  ^6  ^7  ^8  ^9  ^10 ^11 ^12 ^13 ^14 ^15  ^16
      '0-0 1-1 2-2 2-3 3-4 3-5 3-6 4-7 4-8 5-9 5-10 10-15 11-16',
    ])

  def test_mode(self):
    """Test that different calls will apply different modifications when
    multiple modes are enabled."""
    tagger = PlaceholderTagModifier(
      probability=1.0,
      custom_detok_src='zh',
      augment=0.33,
      replace=0.33,
      # tag=0.33 is implicit
    )
    
    example = [
      '429 运输 中队 ( 429 野牛) , 使用 CC - 177',
      '429 Transport Squadron (429 Bison Squadron) - Flying the CC-177',
      '0-0 1-1 2-2 3-3 4-3 5-4 5-5 7-5 8-5 9-6 8-7 9-8 10-9',
    ]

    refs = [
      [ # tag + augment * 2
        '429 __source__ 运输 __target__ Transport __done__ 中队んばずがまぃぱぷろ぀゙べぢ゜そるきと (429 野牛), 使用 CC - 177 w;V|#c<X_f =L=v<ZE"Ug',
        '429 Transport Squadron んばずがまぃぱぷろ ぀゙べぢ゜そるきと (429 Bison Squadron) - Flying the CC-177 w;V|#c<X_f =L=v<ZE"Ug',
      ],
      [ # augment + tag * 2
        '429 运输 ѕӥҸӹҶҀ ӯӷѬҁӔө ҫаэшҖӹ __source__ 中队 __target__ Squadron __done__ (429 野牛), 使用 CC - __source__ 177 __target__ ビフ __done__',
        '429 Transport ѕӥҸӹҶҀ ӯӷѬҁӔө ҫаэшҖӹ Squadron (429 Bison Squadron) - Flying the ビフ',
      ],
      [ # replace * 3
        '429 __source__ 运输 __target__ ϲ΋Ιϵϔώϭ ͷϨͻξϔΛΛ __done__ __source__ 中队 __target__ ͷϨͻξϔΛΛ __done__ (429 野牛), 使用 CC - __source__ 177 __target__ ुबॹ६ॉभऺॴढ॔ ॆ्ॺढ़ऀऱ।७३ॺ __done__',
        '429 ϲ΋Ιϵϔώϭ ͷϨͻξϔΛΛ Squadron (429 Bison Squadron) - Flying ुबॹ६ॉभऺॴढ॔ ॆ्ॺढ़ऀऱ।७३ॺ CC-177',
      ]
    ]

    for ref in refs:
      output = tagger(['\t'.join(example)])
      self.assertEqual(first(output), '\t'.join(ref))

  def test_warn_if_tag_modifier_is_not_last(self):
    with self.assertLogs(level='WARNING') as logger_ctx:
      loader = CurriculumLoader()
      loader.load(dedent("""
        datasets: {}
        stages: []
        seed: 1
        modifiers:
          - Tags: 1.0
          - UpperCase: 1.0
      """))
    self.assertRegex(logger_ctx.output[0], r"Tags modifier should to be the last modifier to be applied")

  def test_exception_if_alignment_is_missing(self):
    tagger = PlaceholderTagModifier()
    with self.assertLogs(level='WARNING') as logger_ctx:
      self.assertEqual(list(tagger(['Hello world\tHallo welt\t'])), [])
      self.assertIn('IndexError', logger_ctx.output[0])

  def test_exception_if_alignment_is_invalid(self):
    tagger = PlaceholderTagModifier()
    with self.assertLogs(level='WARNING') as logger_ctx:
      self.assertEqual(list(tagger(['Hello world\tHallo welt\t0-0 1-2'])), [])
      self.assertIn('ValueError', logger_ctx.output[0])
