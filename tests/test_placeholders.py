import random
import unittest
import tempfile

from textwrap import dedent

from opustrainer.modifiers.placeholders import PlaceholderTagModifier
from opustrainer.trainer import CurriculumLoader
from opustrainer import logger


class TestTagger(unittest.TestCase):
  def setUp(self):
    random.seed(1)

  def test_tagger_out_of_index(self):
    """Alignment pairs that do not map to any tokens should raise an error"""
    tagger = PlaceholderTagModifier(probability=1)
    with self.assertRaisesRegex(ValueError, r'Out-of-bound alignment pairs: .+'):
      output = tagger('Hello world\tHallo Welt\t0-0 1-2')

  def test_tagger_tagging(self):
    """Default mode is tagging, and will hint the target word in the source input"""
    tagger = PlaceholderTagModifier(probability=1)
    output = tagger('Hello world\tHallo Welt\t0-0 1-1')
    self.assertEqual(output, '__source__ Hello __target__ Hallo __done__ __source__ world __target__ Welt __done__\tHallo Welt\t1-0 3-0 6-1 8-1')
    #                         ^0         ^1    ^2         ^3    ^4       ^5         ^6    ^7         ^8   ^9        ^0    ^1

  def test_tagger_replace(self):
    """Replace mode is the same as tagging mode, except that the target word
    will be random noise, teaching the model to just copy it as is."""
    tagger = PlaceholderTagModifier(probability=0.25, replace=1)
    output = tagger('Hello world\tHallo Welt\t0-0 1-1')
    self.assertEqual(output, '''__source__ Hello __target__ িৡহ __done__ world\tিৡহ Welt\t1-0 3-0 5-1''')
    #                           ^0         ^1    ^2         ^3   ^4       ^5      ^0   ^1

  def test_tagger_augment(self):
    """Augment mode will add random noise without tags to both source and target
    sentence, teaching the model to copy strings it doesn't understand."""
    tagger = PlaceholderTagModifier(probability=1, augment=1)
    output = tagger('Hello world\tHallo Welt\t0-0 1-1')
    self.assertEqual(output, '''Hello িৡহ world ЇӤӕѣѮ қӃӄЀҲ\tHallo িৡহ Welt ЇӤӕѣѮ қӃӄЀҲ\t0-0 1-1 2-2 3-3 4-4''')

  def test_tagger_zh_src(self):
    '''Tests the tagger with zh on the source side'''
    tagger = PlaceholderTagModifier(probability=0.6, custom_detok_src='zh')
    with open('contrib/test-data/clean.zhen.10', 'r', encoding='utf-8') as myinput, \
         open('contrib/test-data/clean.zhen.ref.06.4.src', 'r', encoding='utf-8') as reference:
        for line in myinput:
          test = tagger(line)
          ref = reference.readline()[:-1]
          self.assertEqual(test, ref)
  
  def test_tagger_zh_trg(self):
    '''Tests the tagger with zh on the target side'''
    tagger = PlaceholderTagModifier(probability=0.6, custom_detok_src=None, custom_detok_trg='zh')
    with open('contrib/test-data/clean.enzh.10', 'r', encoding='utf-8') as myinput, \
         open('contrib/test-data/clean.enzh.ref.06.4.trg', 'r', encoding='utf-8') as reference:
        for line in myinput:
          test = tagger(line)
          ref = reference.readline()[:-1]
          self.assertEqual(test, ref)

  def test_tagger_no_zh(self):
    '''Tests the tagger without zh detokenizer'''
    tagger = PlaceholderTagModifier(probability=0.6)
    with open('contrib/test-data/clean.enzh.10', 'r', encoding='utf-8') as myinput, \
         open('contrib/test-data/clean.enzh.ref.06.4.none', 'r', encoding='utf-8') as reference:
        for line in myinput:
          test = tagger(line)
          ref = reference.readline()[:-1]
          self.assertEqual(test, ref)

  def test_tagger_zh_src_augment_replace(self):
    '''Tests the tagger with zh on the source side'''
    tagger = PlaceholderTagModifier(probability=0.6, custom_detok_src='zh', custom_detok_trg=None,
                                     augment=0.4, replace=0.4)
    with open('contrib/test-data/clean.zhen.10', 'r', encoding='utf-8') as myinput, \
         open('contrib/test-data/clean.zhen.ref.06.4.04.04.src', 'r', encoding='utf-8') as reference:
        for line in myinput:
          test = tagger(line)
          ref = reference.readline()[:-1]
          self.assertEqual(test, ref)

  def test_warn_if_tag_modifier_is_not_last(self):
    with tempfile.NamedTemporaryFile(suffix='.log', prefix="placeholder") as tmpfile:
        logger.setup_logger(outputfilename=tmpfile.name, disable_stderr=True)
        loader = CurriculumLoader()
        loader.load(dedent("""
          datasets: {}
          stages: []
          seed: 1
          modifiers:
            - Tags: 1.0
            - UpperCase: 1.0
        """))
        logger.logging.shutdown()
        tmpfile.seek(0)
        warning = tmpfile.readline().decode('utf-8')
        self.assertRegex(warning, r"WARNING")
        self.assertRegex(warning, r"Tags modifier should to be the last modifier to be applied")
