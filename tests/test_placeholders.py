import random
import unittest

from textwrap import dedent

from opustrainer.modifiers.placeholders import PlaceholderTagModifier
from opustrainer.trainer import CurriculumLoader


class TestTagger(unittest.TestCase):
  def test_tagger_tagging(self):
    random.seed(1)
    tagger = PlaceholderTagModifier(probability=1)
    output = tagger('Hello world\tHallo Welt\t0-0 1-1')
    self.assertEqual(output, '__source__ Hello __target__ Hallo __done__ __source__ world __target__ Welt __done__\tHallo Welt')

  def test_tagger_out_of_index(self):
    tagger = PlaceholderTagModifier(probability=1)
    with self.assertRaisesRegex(ValueError, 'alignment pair target token index out of range'):
      output = tagger('Hello world\tHallo Welt\t0-0 1-2')

  def test_tagger_augment(self):
    random.seed(1)
    tagger = PlaceholderTagModifier(probability=1, augment=1)
    output = tagger('Hello world\tHallo Welt\t0-0 1-1')
    self.assertEqual(output, '''Hello ټ؇ۤە world ি	Hallo ټ؇ۤە Welt ি''')

  def test_tagger_replace(self):
    random.seed(1)
    tagger = PlaceholderTagModifier(probability=.5, replace=1)
    output = tagger('Hello world\tHallo Welt\t0-0 1-1')
    self.assertEqual(output, '''Hello __source__ world __target__ ি __done__	Hallo ি''')

  def test_tagger_zh_src(self):
    '''Tests the tagger with zh on the source side'''
    random.seed(1)
    tagger = PlaceholderTagModifier(probability=0.6, custom_detok_src='zh')
    with open('contrib/test-data/clean.zhen.10', 'r', encoding='utf-8') as myinput, \
         open('contrib/test-data/clean.zhen.ref.06.4.src', 'r', encoding='utf-8') as reference:
        for line in myinput:
          test = tagger(line)
          ref = reference.readline()[:-1]
          self.assertEqual(test, ref)
  
  def test_tagger_zh_trg(self):
    '''Tests the tagger with zh on the target side'''
    random.seed(1)
    tagger = PlaceholderTagModifier(probability=0.6, custom_detok_src=None, custom_detok_trg='zh')
    with open('contrib/test-data/clean.enzh.10', 'r', encoding='utf-8') as myinput, \
         open('contrib/test-data/clean.enzh.ref.06.4.trg', 'r', encoding='utf-8') as reference:
        for line in myinput:
          test = tagger(line)
          ref = reference.readline()[:-1]
          self.assertEqual(test, ref)

  def test_tagger_no_zh(self):
    '''Tests the tagger without zh detokenizer'''
    random.seed(1)
    tagger = PlaceholderTagModifier(probability=0.6)
    with open('contrib/test-data/clean.enzh.10', 'r', encoding='utf-8') as myinput, \
         open('contrib/test-data/clean.enzh.ref.06.4.none', 'r', encoding='utf-8') as reference:
        for line in myinput:
          test = tagger(line)
          ref = reference.readline()[:-1]
          self.assertEqual(test, ref)

  def test_tagger_zh_src_augment_replace(self):
    '''Tests the tagger with zh on the source side'''
    random.seed(1)
    tagger = PlaceholderTagModifier(probability=0.6, custom_detok_src='zh', custom_detok_trg=None,
                                     augment=0.4, replace=0.4)
    with open('contrib/test-data/clean.zhen.10', 'r', encoding='utf-8') as myinput, \
         open('contrib/test-data/clean.zhen.ref.06.4.04.04.src', 'r', encoding='utf-8') as reference:
        for line in myinput:
          test = tagger(line)
          ref = reference.readline()[:-1]
          self.assertEqual(test, ref)

  def test_warn_if_tag_modifier_is_not_last(self):
    with self.assertWarnsRegex(UserWarning, r'Tags modifier should to be the last modifier to be applied'):
      loader = CurriculumLoader()
      loader.load(dedent("""
        datasets: {}
        stages: []
        seed: 1
        modifiers:
          - Tags: 1.0
          - UpperCase: 1.0
      """))
