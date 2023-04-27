import random
import unittest

from textwrap import dedent

from opustrainer.modifiers.placeholders import PlaceholderTagModifier, get_full_word
from opustrainer.trainer import CurriculumLoader

import sentencepiece as spm


class TestTagger(unittest.TestCase):
  def test_tagger_tagging(self):
    random.seed(1)
    tagger = PlaceholderTagModifier(probability=1)
    output = tagger('Hello world\tHallo Welt\t0-0 1-1')
    self.assertEqual(output, '__source__ Hello __target__ Hallo __done__ __source__ world __target__ Welt __done__\tHallo Welt')

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

  def test_spm_word_finder(self):
    intxt = '▁Sur g ical ▁light ing ▁systems ▁are ▁an ▁integral ▁part ▁of ▁the ▁operating ▁room ▁and ▁are ▁often ▁used ▁to ▁document ▁surgical ▁procedures . ▁We ▁focus ▁on ▁video ▁transmission ▁through ▁support ▁arm ▁light ▁slip ▁ring ▁solutions .'

    outtxt = [('Surgical', [0, 1, 2]),
            ('Surgical', [0, 1, 2]),
            ('Surgical', [0, 1, 2]),
            ('lighting', [3, 4]),
            ('lighting', [3, 4]),
            ('systems', [5]),
            ('are', [6]),
            ('an', [7]),
            ('integral', [8]),
            ('part', [9]),
            ('of', [10]),
            ('the', [11]),
            ('operating', [12]),
            ('room', [13]),
            ('and', [14]),
            ('are', [15]),
            ('often', [16]),
            ('used', [17]),
            ('to', [18]),
            ('document', [19]),
            ('surgical', [20]),
            ('procedures.', [21, 22]),
            ('procedures.', [21, 22]),
            ('We', [23]),
            ('focus', [24]),
            ('on', [25]),
            ('video', [26]),
            ('transmission', [27]),
            ('through', [28]),
            ('support', [29]),
            ('arm', [30]),
            ('light', [31]),
            ('slip', [32]),
            ('ring', [33]),
            ('solutions.', [34, 35]),
            ('solutions.', [34, 35])]
    intok = intxt.split()
    spmmodel = spm.SentencePieceProcessor(model_file='contrib/test-data/vocab.zhen.spm')
    for i in range(len(intok)):
        self.assertEqual(outtxt[i], get_full_word(intok, i, spmmodel))

  def test_tagger_zh_src_spm(self):
    '''Tests the tagger with zh on the target side'''
    random.seed(2)
    tagger = PlaceholderTagModifier(probability=0.6, custom_detok_src='zh', spm_vocab='contrib/test-data/vocab.zhen.spm', spm_run=True)
    line = """▁手术 ▁照明 ▁系统 ▁是 ▁手术 室 ▁的 ▁组成 ▁部分 ▁, ▁同时 ▁还 ▁常 ▁用来 ▁记录 ▁手术 ▁过程 ▁。 ▁我们 ▁专注 于 ▁通过 ▁支 <0xE6> <0x92> <0x91> ▁ <0xE8> <0x87> <0x82> ▁灯 ▁滑 环 ▁解决 ▁方案 ▁来 ▁传输 ▁视频\t▁Sur g ical ▁light ing ▁systems ▁are ▁an ▁integral ▁part ▁of ▁the ▁operating ▁room ▁and ▁are ▁often ▁used ▁to ▁document ▁surgical ▁procedures . ▁We ▁focus ▁on ▁video ▁transmission ▁through ▁support ▁arm ▁light ▁slip ▁ring ▁solutions .\t0-0 1-3 1-4 2-5 3-6 4-2 5-13 6-10 7-7 7-8 7-10 8-9 10-11 12-16 13-17 13-18 14-19 15-20 17-22 18-23 19-24 19-25 20-24 21-28 22-29 27-30 28-30 29-30 30-31 31-32 32-33 33-34 34-34 35-28 35-35 36-27 37-26"""
    print(tagger(line))
    self.assertEqual(line, line)
