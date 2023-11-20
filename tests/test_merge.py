from doctest import Example
import random
import unittest
from opustrainer.alignments import parse_alignments

from opustrainer.modifiers.merge import MergeModifier, merge_sents

class TestMerge(unittest.TestCase):
  def setUp(self):
    random.seed(1)

    # Set up examples
    self.example = [
      '429 运输 中队 ( 429 野牛) , 使用 CC - 177	429 Transport Squadron (429 Bison Squadron) - Flying the CC-177	0-0 1-1 2-2 3-3 4-3 5-4 5-5 7-5 8-5 9-6 8-7 9-8 10-9',
      "微生物 检验 与 食品 安全 控制 .	Food Poisoning and Food Hygiene.	3-0 0-1 1-1 2-1 2-2 3-3 4-3 5-4 6-4"
    ]*10

    # Remove the third column with alignments
    self.example_noalign = ["\t".join(a.split('\t')[:-1]) for a in self.example]

    # counts
    self.psn_cnt = " ".join(self.example).count('Poisoning') # 10
    self.num_cnt = " ".join(self.example).count('429') # 40 because it appears once in src and trg

  def test_merge_sents(self):
    merged = merge_sents(self.example_noalign[0:3])
    expected = '429 运输 中队 ( 429 野牛) , 使用 CC - 177 微生物 检验 与 食品 安全 控制 . 429 运输 中队 ( 429 野牛) , 使用 CC - 177\t429 Transport Squadron (429 Bison Squadron) - Flying the CC-177 Food Poisoning and Food Hygiene. 429 Transport Squadron (429 Bison Squadron) - Flying the CC-177'
    self.assertEqual(merged, expected)

    # Expected based on counts 
    lensrc = sum([len(a.split('\t')[0].split()) for a in self.example_noalign[0:3]])
    lentrg = sum([len(a.split('\t')[1].split()) for a in self.example_noalign[0:3]])

    lenmrgsrc = len(merged.split('\t')[0].split())
    lenmrgtrg = len(merged.split('\t')[1].split())
    self.assertEqual(lensrc, lenmrgsrc)
    self.assertEqual(lentrg, lenmrgtrg)

  def test_merge_sents_with_alignments(self):
    merged = merge_sents(self.example[0:3])
    expected = '429 运输 中队 ( 429 野牛) , 使用 CC - 177 微生物 检验 与 食品 安全 控制 . 429 运输 中队 ( 429 野牛) , 使用 CC - 177\t429 Transport Squadron (429 Bison Squadron) - Flying the CC-177 Food Poisoning and Food Hygiene. 429 Transport Squadron (429 Bison Squadron) - Flying the CC-177\t0-0 1-1 2-2 3-3 4-3 5-4 5-5 7-5 8-5 9-6 8-7 9-8 10-9 14-10 11-11 12-11 13-11 13-12 14-13 15-13 16-14 17-14 18-15 19-16 20-17 21-18 22-18 23-19 23-20 25-20 26-20 27-21 26-22 27-23 28-24'
    self.assertEqual(merged, expected)

    # Validate the alignments against the text itself (i.e. no out-of-bound tokens)
    src, trg, alignments = merged.split('\t')
    pairs = parse_alignments(alignments, src.split(), trg.split())

    # Assert all pairs are unique, and not just duplicates of the input
    self.assertEqual(len(pairs), len(set(pairs)))

    # Assert that all pairs are present
    example_pair_cnt = sum(
      len(parse_alignments(example.split('\t')[2]))
      for example in self.example[0:3]
    )
    self.assertEqual(example_pair_cnt, len(pairs))

  def test_merge_modifier_full(self):
    merger = MergeModifier(0.8)
    merged = list(merger(self.example_noalign))
    
    psn_cnt = " ".join(merged).count('Poisoning')
    num_cnt = " ".join(merged).count('429')

    self.assertNotEqual(len(merged), len(self.example_noalign)) # Assert it being activated
    self.assertEqual(self.psn_cnt, psn_cnt)
    self.assertEqual(self.num_cnt, num_cnt)

  def test_merge_modifier_full_with_alignments(self):
    merger = MergeModifier(0.8)
    merged = list(merger(self.example))
    
    psn_cnt = " ".join(merged).count('Poisoning')
    num_cnt = " ".join(merged).count('429')

    self.assertNotEqual(len(merged), len(self.example)) # Assert it being activated
    self.assertEqual(self.psn_cnt, psn_cnt)
    self.assertEqual(self.num_cnt, num_cnt)
  
 
