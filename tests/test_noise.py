from doctest import Example
import enum
import random
import unittest

from opustrainer.modifiers.noise import NoiseModifier

class TestMerge(unittest.TestCase):
  def setUp(self):
    random.seed(1)

    # Set up examples
    self.example = [
      '429 运输 中队 ( 429 野牛) , 使用 CC - 177	429 Transport Squadron (429 Bison Squadron) - Flying the CC-177	0-0 1-1 2-2 3-3 4-3 5-4 5-5 7-5 8-5 9-6 8-7 9-8 10-9',
      "微生物 检验 与 食品 安全 控制 .	Food Poisoning and Food Hygiene.	3-0 0-1 1-1 2-1 2-2 3-3 4-3 5-4 6-4"
    ]*10

    # Remove column with alignment info
    self.example_noalign = ["\t".join(a.split('\t')[:-1]) for a in self.example]

    # With 20% prob this is triggered 3 times we check one of the matches. We expect new length to be 23
    self.num_nine_noise = "쑥맜\t쑥맜"
    self.num_nine_noise_align = "쑥맜\t쑥맜\t0-0"

  def test_noise(self):
    noiser = NoiseModifier(0.2)
    noised = list(noiser(self.example_noalign))
    self.assertEqual(noised[9], self.num_nine_noise)
    self.assertEqual(len(noised), 23)
    
  def test_noise_align(self):
    noiser = NoiseModifier(0.2)
    noised = list(noiser(self.example))
    self.assertEqual(noised[9], self.num_nine_noise_align)
    self.assertEqual(len(noised), 23)
