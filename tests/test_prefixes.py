import random
import unittest
from opustrainer.modifiers.prefix import PrefixModifier


def first(it):
  return next(iter(it))


class TestPrefix(unittest.TestCase):
  '''Tests several test cases for prefixes'''
  def test_few(self):
    random.seed(1)
    modifier = PrefixModifier(1, 2, 6)
    # Test a line with some bogus alignments
    line1 = "How are you doing today good sir?\tКак я караш днеска добри ми господине?\t1-2 2-3"
    expected1 = "__start__ караш днеска __end__ How are you doing today good sir?\tКак я караш днеска добри ми господине?\t1-2 2-3"
    self.assertEqual(first(modifier([line1])), expected1)

    # Different execution different result
    expected2 = "__start__ я караш днеска добри ми __end__ How are you doing today good sir?\tКак я караш днеска добри ми господине?\t1-2 2-3"
    self.assertEqual(first(modifier([line1])), expected2)

    # Different execution different result. This time not triggered
    expected3 = "__start__ Как я караш __end__ How are you doing today good sir?\tКак я караш днеска добри ми господине?\t1-2 2-3"
    self.assertEqual(first(modifier([line1])), expected3)
    
    # Should return the same string as the word is too short. Also tests parsing without alignment.
    line2 = "Hello\tHola"
    self.assertEqual(first(modifier([line2])), line2)

