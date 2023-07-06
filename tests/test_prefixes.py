import random
import unittest
from opustrainer.modifiers.prefix import PrefixModifier

class TestPrefix(unittest.TestCase):
  '''Tests several test cases for prefixes'''
  def test_few(self):
    random.seed(1)
    modifier = PrefixModifier(1, 2, 6)
    # Test a line with some bogus alignments
    line1: str = "How are you doing today good sir?\tКак я караш днеска добри ми господине?\t1-2 2-3"
    expected1: str = "__start__ караш днеска __end__ How are you doing today good sir?\tКак я караш днеска добри ми господине?\t1-2 2-3"
    output1: str = modifier(line1)
    self.assertEqual(output1, expected1)

    output2: str = modifier(line1) # Different execution different result
    expected2: str = "__start__ я караш днеска добри ми __end__ How are you doing today good sir?\tКак я караш днеска добри ми господине?\t1-2 2-3"
    self.assertEqual(output2, expected2)

    output3: str = modifier(line1) # Different execution different result. This time not triggered
    expected3: str = "__start__ Как я караш __end__ How are you doing today good sir?\tКак я караш днеска добри ми господине?\t1-2 2-3"
    self.assertEqual(output3, expected3)
    
    # Should return the same string as the word is too short. Also tests parsing without alignment.
    line2: str = "Hello\tHola"
    self.assertEqual(modifier(line2), line2)

