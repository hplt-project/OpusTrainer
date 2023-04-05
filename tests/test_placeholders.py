import sys
import random
import unittest
from typing import Callable

from opustrainer.modifiers.placeholders import PlaceholderTagModifier


class TestTagger(unittest.TestCase):
	def setUp(self):
		random.seed(1)

	def test_mode_tagging(self):
		'''Test tagging mode, where it inserts a tag to translate to the already
		   expected output.
		'''		
		tagger = PlaceholderTagModifier(probability=.5)
		output = tagger('Hello world etcetera\tHallo Welt und so\t0-0 1-1 2-2 2-3')
		self.assertEqual(output, 'Hello <tag1> Hallo </tag1> world <tag4> Welt </tag4> etcetera\tHallo Welt und so\t0-0 4-1 8-2 8-3')
		#                         ^0    ^(1)   ^(2)  ^(3)    ^4    ^(5)   ^(6) ^(7)    ^8        ^0    ^1   ^2  ^3

	def test_mode_augment(self):
		'''Test tagger augmenting mode, where it will insert the same random noise
		   in logical places in both the input and the reference side.
		'''
		tagger = PlaceholderTagModifier(probability=.5, augment=1)
		output = tagger('Hello world etcetera\tHallo Welt und so\t0-0 1-1 2-2 2-3')
		self.assertEqual(output, 'Hello ټ؇ۤە٣ٮڛۃ world etcetera\tHallo ټ؇ۤە٣ٮڛۃ Welt und so\t0-0 1-1 2-2 3-3 3-4')
		#                         ^0    ^1       ^2    ^3        ^0    ^1       ^2   ^3  ^4

	def test_mode_replace(self):
		'''Test replace mode, which like tagging mode adds a tag to the input side
		   to guide the model to translate a certain word in a certain way, but
		   instead of guiding it to a logical translation, it guides it to noise.
		'''
		tagger = PlaceholderTagModifier(probability=.5, replace=1)
		output = tagger('Hello world etcetera\tHallo Welt und so\t0-0 1-1 2-2 2-3')
		self.assertEqual(output, 'Hello <tag1> ټ؇ۤە٣ٮڛۃ </tag1> world etcetera\tټ؇ۤە٣ٮڛۃ Welt und so\t0-0 4-1 5-2 5-3')
		#                         ^0    ^(1)   ^(2)     ^(3)    ^4    ^5        ^0       ^1   ^2  ^3

	def _run_modifier(self, modifier:Callable[[str],str], input_file:str, reference_file:str) -> None:
		'''Helper function to run a modifier through an external test file, and compare the output on an
		   external reference file.
		'''
		with open(input_file, 'r', encoding='utf-8') as fin, open(reference_file, 'r', encoding='utf-8') as fref:
			for line, ref in zip(fin, fref):
				self.assertEqual(modifier(line.rstrip()), ref.rstrip())

	# def _run_modifier(self, modifier:Callable[[str],str], input_file:str, reference_file:str) -> None:
	# 	'''Variant of run_modifier that populates the reference file :sunglasses:'''
	# 	with open(input_file, 'r', encoding='utf-8') as fin, open(reference_file, 'w', encoding='utf-8') as fref:
	# 		for line in fin:
	# 			print(modifier(line.rstrip()), file=fref)

	def test_tagger_zh_src(self):
		'''Tests the tagger with zh on the source side'''
		tagger = PlaceholderTagModifier(probability=0.6, num_tags=4, custom_detok_src='zh')
		self._run_modifier(tagger, 'contrib/test-data/clean.zhen.10', 'contrib/test-data/clean.zhen.ref.06.4.src')
	
	def test_tagger_zh_trg(self):
		'''Tests the tagger with zh on the target side'''
		tagger = PlaceholderTagModifier(probability=0.6, num_tags=4, custom_detok_src=None, custom_detok_trg='zh')
		self._run_modifier(tagger, 'contrib/test-data/clean.enzh.10', 'contrib/test-data/clean.enzh.ref.06.4.trg')

	def test_tagger_no_zh(self):
		'''Tests the tagger without zh detokenizer'''
		tagger = PlaceholderTagModifier(probability=0.6, num_tags=4)
		self._run_modifier(tagger, 'contrib/test-data/clean.enzh.10', 'contrib/test-data/clean.enzh.ref.06.4.none')

	def test_tagger_zh_src_augment_replace(self):
		'''Tests the tagger with zh on the source side'''
		tagger = PlaceholderTagModifier(probability=0.6, num_tags=4, custom_detok_src='zh', custom_detok_trg=None,
				  template=" <tag{n}> {token} </tag{n}>", augment=0.4, replace=0.4)
		self._run_modifier(tagger, 'contrib/test-data/clean.zhen.10', 'contrib/test-data/clean.zhen.ref.06.4.04.04.src')
