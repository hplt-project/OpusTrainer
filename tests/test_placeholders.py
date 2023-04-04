import random
import unittest


from opustrainer.modifiers.placeholders import PlaceholderTagModifier


class TestTagger(unittest.TestCase):
	def test_tagger_tagging(self):
		random.seed(1)
		tagger = PlaceholderTagModifier(probability=.5)
		output = tagger('Hello world etcetera\tHallo Welt und so\t0-0 1-1 2-2 2-3')
		self.assertEqual(output, 'Hello <tag4> Hallo </tag4> world <tag1> Welt </tag1> etcetera\tHallo Welt und so\t0-0 4-1 8-2 8-3')
		#                         ^0    ^(1)   ^(2)  ^(3)    ^4    ^(5)   ^(6) ^(7)    ^8        ^0    ^1   ^2  ^3

	def test_tagger_augment(self):
		random.seed(1)
		tagger = PlaceholderTagModifier(probability=.5, augment=1)
		output = tagger('Hello world etcetera\tHallo Welt und so\t0-0 1-1 2-2 2-3')
		self.assertEqual(output, 'Hello world ټ؇ۤە٣ٮڛۃ etcetera\tHallo Welt ټ؇ۤە٣ٮڛۃ und so\t0-0 1-1 2-2 3-3 3-4')
		#                         ^0    ^1    ^2       ^3        ^0    ^1   ^2       ^3  ^4

	def test_tagger_replace(self):
		random.seed(1)
		tagger = PlaceholderTagModifier(probability=.5, replace=1)
		output = tagger('Hello world etcetera\tHallo Welt und so\t0-0 1-1 2-2 2-3')
		self.assertEqual(output, 'Hello world <tag1> ټ؇ۤە٣ٮڛۃ </tag1> etcetera\tHallo ټ؇ۤە٣ٮڛۃ und so\t0-0 1-1 5-2 5-3')
		#                         ^0    ^1    ^(2)   ^(3)     ^(4)    ^5        ^0    ^1       ^2  ^3

	def test_tagger_zh_src(self):
		'''Tests the tagger with zh on the source side'''
		random.seed(1)
		tagger = PlaceholderTagModifier(probability=0.6, num_tags=4, custom_detok_src='zh')
		with open('contrib/test-data/clean.zhen.10', 'r', encoding='utf-8') as myinput, \
		     open('contrib/test-data/clean.zhen.ref.06.4.src', 'r', encoding='utf-8') as reference:
				for line in myinput:
					test = tagger(line)
					ref = reference.readline()[:-1]
					self.assertEqual(test, ref)
	
	def test_tagger_zh_trg(self):
		'''Tests the tagger with zh on the target side'''
		random.seed(1)
		tagger = PlaceholderTagModifier(probability=0.6, num_tags=4, custom_detok_src=None, custom_detok_trg='zh')
		with open('contrib/test-data/clean.enzh.10', 'r', encoding='utf-8') as myinput, \
		     open('contrib/test-data/clean.enzh.ref.06.4.trg', 'r', encoding='utf-8') as reference:
				for line in myinput:
					test = tagger(line)
					ref = reference.readline()[:-1]
					self.assertEqual(test, ref)

	def test_tagger_no_zh(self):
		'''Tests the tagger without zh detokenizer'''
		random.seed(1)
		tagger = PlaceholderTagModifier(probability=0.6, num_tags=4)
		with open('contrib/test-data/clean.enzh.10', 'r', encoding='utf-8') as myinput, \
		     open('contrib/test-data/clean.enzh.ref.06.4.none', 'r', encoding='utf-8') as reference:
				for line in myinput:
					test = tagger(line)
					ref = reference.readline()[:-1]
					self.assertEqual(test, ref)

	def test_tagger_zh_src_augment_replace(self):
		'''Tests the tagger with zh on the source side'''
		random.seed(1)
		tagger = PlaceholderTagModifier(probability=0.6, num_tags=4, custom_detok_src='zh', custom_detok_trg=None,
				  template=" <tag{n}> {token} </tag{n}>", augment=0.4, replace=0.4)
		with open('contrib/test-data/clean.zhen.10', 'r', encoding='utf-8') as myinput, \
		     open('contrib/test-data/clean.zhen.ref.06.4.04.04.src', 'r', encoding='utf-8') as reference:
				for line in myinput:
					test = tagger(line)
					ref = reference.readline()[:-1]
					self.assertEqual(test, ref)
