import random
import unittest


from opustrainer.modifiers.placeholders import PlaceholderTagModifier


class TestTagger(unittest.TestCase):
	def test_tagger_tagging(self):
		random.seed(1)
		tagger = PlaceholderTagModifier(probability=1)
		output = tagger('Hello world\tHallo Welt\t0-0 1-1')
		self.assertEqual(output, 'Hello <tag4> Hallo </tag4> world <tag1> Welt </tag1>\tHallo Welt')

	def test_tagger_augment(self):
		random.seed(1)
		tagger = PlaceholderTagModifier(probability=1, augment=1)
		output = tagger('Hello world\tHallo Welt\t0-0 1-1')
		self.assertEqual(output, '''Hello ゜ そ world ټ؇ۤە	Hallo ゜ そ Welt ټ؇ۤە''')

	def test_tagger_replace(self):
		random.seed(1)
		tagger = PlaceholderTagModifier(probability=.5, replace=1)
		output = tagger('Hello world\tHallo Welt\t0-0 1-1')
		self.assertEqual(output, '''Hello <tag4> ゜ そ </tag4> world <tag1> ټ؇ۤە </tag1>	゜ そ ټ؇ۤە''')

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
