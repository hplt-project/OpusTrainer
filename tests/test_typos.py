import random
import unittest
from contextlib import closing

from opustrainer.trainer import Trainer, CurriculumLoader
from opustrainer.modifiers.typos import TypoModifier


INPUT_DATA = [
	("This is a sentence.", "Dit is een zin."),
	("Why are we doing this?", "Waarom zijn we dit aan het doen?"),
	("8876683648", "8876683648"),
	("---", "---"),
]


def first(it):
  return next(iter(it))


class TestTypos(unittest.TestCase):
	def test_typos(self):
		"""Test modifier only alters column 0"""
		random.seed(1)
		modifier = TypoModifier(1.0)
		output = [line.split('\t') for line in modifier(["\t".join(line) for line in INPUT_DATA])]
		# Assert first column is modified
		self.assertEqual([row[0] for row in output],[
			'Thisis a sentence.',
			'Why are we doing this?',
			'88766863648',
			'---'])
		# Assert second column is untouched.
		self.assertEqual([row[1] for row in output], [row[1] for row in INPUT_DATA])

	def test_typos_specific(self):
		"""Test each modifier is applied only once per sentence"""
		random.seed(1)
		modifier = TypoModifier(1.0, extra_char=1.0, random_space=1.0)
		output = [line.split('\t') for line in modifier(["\t".join(line) for line in INPUT_DATA])]
		self.assertEqual([row[0] for row in output], [
			'This is  a senftence.',
			'Why ad re we doing this?',
			'887 66836478',
			'-- -'])

	def test_trainer_with_typos(self):
		"""Test Typos configuration being picked up by the trainer"""
		config = {
			'datasets': {
				'clean': 'contrib/test-data/clean'
			},
			'stages': [
				'start'
			],
			'start': [
				'clean 1.0',
				'until clean 1',
			],
			'modifiers': [
				{
					'Typos': 1.0,
					'extra_char': 1.0,
					'random_space': 1.0,
				}
			],
			'seed': 1
		}
		
		curriculum = CurriculumLoader().load(config)

		# Reference batches (trainer runs without resuming)
		with closing(Trainer(curriculum)) as trainer:
			batches = list(trainer.run())


		self.assertEqual(batches[0][:5], [
			' cklean700\n',
			'c klean688\n',
			'cles an220\n',
			'clesan 281\n',
			'c lean17109\n']
		)

	def test_regression_40(self):
		random.seed(1)
		disable_all = {
			name: 0.0
			for name in TypoModifier.modifiers.keys()
		}
		line = '٦\t6'
		modifier = TypoModifier(1.0, **{**disable_all, 'extra_char':1.0})
		modified = next(iter(modifier([line])))
		self.assertNotEqual(modified, line) # test it changed
		self.assertRegex(modified, r'^..\t.$') # test it has 1 extra char

	def test_zero_prob(self):
		"""Test probability parameter (if 0, no typos)"""
		random.seed(1)
		modifier = TypoModifier(0.0, extra_char=1.0, random_space=1.0)
		output = [line.split('\t') for line in modifier(["\t".join(line) for line in INPUT_DATA])]
		self.assertEqual([row[0] for row in output], [row[0] for row in INPUT_DATA])
