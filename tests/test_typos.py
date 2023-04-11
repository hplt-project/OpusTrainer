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


class TestTypos(unittest.TestCase):
	def test_typos(self):
		"""Test modifier only alters column 0"""
		random.seed(1)
		modifier = TypoModifier(1.0)
		output = [modifier("\t".join(line)).split("\t") for line in INPUT_DATA]
		# Assert first column is modified
		self.assertEqual([row[0] for row in output], [
			'This s a sentence.',
			'Why are we doiing this?',
			'88T6683648',
			'---'
		])
		# Assert second column is untouched.
		self.assertEqual([row[1] for row in output], [row[1] for row in INPUT_DATA])

	def test_typos_specific(self):
		"""Test each modifier is applied only once per sentence"""
		random.seed(1)
		modifier = TypoModifier(1.0, extra_char=1.0, random_space=1.0)
		output = [modifier("\t".join(line)).split("\t") for line in INPUT_DATA]
		self.assertEqual([row[0] for row in output], [
			'Tyhis  is a sentence.',
			'W hy are we doing thjis?',
			'8 8756683648',
			'-- -'
		])

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
			'cl ewan531\n',
			'cole an739\n',
			'cl esan742\n',
			'clean2 580\n',
			'c lsean741\n'
		])
