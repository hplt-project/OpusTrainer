#!/usr/bin/env python3
import unittest
from opustrainer.trainer import parse_args


class TestArgumentParser(unittest.TestCase):
	def test_overlapping_marian_args(self):
		"""Assert that you can pass arguments that exist both in trainer and marian
		to each respectively."""
		parsed = parse_args(['--config', 'trainer.yml', 'marian', '--config', 'marian.yml'])
		expected = {
			'config': 'trainer.yml',
			'trainer': ['marian', '--config', 'marian.yml']
		}
		# The `{**vars(parsed), **expected}` bit makes it so any entries in `parsed`
		# that are not mentioned in `expected` are not tested. Once upon a time
		# `assertDictContainsSubset` existed, but that's deprecated now and can't do
		# any `|` or `<=` operators here because we're targeting Python 3.8 :sad:
		self.assertEqual({**vars(parsed), **expected}, vars(parsed))

	def test_marian_log_args(self):
		"""Assert that you can pass `--log` to trainer even when argparse will find
		this confusing."""
		parsed = parse_args(['--config', 'trainer.yml', '--log-file', 'trainer.log', 'marian', '--log', 'marian.log'])
		expected = {
			'log_file': 'trainer.log',
			'config': 'trainer.yml',
			'trainer': ['marian', '--log', 'marian.log']
		}
		self.assertEqual({**vars(parsed), **expected}, vars(parsed))
