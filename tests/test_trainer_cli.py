#!/usr/bin/env python3
import unittest
import sys
from subprocess import Popen
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

import yaml

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

	def test_early_stopping(self):
		"""Test letting the trainer move to the next stage using early-stopping"""
		head_lines = 10000

		basepath = Path('contrib').absolute()

		config = {
				'datasets': {
						'clean': str(basepath / 'test-data/clean'),
						'medium': str(basepath / 'test-data/medium'),
				},
				'stages': [
						'start',
						'mid',
				],
				'start': [
						'clean 1.0',
						'until clean inf'
				],
				'mid': [
						'medium 1.0',
						'until medium inf',
				],
				'seed': 1111
		}

		with TemporaryDirectory() as tmp, TemporaryFile() as fout, TemporaryFile() as ferr:
			with open(Path(tmp) / 'config.yml', 'w+t') as fcfg:
				yaml.safe_dump(config, fcfg)

			child = Popen([
				sys.executable,
				'-m', 'opustrainer',
				'--do-not-resume',
				'--no-shuffle',
				'--config', str(Path(tmp) / 'config.yml'),
				'head', '-n', str(head_lines)
			], stdout=fout, stderr=ferr)

			retval = child.wait(30)
			fout.seek(0)
			ferr.seek(0)

			# Assert we exited neatly
			self.assertEqual(retval, 0, msg=ferr.read().decode())

			# Assert we got the number of lines we'd expect
			line_count = sum(1 for _ in fout)
			self.assertEqual(line_count, len(config['stages']) * head_lines)
