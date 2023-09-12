#!/usr/bin/env python3
'''Tests the available functionality'''
import os
import tempfile
import unittest

from typing import IO, Type
from collections import Counter
from contextlib import closing
from textwrap import dedent
from io import StringIO
from itertools import chain

import yaml

from opustrainer.trainer import Curriculum, CurriculumLoaderError, Dataset, DatasetReader, AsyncDatasetReader, CurriculumLoader, Trainer, StateTracker, Stage
from opustrainer.logger import log_once

TEST_FILE: str

def setUpModule():
	global TEST_FILE
	fd, TEST_FILE = tempfile.mkstemp(text=True)
	
	with open(fd, 'w') as fh:
		for n in range(1000):
			fh.write(f'line{n}\n')


def tearDownModule():
	os.unlink(TEST_FILE)


class TestDatasetReader(unittest.TestCase):
	testset: IO[str]

	reader: Type[DatasetReader] = DatasetReader

	def test_repeating_read(self):
		"""Test whether when we read 3000 lines from a 1000 lines dataset we do
		inded read each line 3 times.
		"""
		# Read 3000 lines
		with closing(self.reader(Dataset('test', [TEST_FILE]), seed=1234)) as reader:
			counter = Counter(line for _, line in zip(range(3000), reader))

		# We should have 1000 unique lines (all lines in TEST_FILE)
		self.assertEqual(len(counter), 1000)
		# And each line should be read exactly 3 times
		self.assertEqual(set(counter[key] for key in counter.keys()), {3})
		# ideally in a different order than previous read.

	def test_shuffled_read(self):
		"""That that when we read 2000 lines from a 1000 line testfile, we read
		each line twice, but we do read them in a different order.
		"""
		# Read 3000 lines
		with closing(self.reader(Dataset('test', [TEST_FILE]), seed=1234)) as reader:
			lines1 = [line for _, line in zip(range(1000), reader)]
			lines2 = [line for _, line in zip(range(1000), reader)]
		#	We should have read the same lines
		self.assertEqual(frozenset(lines1), frozenset(lines2))
		# but in a different order
		self.assertNotEqual(lines1, lines2)

	def test_offsets(self):
		"""Test whether `epoch` and `line` properties of a DatasetReader are
		counting properly.
		"""
		with closing(self.reader(Dataset('test', [TEST_FILE]), seed=1234)) as reader:
			for _ in zip(range(500), reader):
				pass

			self.assertEqual(reader.epoch, 0)
			self.assertEqual(reader.line, 500)

			for _ in zip(range(1250), reader):
				pass

			self.assertEqual(reader.epoch, 1)
			self.assertEqual(reader.line, 750)

	def test_resume_offset(self):
		"""Test whether resuming from a DatasetReader with the same testfile and the
		same seed does indeed yield the entire dataset with no duplicates or
		omissions.
		"""
		counter = Counter()

		with closing(self.reader(Dataset('test', [TEST_FILE]), seed=1234)) as reader1:
			counter.update(line for _, line in zip(range(250), reader1))
			state = reader1.state()

		with closing(self.reader(Dataset('test', [TEST_FILE]), seed=1234)) as reader2:
			reader2.restore(state)
			counter.update(line for _, line in zip(range(750), reader2))

		# We should have read 250 + 750 lines exactly
		self.assertEqual(len(counter), 1000)
		# and they should be all read only once.
		self.assertEqual(set(counter[key] for key in counter.keys()), {1})

	def test_resume_order(self):
		"""Test whether when we resume reading a DatasetReader, the order is the
		same as if we'd read everything from the initial (non-resumed) reader.
		"""
		counter = Counter()

		with closing(self.reader(Dataset('test', [TEST_FILE]), seed=1234)) as reader1, \
			closing(self.reader(Dataset('test', [TEST_FILE]), seed=1234)) as reader2:
			lines1 = [line for _, line in zip(range(250), reader1)]
			reader2.restore(reader1.state())
			lines1.extend(line for _, line in zip(range(750), reader2))

		with closing(self.reader(Dataset('test', [TEST_FILE]), seed=1234)) as reader1, \
			closing(self.reader(Dataset('test', [TEST_FILE]), seed=1234)) as reader2:
			lines2 = [line for _, line in zip(range(500), reader1)]
			reader2.restore(reader1.state())
			lines2.extend(line for _, line in zip(range(500), reader2))

		# Both reads should have the same amount and unique sentences
		self.assertEqual(len(lines1), len(lines2))
		self.assertEqual(set(lines1), set(lines2))
		# They also should have the same order
		self.assertEqual(lines1, lines2)


class TestAsyncDatasetReader(TestDatasetReader):
	"""Run all the same tests, but on the async reader that shuffles in advance."""
	reader = AsyncDatasetReader


class TestTrainer(unittest.TestCase):
	def test_resume(self):
		"""End-to-end test for resuming training where we test that a resumed
		trainer continues with the same sentences in the same order as the original
		trainer would have, if it were to continue after dumping its state.
		"""
		config = {
			'datasets': {
				'clean': 'contrib/test-data/clean',
				'medium': 'contrib/test-data/medium',
				'dirty': 'contrib/test-data/dirty'
			},
			'stages': [
				'start',
				'mid'
			],
			'start': [
				'clean 0.8',
				'medium 0.2',
				'dirty 0',
				'until clean 1'
			],
			'mid': [
				'clean 0.6',
				'medium 0.3',
				'dirty 0.1',
				'until medium 1',
			],
			'seed': 1111
		}
		
		curriculum = CurriculumLoader().load(config)

		# Reference batches (trainer runs without resuming)
		with closing(Trainer(curriculum)) as trainer_ref:
			batches_ref = list(trainer_ref.run())

		# State tracker (using tmpdir to make sure the file does not exist)
		with tempfile.TemporaryDirectory() as tmpdir:
			state_tracker = StateTracker(os.path.join(tmpdir, 'state_file'))

			# Train on trainer1
			with closing(Trainer(curriculum)) as trainer1:
				batches = [batch for _, batch in zip(range(10), state_tracker.run(trainer1))]

			# Resume on trainer2
			with closing(Trainer(curriculum)) as trainer2:
				batches.extend(state_tracker.run(trainer2))
			
		self.assertEqual(batches, batches_ref)

	def test_deterministic_parallel(self):
		"""End-to-end test to confirm that training with 2 workers or with 4 workers
		should yield the same training data going to the trainer.
		"""
		config = {
			'datasets': {
				'clean': 'contrib/test-data/clean',
				'medium': 'contrib/test-data/medium',
				'dirty': 'contrib/test-data/dirty'
			},
			'stages': [
				'start',
				'mid'
			],
			'start': [
				'clean 0.8',
				'medium 0.2',
				'dirty 0',
				'until clean 1'
			],
			'mid': [
				'clean 0.6',
				'medium 0.3',
				'dirty 0.1',
				'until medium 1',
			],
			'modifiers': [
				{'UpperCase': 0.25}
			],
			'seed': 1111
		}
		
		curriculum = CurriculumLoader().load(config)

		# Run with one worker and default chunk size
		with closing(Trainer(curriculum)) as trainer:
			batches_linear = list(trainer.run(processes=1))

		# Run with three workers, and default batch size
		with closing(Trainer(curriculum)) as trainer:
			batches_parallel = list(trainer.run(processes=4))

		self.assertEqual(batches_linear, batches_parallel)


class TestCurriculumLoader(unittest.TestCase):
	def test_simple(self):
		"""Test loading of a minimal configuration"""
		config = {
			'datasets': {
				'clean': 'contrib/test-data/clean',
			},
			'stages': [
				'start'
			],
			'start': [
				'clean 1.0',
				'until clean 5'
			],
			'seed': 1111,
			'modifiers': [
				{'UpperCase': 1.0}
			]
		}
		curriculum = CurriculumLoader().load(config)
		self.assertIsInstance(curriculum, Curriculum)
		self.assertEqual(curriculum.datasets, {
			'clean': Dataset(name='clean', files=['./contrib/test-data/clean'])
		})
		self.assertEqual(curriculum.stages, {
			'start': Stage(
				name='start',
				datasets=[
					(Dataset('clean', ['./contrib/test-data/clean']), 1.0)
				],
				until_dataset='clean',
				until_epoch=5,
				modifiers=None)
		})
		self.assertEqual(curriculum.seed, 1111)
		self.assertEqual(len(curriculum.modifiers), 1)

	def test_no_until(self):
		"""Test that omitting the until clause raises an error"""
		config = {
			'datasets': {
				'clean': 'contrib/test-data/clean',
			},
			'stages': [
				'start'
			],
			'start': [
				'clean 1.0'
			],
			'seed': 1
		}
		with self.assertRaisesRegex(CurriculumLoaderError, 'until clause'):
			CurriculumLoader().load(config)
	
	def test_undefined_dataset(self):
		"""Test that mentioning a wrong dataset raises an error"""
		config = {
			'datasets': {
				'clean': 'contrib/test-data/clean',
			},
			'stages': [
				'start'
			],
			'start': [
				'cleany 1.0',
				'until cleany 1'
			],
			'seed': 1
		}
		with self.assertRaisesRegex(CurriculumLoaderError, 'unknown dataset'):
			CurriculumLoader().load(config)

	def test_undefined_modifier(self):
		"""Test that mentioning an unknown modifier raises an error"""
		config = {
			'datasets': {
				'clean': 'contrib/test-data/clean',
			},
			'stages': [
				'start'
			],
			'start': [
				'clean 1.0',
				'until clean 1'
			],
			'modifiers': [
				{'NonExistingModifier': 1.0}
			],
			'seed': 1
		}
		with self.assertRaisesRegex(CurriculumLoaderError, 'unknown modifier'):
			CurriculumLoader().load(config)

	def test_extended_stage_configuration(self):
		"""Test the extended stage configuration"""
		config = {
			'datasets': {
				'clean': 'contrib/test-data/clean',
			},
			'stages': [
				'start'
			],
			'start': {
				'mix': [
					'clean 1.0',
					'until clean 1'
				],
				'modifiers': [
					{'UpperCase': 1.0}
				],
			},
			'seed': 1
		}
		curriculum = CurriculumLoader().load(config)
		self.assertEqual(len(curriculum.stages['start'].datasets), 1)
		self.assertEqual(curriculum.stages['start'].until_dataset, 'clean')
		self.assertTrue(curriculum.stages['start'].modifiers is not None and len(curriculum.stages['start'].modifiers) == 1)

	def test_combined_stage_configuration(self):
		"""Test that in the extended stage configuration modifier lists are
		flattened which allows us to use YAML references more easily."""

		# Writing this in YAML directly to reflect how this is intended to be useful
		config_yaml = dedent("""
			datasets:
				clean: contrib/test-data/clean

			stages:
			- start

			modifiers: &global_modifiers
			- TitleCase: 0.5
			
			start:
				mix: 
				- clean 1.0
				- until clean 1
				modifiers:
				- UpperCase: 0.5
				- *global_modifiers
			
			seed: 1
		""").replace("\t", "  ")

		config = yaml.safe_load(StringIO(config_yaml))

		curriculum = CurriculumLoader().load(config)
		self.assertEqual([modifier.__class__.__name__ for modifier in curriculum.stages['start'].modifiers or []], ['UpperCaseModifier', 'TitleCaseModifier'])

	def test_modifier_error_line_context(self):
		"""Test that when a modifier fails, we get context information about the line that failed"""
		with tempfile.NamedTemporaryFile('w', encoding='utf-8') as fd:
			fd.write('This is a test\tDas ist ein Test\t0-0 1-1 2-2 3-3\n')
			fd.write('Hello world\tHallo Welt\t0-0 1-2\n') # 2 is out-of-bounds
			fd.flush()

			config = {
				'datasets': {
					'clean': fd.name,
				},
				'stages': [
					'start'
				],
				'start': [
					'clean 1.0',
					'until clean 1'
				],
				'modifiers': [
					{'Tags': 1.0}
				],
				'seed': 1
			}
			curriculum = CurriculumLoader().load(config)

			trainer = Trainer(curriculum)
			
			with self.assertLogs(level='WARNING') as logger_ctx:
				output = list(chain.from_iterable(trainer.run(batch_size=1)))
				# Assert we skipped the line
				self.assertEqual(len(output), 1)
				# Assert that we got the general error message
				self.assertRegex(logger_ctx.output[0], r'Skipping line because of exception:')
				# Assert that we got the specific error as well
				self.assertRegex(logger_ctx.output[0], r'ValueError\(\'Out-of-bound alignment pairs\'\)')

	def test_num_fields(self):
		"""Tests the num field limiter"""
		with tempfile.NamedTemporaryFile('w', encoding='utf-8') as fd:
			fd.write('This is a test\tDas ist ein Test\t0-0 1-1 2-2 3-3\n')
			fd.write('This is a test\tDas ist ein Test\t0-0 1-1 2-2 3-3\textra fields\n')
			fd.write('Hello world\tHallo Welt\n') # Not enough fields
			fd.flush()

			config = {
				'datasets': {
					'clean': fd.name,
				},
				'stages': [
					'start'
				],
				'start': [
					'clean 1.0',
					'until clean 1'
				],
				'seed': 1,
				'num_fields': 3
			}
			curriculum = CurriculumLoader().load(config)

			for reader in [DatasetReader, AsyncDatasetReader]:
				with self.subTest(reader=reader.__name__), \
				 self.assertLogs(level='WARNING') as logger_ctx, \
				 closing(Trainer(curriculum, reader=reader)) as trainer:
					# Reset the log_once cache
					log_once.cache_clear()

					output = list(chain.from_iterable(trainer.run(batch_size=1)))
					# Assert we skipped the line
					self.assertEqual(len(output), 2)
					# Assert we properly cropped the lines and left the others unchanged
					self.assertEqual(output, [
						'This is a test\tDas ist ein Test\t0-0 1-1 2-2 3-3\n',
						'This is a test\tDas ist ein Test\t0-0 1-1 2-2 3-3\n'
					])
					# Assert that we got an error message for one line
					self.assertRegex(logger_ctx.output[0], r'\[Trainer\] Expected 3 fields in clean line:')
