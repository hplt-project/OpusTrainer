import sys
import unittest
import subprocess


class TestEndToEnd(unittest.TestCase):
	'''Tests the pipeline end-to-end. Aimed to to test the parser.'''
	def test_full_enzh(self):
		process = subprocess.run([sys.executable, '-m', 'opustrainer', '-c', 'contrib/test_enzh_config.yml', '-d', '--sync'], stdout = subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
		output = process.stdout
		reference: str = ""
		with open('contrib/test-data/test_enzh_config.expected.out', 'r', encoding='utf-8') as reffile:
			reference: str = "".join(reffile.readlines())
		self.assertEqual(output, reference)

	def test_full_zhen(self):
		process = subprocess.run([sys.executable, '-m', 'opustrainer', '-c', 'contrib/test_zhen_config.yml', '-d', '--sync'], stdout = subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
		output = process.stdout
		reference: str = ""
		with open('contrib/test-data/test_zhen_config.expected.out', 'r', encoding='utf-8') as reffile:
			reference: str = "".join(reffile.readlines())
		self.assertEqual(output, reference)

	def test_prefix_augment(self):
		process = subprocess.run([sys.executable, '-m', 'opustrainer', '-c', 'contrib/test_zhen_prefix_config.yml', '-d', '--sync'], stdout = subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
		output = process.stdout
		reference: str = ""
		with open('contrib/test-data/test_zhen_config_prefix.expected.out', 'r', encoding='utf-8') as reffile:
			reference: str = "".join(reffile.readlines())
		self.assertEqual(output, reference)

	def test_no_shuffle(self):
		process = subprocess.run([sys.executable, '-m', 'opustrainer', '-c', 'contrib/test_enzh_config_plain.yml', '-d', '-n'], stdout = subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
		output = process.stdout
		reference: str = ""
		with open('contrib/test-data/clean.enzh.10', 'r', encoding='utf-8') as reffile:
			reference: str = "".join(reffile.readlines())
		# Since we read 100 lines at a time, we wrap. Often.
		# Hence, for the test to pass we need to read the number of lines in the test file
		reference_arr = reference.split('\n')
		output_arr = output.split('\n')
		for i in range(len(reference_arr)):
			# Skip final empty newline
			if reference_arr[i] != '':
				self.assertEqual(output_arr[i], reference_arr[i])

	def test_advanced_config(self):
		process = subprocess.run([sys.executable, '-m', 'opustrainer', '-c', 'contrib/test_enzh_tags_advanced_config.yml', '-d', '-n'], stdout = subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
		output = process.stdout
		reference: str = ""
		with open('contrib/test-data/test_enzh_tags_advanced_config.expected.out', 'r', encoding='utf-8') as reffile:
			reference: str = "".join(reffile.readlines())
		self.assertEqual(output, reference)

	def test_stage_config(self):
		process = subprocess.run([sys.executable, '-m', 'opustrainer', '-c', 'contrib/test_enzh_tags_stage_config.yml', '-d', '-n'], stdout = subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
		output = process.stdout
		reference: str = ""
		with open('contrib/test-data/test_enzh_tags_stage_config.expected.out', 'r', encoding='utf-8') as reffile:
			reference: str = "".join(reffile.readlines())
		self.assertEqual(output, reference)
