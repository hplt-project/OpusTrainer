import sys
import unittest
import subprocess


class TestEndToEnd(unittest.TestCase):
	'''Tests the pipeline end-to-end. Aimed to to test the parser.'''
	def test_full_enzh(self):
		output: str = subprocess.check_output([sys.executable, '-m', 'opustrainer', '-c', 'contrib/test_enzh_config.yml', '-d', '--sync'], encoding="utf-8")
		reference: str = ""
		with open('contrib/test-data/test_enzh_config.expected.out', 'r', encoding='utf-8') as reffile:
			reference: str = "".join(reffile.readlines())
		self.assertEqual(output, reference)

	def test_full_zhen(self):
		output: str = subprocess.check_output([sys.executable, '-m', 'opustrainer', '-c', 'contrib/test_zhen_config.yml', '-d', '--sync'], encoding="utf-8")
		reference: str = ""
		with open('contrib/test-data/test_zhen_config.expected.out', 'r', encoding='utf-8') as reffile:
			reference: str = "".join(reffile.readlines())
		self.assertEqual(output, reference)

	def test_prefix_augment(self):
		output: str = subprocess.check_output([sys.executable, '-m', 'opustrainer', '-c', 'contrib/test_zhen_prefix_config.yml', '-d', '--sync'], encoding="utf-8")
		reference: str = ""
		with open('contrib/test-data/test_zhen_config_prefix.expected.out', 'r', encoding='utf-8') as reffile:
			reference: str = "".join(reffile.readlines())
		self.assertEqual(output, reference)

	def test_no_shuffle(self):
		output: str = subprocess.check_output([sys.executable, '-m', 'opustrainer', '-c', 'contrib/test_enzh_config_plain.yml', '-d', '-n'], encoding="utf-8")
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
