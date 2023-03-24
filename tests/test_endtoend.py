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
