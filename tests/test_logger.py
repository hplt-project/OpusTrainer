import sys
import unittest
import subprocess
import tempfile

from typing import List

class TestLogger(unittest.TestCase):
    '''Tests the logger using.'''
    def test_file_and_stderr(self):
        with tempfile.NamedTemporaryFile(suffix='.log', prefix="opustrainer") as tmpfile:     
            process = subprocess.run([sys.executable, '-m', 'opustrainer', '-c', 'contrib/test_enzh_config_plain.yml', '-d', '-l', tmpfile.name], stdout = subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
            log = process.stderr
            # Check that the stderr output is the same the output written to the file
            with open(tmpfile.name, 'r', encoding='utf-8') as logfile:
                file_output: str = "".join(logfile.readlines())
            self.assertEqual(log, file_output)
            # Check if the log is the same as the reference log. This is more complicated as we have a time field
            with open('contrib/test-data/test_enzh_config_plain_expected.log', 'r', encoding='utf-8') as reffile:
                reference: List[str] = reffile.readlines()
                loglist: List[str] = log.split('\n')
                # Strip the time field and test
                for i in range(len(reference)):
                    ref = reference[i].split('[Trainer]')[1]
                    ref = ref.strip('\n')
                    logout = loglist[i].split('[Trainer]')[1]
                    self.assertEqual(logout, ref)
            
