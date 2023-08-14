import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List


class TestEndToEnd(unittest.TestCase):
    '''Tests the pipeline end-to-end. Aimed to to test consistent output.'''

    def __clean_states(self):
        """Remove state files"""
        basepath = Path("contrib")
        for state in basepath.glob("*.yml.state"):
            state.unlink(missing_ok=True)

    def setUp(self) -> None:
        self.__clean_states()

    def tearDown(self) -> None:
        self.__clean_states()

    def assertEndToEnd(self, args: List[str], path:str, *, returncode:int=0):
        process = subprocess.run(
            [sys.executable, '-m', 'opustrainer', *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8")

        self.assertEqual(process.returncode, returncode, msg=process.stderr)

        # Useful to generate the data, hehe
        if False and path.endswith('.expected.out'):
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(process.stdout)

        with open(path, 'r', encoding='utf-8') as fh:
            reference = fh.read()

        self.assertEqual(process.stdout, reference)

    def test_full_enzh(self):
        self.assertEndToEnd(
            ['-c', 'contrib/test_enzh_config.yml', '-d', '--sync'],
            'contrib/test-data/test_enzh_config.expected.out')

    def test_full_zhen(self):
        self.assertEndToEnd(
            ['-c', 'contrib/test_zhen_config.yml', '-d', '--sync'],
            'contrib/test-data/test_zhen_config.expected.out')

    def test_prefix_augment(self):
        self.assertEndToEnd(
            ['-c', 'contrib/test_zhen_prefix_config.yml', '-d', '--sync'],
            'contrib/test-data/test_zhen_config_prefix.expected.out')

    def test_no_shuffle(self):
        """Confirms that when an empty training procedure is used with
        the `--no-shuffle` option, it reproduces its input."""
        for mode in [[], ['--sync']]:
            with self.subTest(mode=mode):
                self.assertEndToEnd(
                    [
                        '-c', 'contrib/test_enzh_config_plain.yml',
                        '-d',
                        *mode,
                        '--no-shuffle',
                        '--batch-size', '1' # batch-size << dataset size otherwise we overproduce
                    ],
                    'contrib/test-data/clean.enzh.10')

    def test_advanced_config(self):
        self.assertEndToEnd(
            ['-c', 'contrib/test_enzh_tags_advanced_config.yml', '-d', '-n'],
            'contrib/test-data/test_enzh_tags_advanced_config.expected.out')

    def test_stage_config(self):
        self.assertEndToEnd(
            ['-c', 'contrib/test_enzh_tags_stage_config.yml', '-d', '-n'],
            'contrib/test-data/test_enzh_tags_stage_config.expected.out')

    def test_log_file_and_stderr(self):
        """Test that log messages go to stdout and to the logfile. Note that the
        log file contains 10 iterations because the dataset is only 10 lines but
        the batch size is 100. Termination conditions are only checked at the
        end of the batch. TODO: this might be a bug, or at least unexpected."""
        with tempfile.NamedTemporaryFile(suffix='.log', prefix="opustrainer", mode='w+', encoding="utf-8") as tmpfile:
            process = subprocess.run(
                [sys.executable, '-m', 'opustrainer', '-c', 'contrib/test_enzh_config_plain.yml', '-d', '-l', tmpfile.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                encoding="utf-8")
            self.assertEqual(process.returncode, 0)
            
            # Check that the stderr output is the same the output written to the file
            tmpfile.seek(0)
            self.assertEqual(process.stderr, tmpfile.read())
            
            # Check if the log is the same as the reference log. This is more complicated as we have a time field
            with open('contrib/test-data/test_enzh_config_plain_expected.log', 'r', encoding='utf-8') as reflog:
                # Loglist has one extra `\n` compared to reference list, due to stderr flushing an extra empty line?
                loglist = process.stderr.splitlines(keepends=True)
                reference = reflog.readlines()
                # Strip the time field and test
                remove_timestamp = lambda line: line.split('[Trainer]', maxsplit=1)[1]
                self.assertEqual(
                    [remove_timestamp(line) for line in loglist],
                    [remove_timestamp(line) for line in reference])
