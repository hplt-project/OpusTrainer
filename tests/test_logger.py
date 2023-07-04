import unittest
import tempfile

from opustrainer import logger

class TestLogger(unittest.TestCase):
    '''Tests the logger using end-to-end config.'''
    def test_log_once(self):
        '''Tests the log_once functionality'''
        with tempfile.NamedTemporaryFile(suffix='.log', prefix="logger") as tmpfile:
            logger.setup_logger(outputfilename=tmpfile.name, disable_stderr=True)
            logger.log("Test message")
            logger.log_once("Once message")
            logger.log_once("Once message")
            logger.log("Test message")
            logger.log_once("Once message")
            logger.log_once("Once message2")
            logger.log_once("Once message2")
            logger.log("Final message")
            logger.logging.shutdown()
            with open(tmpfile.name, 'r', encoding='utf-8') as reffile:
                line1 = reffile.readline().strip().split(' [Trainer] ')[1]
                self.assertEqual(line1,"[INFO] Test message")
                line2 = reffile.readline().strip().split(' [Trainer] ')[1]
                self.assertEqual(line2,"[INFO] Once message")
                line3 = reffile.readline().strip().split(' [Trainer] ')[1]
                self.assertEqual(line3,"[INFO] Test message")
                line4 = reffile.readline().strip().split(' [Trainer] ')[1]
                self.assertEqual(line4,"[INFO] Once message2")
                line5 = reffile.readline().strip().split(' [Trainer] ')[1]
                self.assertEqual(line5,"[INFO] Final message")
