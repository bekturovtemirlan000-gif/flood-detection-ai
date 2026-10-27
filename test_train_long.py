"""Regression coverage for transient Windows checkpoint locks."""
import unittest
from unittest.mock import Mock, patch
from train_long import replace_with_retry


class AtomicWriteTests(unittest.TestCase):
    @patch('train_long.time.sleep')
    def test_transient_permission_error_is_retried(self, sleep):
        source = Mock()
        source.replace.side_effect = [PermissionError('locked'), None]
        destination = Mock()
        replace_with_retry(source, destination)
        self.assertEqual(source.replace.call_count, 2)
        sleep.assert_called_once_with(.05)

    @patch('train_long.time.sleep')
    def test_persistent_permission_error_is_not_hidden(self, sleep):
        source = Mock()
        source.replace.side_effect = PermissionError('locked')
        with self.assertRaises(PermissionError):
            replace_with_retry(source, Mock())
        self.assertEqual(source.replace.call_count, 9)
        self.assertEqual(sleep.call_count, 8)

    def test_other_io_errors_propagate(self):
        source = Mock()
        source.replace.side_effect = OSError('disk failure')
        with self.assertRaises(OSError):
            replace_with_retry(source, Mock())
        source.replace.assert_called_once()


if __name__ == '__main__':
    unittest.main()
