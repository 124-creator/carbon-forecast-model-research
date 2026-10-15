import importlib
import os
from pathlib import Path
import unittest

try:
    replay = importlib.import_module('replay_remote_scores')
except ModuleNotFoundError:
    replay = None


class ReplayTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('CARBON_REVIEW_ARTIFACT_ROOT'), 'Pinned numerical artifacts are local-only')
    def test_original_label_replay_checks_every_pinned_configuration(self):
        self.assertIsNotNone(replay, 'Independent scorer has not been implemented')
        result = replay.verify(Path(os.environ['CARBON_REVIEW_ARTIFACT_ROOT']), Path(__file__).resolve().parents[3])
        self.assertEqual(result['configurations'], 87)
        self.assertEqual(result['origins'], 451)
        self.assertLessEqual(result['maximum_metric_difference'], 1e-10)
        self.assertEqual(len(result['stages']), 3)
        self.assertTrue(all(not r['mechanism_pass'] and not r['accuracy_pass'] for r in result['stages'].values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
