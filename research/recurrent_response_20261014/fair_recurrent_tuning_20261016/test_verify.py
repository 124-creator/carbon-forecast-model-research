import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from fair_data import load_prefix
from verify_fair import independent_scores, require_complete

class VerifyTest(unittest.TestCase):
    def test_incomplete_run_never_passes(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as name:
            directory = Path(name)
            with self.assertRaises(ValueError):
                require_complete(directory)
            (directory / 'RUN_COMPLETE.json').write_text(json.dumps({'complete': False}), encoding='utf-8')
            with self.assertRaises(ValueError):
                require_complete(directory)
            (directory / 'RUN_COMPLETE.json').write_text(json.dumps({'complete': True}), encoding='utf-8')
            self.assertTrue(require_complete(directory)['complete'])
            (directory / 'INCOMPLETE.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                require_complete(directory)

    def test_original_price_scores_separate_seed_losses(self):
        price, dates, _ = load_prefix()
        origins = np.arange(459, 930)
        truth = np.stack([price[t + 1:t + 21] - price[t] for t in origins])
        predictions = np.stack((truth + 1, truth - 1))
        result = independent_scores(price, dates, origins, predictions)
        self.assertGreater(result['score'], 0)
        self.assertLess(result['ensemble'], 1e-26)
        self.assertEqual(result['counts'], {'2020': 224, '2021': 227})
        predictions[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            independent_scores(price, dates, origins, predictions)

if __name__ == '__main__':
    unittest.main()
