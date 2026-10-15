import unittest
import csv
import io
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from verify_events import score_predictions, verdict, replay, read_original_rows


class ScoreTests(unittest.TestCase):
    def test_original_reader_never_requests_a_suffix(self):
        source = Path(__file__).resolve().parents[3] / 'data/parsed/figshare_28682348/guangzhou_reported.csv'
        with source.open('rb', buffering=0) as stream:
            approved = stream.read(29204)
        class Guard(io.BytesIO):
            def read(self, size: int | None = -1):
                if size != 29204:
                    raise AssertionError('Original reader requested unbounded or suffix bytes')
                return super().read(size)
        guarded = Guard(approved + b'FORBIDDEN_SUFFIX_NOT_REAL_DATA')
        with patch('verify_events.Path.open', return_value=guarded):
            rows = read_original_rows(source, 950)
        self.assertEqual(len(rows), 950)
        self.assertEqual(rows[-1]['date'], '2021-12-31')

    def test_missing_trajectories_are_reported_not_imputed(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            folder = Path(tmp)
            source = folder / 'SYNTHETIC_UNIT_FIXTURE.csv'
            dates = np.concatenate([np.datetime64(f'{year}-01-01') + np.arange(60) for year in (2018, 2020, 2021)])
            with source.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['date', 'author_reported_price'])
                writer.writerows((str(date), '30') for date in dates)
            result = replay(folder, source, prefix_n=180)
            self.assertEqual(result['metrics'], {})
            self.assertEqual(len(result['missing_seed_trajectories']), 12)
            self.assertFalse(result['verdict']['complete'])

    def test_incomplete_run_cannot_pass_even_with_good_synthetic_scores(self):
        def fixture(score):
            return dict(score=score, seeds=[score, score], annual={'2020': score, '2021': score},
                        cells=[dict(ratio=score) for _ in range(6)])
        scores = {mode: fixture(1.) for mode in ('price', 'clock', 'raw_hold', 'raw_event', 'hold')}
        scores['event'] = fixture(.8)
        result = verdict(scores, complete=False)
        self.assertFalse(result['development_gate_passed'])
        self.assertFalse(result['paper_acceptance_passed'])
        result = verdict({'price': fixture(1.), 'event': fixture(1.1)}, complete=False)
        self.assertTrue(result['decisive_necessary_gate_failure'])

    def test_mean_seed_loss_is_not_ensemble_loss(self):
        truth = np.ones((4, 20))
        prediction = np.stack((truth, -truth))
        result = score_predictions(prediction, truth, np.array(['2020', '2020', '2021', '2021']))
        self.assertEqual(result['score'], float(((prediction - truth) ** 2).mean()))
        self.assertEqual(result['ensemble'], float(((prediction.mean(0) - truth) ** 2).mean()))
        self.assertNotEqual(result['score'], result['ensemble'])
        self.assertEqual(len(result['cells']), 6)


if __name__ == '__main__':
    unittest.main()
