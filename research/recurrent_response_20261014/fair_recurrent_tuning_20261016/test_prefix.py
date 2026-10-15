import io
import unittest
from pathlib import Path
from unittest.mock import patch
from fair_data import load_prefix

class Guarded(io.BytesIO):
    def __init__(self, raw):
        super().__init__(raw)
        self.requests = []
    def read(self, size: int | None = -1):
        self.requests.append(size)
        if size != 29204 or len(self.requests) > 1:
            raise AssertionError('Reader requested outside the approved prefix')
        return super().read(size)

class PrefixTest(unittest.TestCase):
    def test_approved_bytes_only_and_authenticated(self):
        root = Path(__file__).resolve().parents[3]
        path = root / 'data/parsed/figshare_28682348/guangzhou_reported.csv'
        with path.open('rb', buffering=0) as handle:
            raw = handle.read(29204)
        guard = Guarded(raw + b'FORBIDDEN_SUFFIX_TEST_ONLY')
        with patch('fair_data.Path.open', return_value=guard):
            price, dates, receipt = load_prefix(path)
        self.assertEqual(guard.requests, [29204])
        self.assertEqual(len(price), 950)
        self.assertEqual(dates[-1], '2021-12-31')
        self.assertFalse(receipt['next_byte_requested'])
        with patch('fair_data.Path.open', return_value=Guarded(b'X' + raw[1:])):
            with self.assertRaises(ValueError):
                load_prefix(path)

if __name__ == '__main__':
    unittest.main()
