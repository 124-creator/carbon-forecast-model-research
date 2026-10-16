"""Regression tests for the separately declared missing-fit completion."""
import os
import tempfile
import unittest
from pathlib import Path


class SupplementTests(unittest.TestCase):
    def test_merged_paths_reject_duplicates_and_unpaired_artifacts(self):
        from hsc_supplement import merged_paths
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temp:
            a, b = Path(temp) / 'a', Path(temp) / 'b'
            a.mkdir()
            b.mkdir()
            (a / 'x.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                merged_paths(a, b)
            (a / 'x.npz').write_bytes(b'not loaded by inventory')
            self.assertEqual(set(merged_paths(a, b)), {'x'})
            (b / 'x.json').write_text('{}', encoding='utf-8')
            (b / 'x.npz').write_bytes(b'not loaded by inventory')
            with self.assertRaises(ValueError):
                merged_paths(a, b)

    def test_original_hash_check_rejects_changed_bytes(self):
        from hsc_supplement import check_originals, file_hashes
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temp:
            root = Path(temp)
            (root / 'x.json').write_text('{}', encoding='utf-8')
            hashes = file_hashes(root, [root / 'x.json'])
            check_originals(root, hashes)
            (root / 'x.json').write_text('{"changed":true}', encoding='utf-8')
            with self.assertRaises(ValueError):
                check_originals(root, hashes)

    def test_independent_inventory_merges_only_disjoint_pairs(self):
        from hsc_verify import saved_market_inventory
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temp:
            a, b = Path(temp) / 'a', Path(temp) / 'b'
            a.mkdir()
            b.mkdir()
            selection = {'raw_gru': {'config_id': 'c02'}}
            for i, cut in enumerate(range(459, 900, 40)):
                for seed in (17, 29):
                    root = a if i < 4 else b
                    stem = root / f'raw_gru_c02_s{seed}_t{cut}'
                    stem.with_suffix('.json').write_text('{}', encoding='utf-8')
                    stem.with_suffix('.npz').write_bytes(b'inventory only')
            saved, missing, complete = saved_market_inventory(a, selection, supplemental=b)
            self.assertEqual((len(saved), missing, complete), (24, [], ['raw_gru']))
            (b / 'raw_gru_c02_s17_t459.json').write_text('{}', encoding='utf-8')
            (b / 'raw_gru_c02_s17_t459.npz').write_bytes(b'inventory only')
            with self.assertRaises(ValueError):
                saved_market_inventory(a, selection, supplemental=b)


if __name__ == '__main__':
    unittest.main()
