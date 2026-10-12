import importlib.util
from pathlib import Path
import unittest
import json
import hashlib
import tempfile

ROOT = Path(__file__).resolve().parents[1]

class ProbeTests(unittest.TestCase):
    def test_maturity_respects_cutoff(self):
        path = ROOT / 'tools' / 'research_probe.py'
        self.assertTrue(path.is_file(), 'Research probe not implemented')
        spec = importlib.util.spec_from_file_location('probe', path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [{'target_index': 9}, {'target_index': 10}, {'target_index': 11}]
        self.assertEqual(module.matured(rows, 10), rows[:2])

    def test_audit_checks_bytes_and_conflicts(self):
        spec = importlib.util.spec_from_file_location('probe', ROOT / 'tools/research_probe.py')
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'audit'), 'Audit not implemented')
        row = dict(currentDay='2020-01-01', productCode='000001', productName='GDEA',
                   endPrice=10, exchangeRate='1', message='当日无成交')
        body = json.dumps({'data': {'total': 1, 'rows': [row]}}).encode()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / 'page.json'
            path.write_bytes(body)
            sources = [dict(filename='page.json', bytes=len(body), sha256=hashlib.sha256(body).hexdigest())]
            result = module.audit(Path(directory), sources)
            self.assertEqual(result['rows'], 1)
            self.assertEqual(result['message_quantity_conflicts'], 1)
            self.assertFalse(result['training_allowed'])
            path.write_bytes(body + b' ')
            with self.assertRaises(ValueError):
                module.audit(Path(directory), sources)

    def test_objective_difference_is_not_market_gain(self):
        spec = importlib.util.spec_from_file_location('probe', ROOT / 'tools/research_probe.py')
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'objective_probe'), 'Objective probe not implemented')
        result = module.objective_probe()
        self.assertEqual(result['absolute_worst_risk_optimum'], 1.0)
        self.assertEqual(result['excess_worst_risk_optimum'], 0.0)
        self.assertFalse(result['market_data_used'])

if __name__ == '__main__':
    unittest.main()
