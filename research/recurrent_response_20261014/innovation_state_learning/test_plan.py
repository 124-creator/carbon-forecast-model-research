"""The execution plan must exactly match the frozen fit budget."""
import importlib
import importlib.util
import unittest


class PlanTests(unittest.TestCase):
    def test_frozen_fit_plan_has_no_duplicate_constant_seed(self):
        self.assertIsNotNone(importlib.util.find_spec('run_innovation'), 'frozen experiment runner missing')
        module = importlib.import_module('run_innovation')
        tasks = module.planned_jobs()
        self.assertEqual(len(tasks), len(set(tasks)))
        self.assertEqual(len(tasks) * 12, 156)
        self.assertEqual([t for t in tasks if t[0] == 'constant_shift'], [('constant_shift', 17)])
        self.assertEqual(sorted(t[1] for t in tasks if t[0] == 'innovation_shift'), [17, 29])


if __name__ == '__main__':
    unittest.main()
