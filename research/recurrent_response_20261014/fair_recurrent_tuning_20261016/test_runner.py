import unittest
from protocol import plan
from run_fair import trajectory_jobs

class RunnerTest(unittest.TestCase):
    def test_identical_selected_default_is_explicitly_aliased(self):
        families = plan()['families']
        selection = {family: dict(config_id='c07') for family in families}
        jobs, aliases = trajectory_jobs(selection)
        self.assertEqual(len(jobs), len(families) * len(plan()['market_seeds']))
        self.assertEqual(len({j['key'] for j in jobs}), len(jobs))
        self.assertTrue(all(v['default'] == v['selected'] for v in aliases.values()))
        selection = {family: dict(config_id='c00') for family in families}
        jobs, aliases = trajectory_jobs(selection)
        self.assertEqual(len(jobs), 2 * len(families) * len(plan()['market_seeds']))
        self.assertTrue(all(v['default'] != v['selected'] for v in aliases.values()))

if __name__ == '__main__':
    unittest.main()
