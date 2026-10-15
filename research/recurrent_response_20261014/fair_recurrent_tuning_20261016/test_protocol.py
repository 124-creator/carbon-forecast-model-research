import unittest
from protocol import configurations, plan, choose

class ProtocolTest(unittest.TestCase):
    def test_selection_requires_every_finite_trial(self):
        records = [dict(family=f, config_id=c['id'], cut=cut, selection_score=1 + i / 100)
                   for f in plan()['families'] for i, c in enumerate(configurations())
                   for cut in plan()['search_cuts']]
        result = choose(records)
        self.assertEqual({value['config_id'] for value in result.values()}, {'c00'})
        for incomplete in (records[:-1], records + [records[0]], [dict(r, selection_score=float('nan')) for r in records]):
            with self.assertRaises(ValueError):
                choose(incomplete)

    def test_search_is_finite_and_precedes_market(self):
        grid = configurations()
        schedule = plan()
        self.assertEqual(len(grid), 8)
        self.assertEqual(len({(r['hidden'], r['lr'], r['weight_decay']) for r in grid}), 8)
        self.assertEqual(grid[-1], dict(id='c07', hidden=16, lr=.003, weight_decay=.001))
        self.assertEqual(schedule['search_cuts'], [359, 419])
        self.assertLess(max(cut + 19 + 20 for cut in schedule['search_cuts']), 459)
        self.assertEqual(schedule['market_origins'], list(range(459, 930)))
        self.assertEqual(schedule['market_cuts'], [459 + 40 * i for i in range(12)])
        self.assertEqual(schedule['max_total_fits'], 268)

if __name__ == '__main__':
    unittest.main()
