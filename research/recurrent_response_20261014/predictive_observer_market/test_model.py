import unittest
import torch


class ModelTests(unittest.TestCase):
    def test_conditional_outputs_future_vector_and_trains_variance(self):
        from model import Model
        torch.set_num_threads(1)
        torch.manual_seed(17)
        model = Model('RKNconditional')
        output = model(torch.randn(4, 60))
        self.assertEqual(tuple(output.shape), (4, 20))
        self.assertTrue(bool(torch.isfinite(output).all()))
        output.square().mean().backward()
        self.assertGreater(float(model.variance.weight.grad.norm()), 0)

    def test_global_variance_has_no_feature_weight_gradient(self):
        from model import Model
        torch.manual_seed(17)
        conditional = Model('RKNconditional')
        torch.manual_seed(17)
        global_model = Model('RKNglobal')
        for name, value in conditional.state_dict().items():
            self.assertTrue(torch.equal(value, global_model.state_dict()[name]))
        global_model(torch.randn(4, 60)).square().mean().backward()
        self.assertEqual(float(global_model.variance.weight.grad.norm()), 0)
        self.assertGreater(float(global_model.variance.bias.grad.norm()), 0)

    def test_patch_gru_sees_all_sixty_changes(self):
        from model import Model
        torch.manual_seed(17)
        model = Model('GRUpatch')
        x = torch.randn(4, 60, requires_grad=True)
        output = model(x)
        self.assertEqual(tuple(output.shape), (4, 20))
        output.square().mean().backward()
        self.assertTrue(bool((x.grad.abs().sum(dim=0) > 0).all()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1444)

    def test_fit_ignores_all_prices_after_training_cutoff(self):
        import numpy as np
        from run_market import fit, predict
        p = np.sin(np.arange(250) / 7) + np.arange(250) / 100
        q = p.copy()
        q[200:] += 10000
        for kind in ['GRUpatch', 'RKNglobal', 'RKNconditional']:
            a, sa, row = fit(p, 200, kind, 17, max_epochs=2)
            b, sb, _ = fit(q, 200, kind, 17, max_epochs=2)
            self.assertEqual(sa, sb)
            self.assertLess(row['max_training_target'], row['validation_first'])
            self.assertLess(row['max_validation_target'], 200)
            for name, value in a.state_dict().items():
                self.assertTrue(torch.equal(value, b.state_dict()[name]))
            self.assertTrue(np.array_equal(predict(a, sa, p, [199]), predict(b, sb, q, [199])))

    def test_score_averages_seed_losses_not_seed_predictions(self):
        import numpy as np
        from run_market import summarize, KINDS
        saved = dict(truth=np.ones((4, 20)), years=np.array(['2020', '2020', '2021', '2021']))
        for kind in KINDS:
            saved[kind + '_17'] = np.full((4, 20), 2.)
            saved[kind + '_29'] = np.zeros((4, 20))
        result = summarize(saved)
        for kind in KINDS:
            self.assertEqual(result['scores'][kind]['score'], 1.)
        self.assertFalse(result['development_gate'])

    def test_raw_price_replay_preserves_seed_loss_averaging(self):
        import numpy as np
        from verify_observer import score_raw_prices
        issued = np.full(4, 100.)
        actual = np.full((4, 20), 101.)
        prediction = np.stack([np.full((4, 20), 2.), np.zeros((4, 20))])
        score, cells = score_raw_prices(prediction, actual, issued, np.array(['2020', '2020', '2021', '2021']))
        self.assertEqual(score, 1.)
        self.assertEqual(len(cells), 6)


if __name__ == '__main__':
    unittest.main()
