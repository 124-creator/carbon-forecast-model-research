import importlib
import unittest
import numpy as np
import torch

try:
    state = importlib.import_module('recurrent_state')
except ModuleNotFoundError:
    state = None


class StateTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        torch.manual_seed(13)

    def test_profile_head_solves_regularized_normal_equation(self):
        self.assertIsNotNone(state, 'Missing model implementation')
        g = torch.randn(30, 4)
        e = torch.randn(30, 3)
        b = state.profile_head(g, e)
        expected = torch.linalg.solve(2 * g.T @ g / 30 + .01 * torch.eye(4), g.T @ e / 30)
        torch.testing.assert_close(b, expected, rtol=1e-13, atol=1e-13)

    def test_mature_head_ignores_equal_or_future_maturity_targets(self):
        self.assertTrue(hasattr(state, 'mature_heads'), 'Missing maturity-conditioned head')
        g = torch.randn(80, 4, requires_grad=True)
        e = torch.randn(80, 3)
        orig = torch.arange(80)
        query = torch.tensor([70])
        head, counts = state.mature_heads(g, e, orig, query, complete=False)
        changed = e.clone()
        changed[68:, 1] += 1000
        altered, _ = state.mature_heads(g, changed, orig, query, complete=False)
        torch.testing.assert_close(head[:, 1], altered[:, 1], rtol=0, atol=0)
        allowed = orig + 2 < 70
        expected = state.profile_head(g[allowed], e[allowed, 1:2])[:, 0]
        torch.testing.assert_close(head[0, 1], expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(int(counts[0, 1]), int(allowed.sum()))

    def test_prequential_loss_has_correct_gradient_through_inner_fit(self):
        self.assertTrue(hasattr(state, 'prequential_loss'), 'Missing joint prequential objective')
        g = torch.randn(80, 3, requires_grad=True)
        e = torch.randn(80, 4)
        orig = torch.arange(80)
        queries = torch.tensor([65, 70, 75])
        loss = state.prequential_loss(g, e, orig, queries, complete=False)
        gradient = torch.autograd.grad(loss, g)[0]
        plus = g.detach().clone()
        minus = g.detach().clone()
        plus[5, 1] += 1e-6
        minus[5, 1] -= 1e-6
        finite = (state.prequential_loss(plus, e, orig, queries, False) -
                  state.prequential_loss(minus, e, orig, queries, False)) / 2e-6
        self.assertLess(abs(float(finite - gradient[5, 1])), 1e-8)
        self.assertGreater(abs(float(gradient[5, 1])), 1e-9)

    def test_recurrent_feature_is_causal_under_both_history_encodings(self):
        self.assertTrue(hasattr(state, 'RecurrentPredictor'), 'Missing complete recurrent model')
        model = state.RecurrentPredictor()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1232)
        delta = torch.randn(150)
        changed = delta.clone()
        changed[90:] += 1000
        origins = torch.tensor([60, 75, 90])
        context = torch.randn(3, 3)
        for kind in ['window', 'full']:
            a = model.features(delta, context, origins, kind)
            b = model.features(changed, context, origins, kind)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        full = model.features(delta, context[:1], origins[:1], 'full')
        window = model.features(delta, context[:1], origins[:1], 'window')
        torch.testing.assert_close(full, window, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
