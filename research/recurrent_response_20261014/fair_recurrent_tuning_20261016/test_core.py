import unittest
import numpy as np
import torch
from fair_core import configure, reference, RecurrentNet

class CoreTest(unittest.TestCase):
    def test_initial_readout_solves_common_expanded_risk(self):
        configure()
        torch.manual_seed(12)
        features = torch.randn(80, 9)
        residual = torch.randn(80, 20)
        for mode in ('gru_dense', 'gru_profile', 'gru_rank6', 'lstm_dense'):
            model = RecurrentNet(mode, 4)
            model.initialize(features, residual)
            coeff = model.readout.detach().clone().requires_grad_()
            expanded = coeff @ model.basis.T
            correction = features @ expanded
            objective = ((correction - residual).square() + correction.square()).mean() + .01 * expanded.square().sum() / 20
            gradient = torch.autograd.grad(objective, coeff)[0]
            self.assertLess(float(gradient.abs().max()), 1e-10)
            np.testing.assert_allclose(model.correction(features).detach().numpy(), correction.detach().numpy(), atol=1e-13)

    def test_matched_features_ignore_future_price_and_reference(self):
        configure()
        rng = np.random.default_rng(6)
        price = 20 + np.cumsum(rng.normal(0, .2, 180))
        future = price.copy()
        future[101:] += 500
        base, context = reference(price)
        other_base, other_context = reference(future)
        np.testing.assert_array_equal(base[:101], other_base[:101])
        np.testing.assert_array_equal(context[:101], other_context[:101])
        origins = np.array([60, 80, 100])
        same = []
        for mode in ('gru_dense', 'gru_profile', 'gru_rank6', 'lstm_dense'):
            torch.manual_seed(17)
            model = RecurrentNet(mode, 4)
            g = model.features(price, context, origins, 1.)
            changed = model.features(future, other_context, origins, 1.)
            np.testing.assert_array_equal(g.detach().numpy(), changed.detach().numpy())
            self.assertEqual(tuple(g.shape), (3, 9))
            if mode.startswith('gru'):
                same.append(g.detach().numpy())
        for g in same[1:]:
            np.testing.assert_array_equal(g, same[0])

if __name__ == '__main__':
    unittest.main()
