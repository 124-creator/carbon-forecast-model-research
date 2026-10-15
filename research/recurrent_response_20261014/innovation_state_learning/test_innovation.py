"""Tests use sequential arithmetic independent of the production scan."""
import importlib
import importlib.util
import unittest
import torch


class InnovationTests(unittest.TestCase):
    def test_parallel_states_equal_sequential_states(self):
        self.assertIsNotNone(importlib.util.find_spec('innovation_model'), 'new state core is not implemented')
        module = importlib.import_module('innovation_model')
        torch.manual_seed(4)
        a = torch.randn(19, 3, 3, dtype=torch.float64) * .1
        b = torch.randn(19, 3, dtype=torch.float64)
        state = torch.zeros(3, dtype=torch.float64)
        expected = []
        for at, bt in zip(a, b):
            state = at @ state + bt
            expected.append(state)
        torch.testing.assert_close(module.affine_scan(a, b), torch.stack(expected), atol=1e-12, rtol=1e-12)


    def test_innovation_model_matches_causal_filter_with_contractive_operators(self):
        module = importlib.import_module('innovation_model')
        self.assertTrue(hasattr(module, 'InnovationNet'), 'forecast-state model missing')
        torch.manual_seed(7)
        delta = torch.randn(31, dtype=torch.float64)
        context = torch.randn(32, 3, dtype=torch.float64)
        base = torch.randn(32, 20, dtype=torch.float64) * .1
        model = module.InnovationNet('innovation_shift').double()
        with torch.no_grad():
            model.B.fill_(.07)
        gains = model.gains(delta, context)
        state = torch.zeros(20, dtype=torch.float64)
        expected = [state]
        shift = torch.diag(torch.ones(19, dtype=torch.float64), diagonal=1)
        weight = .9 ** (-torch.arange(20, dtype=torch.float64))
        for i, gain in enumerate(gains):
            operator = shift.clone()
            operator[:, 0] -= gain
            weighted = weight[:, None] * operator / weight[None, :]
            self.assertLessEqual(float(torch.linalg.svdvals(weighted)[0]), .99000000001)
            state = shift @ state + gain * (delta[i] - base[i, 0] - state[0])
            expected.append(state)
        prediction = model(delta, context, base)
        torch.testing.assert_close(prediction, base + torch.stack(expected).cumsum(-1), atol=2e-12, rtol=2e-12)
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        changed = delta.clone()
        changed[20:] += 100
        torch.testing.assert_close(model(changed, context, base)[:21], prediction[:21], atol=2e-12, rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
