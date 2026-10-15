import unittest
import torch

torch.set_num_threads(1)


class SeparatedStateTests(unittest.TestCase):
    def test_band_state_update_does_not_freely_mix_other_components(self):
        from separated_model import SeparatedForecastModel
        torch.manual_seed(17)
        model = SeparatedForecastModel('fixed')
        components = torch.randn(2, 4)
        hidden = torch.randn(2, 4, 8)
        changed = components.clone()
        changed[:, 2] += 10
        before = model.update_bands(components, hidden)
        after = model.update_bands(changed, hidden)
        torch.testing.assert_close(before[:, [0, 1, 3]], after[:, [0, 1, 3]], rtol=0, atol=0)
        self.assertGreater(float((before[:, 2]-after[:, 2]).abs().max()), 0.)

    def test_complete_forecast_loss_trains_cross_scale_feedback(self):
        from separated_model import SeparatedForecastModel
        torch.manual_seed(17)
        model = SeparatedForecastModel('coupled')
        inputs = torch.randn(3, 60)
        prediction, trace = model(inputs, return_trace=True)
        self.assertEqual(tuple(prediction.shape), (3, 20))
        self.assertEqual(tuple(trace['hidden'].shape), (3, 60, 4, 8))
        torch.testing.assert_close(trace['components'].sum(-1), inputs)
        prediction.square().mean().backward()
        self.assertGreater(float(model.feedback.weight.grad.norm()), 0.)


if __name__ == '__main__':
    unittest.main()
