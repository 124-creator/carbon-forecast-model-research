import unittest
import torch

torch.set_num_threads(1)


class CoupledDecompositionTests(unittest.TestCase):
    def test_ladder_components_conserve_the_observed_change(self):
        from coupled_model import ladder_step
        x = torch.tensor([2., -3.])
        previous = torch.tensor([[.1, .2, .3], [-.4, -.5, -.6]])
        rates = torch.tensor([[.05, .2, .5], [.1, .3, .8]])
        states, bands = ladder_step(x, previous, rates)
        torch.testing.assert_close(bands.sum(-1), x)
        torch.testing.assert_close(states, (1-rates)*previous+rates*x[:, None])

    def test_fixed_model_has_full_horizon_output_and_ordered_rates(self):
        from coupled_model import ForecastModel
        torch.manual_seed(17)
        model = ForecastModel('fixed')
        prediction, trace = model(torch.randn(3, 60), return_trace=True)
        self.assertEqual(tuple(prediction.shape), (3, 20))
        self.assertEqual(tuple(trace['components'].shape), (3, 60, 4))
        rates = trace['rates']
        self.assertTrue(bool(((rates > 0) & (rates < 1)).all()))
        self.assertTrue(bool((rates[:, :, 1:] > rates[:, :, :-1]).all()))

    def test_prediction_loss_trains_global_decomposition_rates(self):
        from coupled_model import ForecastModel
        torch.manual_seed(17)
        model = ForecastModel('global')
        model(torch.randn(5, 60)).square().mean().backward()
        self.assertIsNotNone(model.raw_rates.grad)
        self.assertGreater(float(model.raw_rates.grad.norm()), 0.)

    def test_prediction_loss_reaches_state_feedback_weights(self):
        from coupled_model import ForecastModel
        torch.manual_seed(17)
        model = ForecastModel('coupled')
        x = torch.randn(5, 60)
        model(x).square().mean().backward()
        self.assertIsNotNone(model.feedback.weight.grad)
        self.assertGreater(float(model.feedback.weight.grad.norm()), 0.)

    def test_plain_reference_matches_existing_gru_definition(self):
        from coupled_model import ForecastModel
        x = torch.randn(3, 60)
        torch.manual_seed(17)
        encoder = torch.nn.GRU(1, 16, batch_first=True)
        head = torch.nn.Linear(16, 20)
        expected = head(encoder(x[:, :, None])[0][:, -1])
        torch.manual_seed(17)
        model = ForecastModel('plain')
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)

    def test_decomposed_variants_have_identical_initial_predictions(self):
        from coupled_model import ForecastModel
        x = torch.randn(4, 60)
        predictions = []
        for kind in ['fixed', 'global', 'coupled']:
            torch.manual_seed(29)
            predictions.append(ForecastModel(kind)(x))
        for prediction in predictions[1:]:
            torch.testing.assert_close(prediction, predictions[0], rtol=0, atol=0)

    def test_later_inputs_cannot_change_earlier_encoder_states(self):
        from coupled_model import ForecastModel
        torch.manual_seed(17)
        model = ForecastModel('coupled')
        with torch.no_grad():
            model.feedback.weight.normal_(0, .1)
        x = torch.randn(3, 60)
        changed = x.clone()
        changed[:, 31:] += 1000
        original = model(x, return_trace=True)[1]
        perturbed = model(changed, return_trace=True)[1]
        for key in ['components', 'rates', 'hidden']:
            torch.testing.assert_close(original[key][:, :31], perturbed[key][:, :31], rtol=0, atol=0)
        torch.testing.assert_close(original['components'].sum(-1), x)

    def test_raw_price_scorer_assigns_one_to_random_walk(self):
        from verify_coupled import score_raw_prices
        issued = torch.arange(1., 5.).numpy()
        actual = issued[:, None]+torch.arange(1., 21.).numpy()[None]
        years = ['2020', '2020', '2021', '2021']
        prediction = torch.zeros(2, 4, 20).numpy()
        score, cells = score_raw_prices(prediction, actual, issued, years)
        self.assertEqual(score, 1.)
        self.assertEqual(len(cells), 6)

    def test_fit_is_unchanged_by_prices_after_the_maturity_cutoff(self):
        from run_coupled import fit
        import numpy as np
        prices = np.cumsum(np.random.default_rng(3).normal(size=600))
        model, scale, row = fit(prices, 459, 'coupled', 17, max_epochs=1)
        changed = prices.copy()
        changed[459:] += 10000
        other, other_scale, _ = fit(changed, 459, 'coupled', 17, max_epochs=1)
        self.assertEqual(scale, other_scale)
        self.assertLess(row['max_training_target'], row['validation_first'])
        self.assertLess(row['max_validation_target'], 459)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
