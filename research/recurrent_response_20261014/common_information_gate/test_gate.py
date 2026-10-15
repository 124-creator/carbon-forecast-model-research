import unittest


class CommonInformationTests(unittest.TestCase):
    def test_same_target_oracles_pass_common_history_but_fail_full_past(self):
        from check_common_information import check
        result = check()
        self.assertEqual(result['oracle_common_conditional_squared'], 0.)
        self.assertEqual(result['oracle_full_past_conditional_squared'], 1.)
        self.assertEqual(result['oracle_pathwise_revision_squared'], 2.)
        self.assertEqual(result['oracle_prediction_mse'], 2.)

    def test_pathwise_penalty_biases_an_otherwise_correct_predictor(self):
        from check_common_information import check
        cases = check()['penalty_cases']
        self.assertAlmostEqual(cases['common']['prediction_mse'], 2.)
        self.assertGreater(cases['past']['prediction_mse'], 2.)
        self.assertGreater(cases['path']['prediction_mse'], 2.)

    def test_price_anchor_offset_is_retained_in_augmented_regression(self):
        from check_common_information import check
        result = check()
        self.assertLess(result['augmented_coefficient_max_error'], 1e-12)
        self.assertGreater(result['omitted_anchor_prediction_mse'], result['oracle_prediction_mse'])

    def test_common_history_constraint_alone_does_not_identify_forecasts(self):
        from check_common_information import check
        result = check()
        self.assertEqual(result['constant_predictor_common_penalty'], 0.)
        self.assertEqual(result['constant_predictor_mse'], 4.)


if __name__ == '__main__':
    unittest.main()
