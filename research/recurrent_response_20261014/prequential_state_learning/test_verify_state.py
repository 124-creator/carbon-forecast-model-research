import importlib
import unittest

try:
    verifier = importlib.import_module('verify_state')
except ModuleNotFoundError:
    verifier = None


class VerificationTests(unittest.TestCase):
    def test_verifier_rejects_label_at_or_after_fit_cutoff(self):
        self.assertIsNotNone(verifier, 'Missing independent verifier')
        log = dict(end=459, train_max_target=378, validation_first_origin=379,
                   validation_max_target=459, parameters=1232)
        with self.assertRaises(ValueError):
            verifier.check_fit_log(log)
        log['validation_max_target'] = 458
        verifier.check_fit_log(log)


if __name__ == '__main__':
    unittest.main(verbosity=2)
