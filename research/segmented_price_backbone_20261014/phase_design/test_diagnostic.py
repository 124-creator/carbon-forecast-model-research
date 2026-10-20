import unittest
from diagnostic import run_checks
class Contract(unittest.TestCase):
    def test_shared_capacity_information_and_equivalence(self):
        r=run_checks()
        self.assertLess(r['injective_relu_reconstruction_max_error'],1e-12)
        self.assertGreater(r['injective_hard_shift_defect'],1e-4)
        self.assertLess(r['hard_vs_polyphase_zero_error'],1e-12)
        self.assertGreater(r['multiphase_gru_vs_one_sliding_gru_error'],1e-6)
        self.assertLess(r['polyphase_vs_sliding_embedding_max_error'],1e-12)
        self.assertLess(r['learned_pool_vs_ensemble_max_error'],1e-12)
        self.assertLess(r['information_sorted_window_error'],1e-12)
        self.assertEqual(len(set(r['parameter_counts'].values())),1)
        self.assertGreater(r['hard_embedding_shift_defect'],1e-4)
        self.assertLess(r['dense_embedding_shift_defect'],1e-12)
        self.assertLess(r['full_group_invariance_error'],1e-12)
        self.assertGreater(r['phase_mean_not_full_group_error'],1e-6)
if __name__=='__main__': unittest.main()
