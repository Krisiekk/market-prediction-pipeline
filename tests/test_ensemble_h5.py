import unittest

from src.models.ensemble_h5 import _chosen_predictions, _validate, build_oof, summarize


class EnsembleH5Tests(unittest.TestCase):
    def test_saved_oof_are_aligned_and_fixed_ensembles_are_computable(self):
        chosen = _chosen_predictions()
        _validate(chosen)
        oof = build_oof(chosen)
        self.assertEqual(len(oof), 3985)
        self.assertEqual(oof.groupby('fold').size().to_dict(), {i: 797 for i in range(1, 6)})
        self.assertFalse(oof.date.duplicated().any())
        self.assertTrue((oof.primary_probability_ensemble == .5 * (oof.logistic_probability + oof.extratrees_probability)).all())
        for _, fold in oof.groupby('fold'):
            self.assertAlmostEqual(fold.logistic_rank.min(), 0.)
            self.assertAlmostEqual(fold.logistic_rank.max(), 1.)
            self.assertAlmostEqual(fold.svc_rank.min(), 0.)
            self.assertAlmostEqual(fold.svc_rank.max(), 1.)
        summary, fold_results = summarize(oof)
        self.assertEqual(summary.variant.nunique(), 8)
        self.assertEqual(len(fold_results), 40)


if __name__ == '__main__':
    unittest.main()
