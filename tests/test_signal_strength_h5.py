import unittest

import numpy as np
import pandas as pd

from src.models.signal_strength_h5 import (
    _selection, analyze_coverage, analyze_quantiles, attach_future_return,
    load_and_validate,
)


class SignalStrengthH5Tests(unittest.TestCase):
    def test_common_oof_folds_and_fixed_coverage_analysis(self):
        oof = load_and_validate()
        joined = attach_future_return(oof)
        coverage, folds, signals = analyze_coverage(joined)
        quantiles, quantile_folds = analyze_quantiles(joined)
        self.assertEqual(len(oof), 3985)
        self.assertEqual(folds.groupby(['variant','coverage_target']).size().unique().tolist(), [5])
        self.assertEqual(quantile_folds.groupby(['variant','quantile']).size().unique().tolist(), [5])
        self.assertEqual(coverage.variant.nunique(), 3)
        self.assertEqual(quantiles['quantile'].nunique(), 5)
        self.assertTrue((coverage.n_signals <= coverage.n_selected).all())
        self.assertEqual(signals.groupby(['variant','coverage_target']).date.nunique().min(), 3985)
        self.assertEqual(len(signals), 3985 * 5 * 3)

    def test_selection_uses_only_fold_local_score_and_strength(self):
        sample = pd.DataFrame({'date':pd.date_range('2024-01-01',periods=10),
            'score':[.50,.51,.52,.53,.54,.46,.47,.48,.49,.55],
            'y_true':[0]*10,'future_return':np.arange(10)})
        selected, strength = _selection(sample, 'score', .2)
        altered = sample.assign(y_true=1-sample.y_true, future_return=-sample.future_return)
        selected2, strength2 = _selection(altered, 'score', .2)
        self.assertEqual(selected.tolist(), selected2.tolist())
        self.assertTrue(np.array_equal(strength.to_numpy(), strength2.to_numpy()))
        self.assertEqual(int(selected.sum()), 2)
        self.assertEqual(set(sample.loc[selected,'date']), {sample.date.iloc[9],sample.date.iloc[4]})


if __name__ == '__main__':
    unittest.main()
