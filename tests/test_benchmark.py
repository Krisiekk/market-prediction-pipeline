import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import build_features as builder
from src.models.benchmark import TECHNICAL, make_models, prepare_dataset, summarize
from src.validation.splits import create_time_series_splits


class BenchmarkTests(unittest.TestCase):
    def test_gap_and_legacy_default(self):
        data = pd.DataFrame({'x': range(120)})
        for gap in (0, 1, 5):
            for train, test in create_time_series_splits(data, gap=gap):
                self.assertEqual(test[0] - train[-1] - 1, gap)

    def test_lag_before_drop_and_target_on_original_grid(self):
        raw = pd.DataFrame({'datetime': pd.bdate_range('2020-01-01', periods=8),
                            'close': [10., 12., 11., 15., 14., 17., 16., 20.]})
        def macro(df):
            return df.assign(macro=[1., 2., np.nan, 4., 5., 6., 7., 8.])
        with patch.object(builder.indicators, 'add_all_indicators', side_effect=lambda df: df), \
                patch.object(builder.intermarket, 'add_intermarket_features', side_effect=macro):
            result = builder.build_features(raw, horizon=2, intermarket_lag=1)
        self.assertEqual(result.datetime.tolist(), raw.datetime.iloc[[1, 2, 4, 5]].tolist())
        self.assertEqual(result.macro.tolist(), [1., 2., 4., 5.])
        expected = (raw.close.shift(-2) > raw.close).iloc[[1, 2, 4, 5]].astype(int)
        self.assertEqual(result.target.tolist(), expected.tolist())

    def test_models_scale_inside_pipeline(self):
        models = make_models()
        self.assertEqual(len(models), 6)
        for name in ('LogisticRegression', 'SVC'):
            self.assertIsInstance(models[name], Pipeline)
            self.assertIn('standardscaler', models[name].named_steps)
        self.assertFalse(models['HistGradientBoostingClassifier'].early_stopping)
        self.assertFalse(models['SVC'].named_steps['svc'].probability)

    def test_paired_summary_uses_pp_and_sample_std(self):
        rows = [dict(horizon=1, feature_set=variant, model='test', fold=i,
                     roc_auc=.5 + i * .01 + delta, balanced_accuracy=.5, accuracy=.6)
                for variant, delta in [('TECHNICAL', 0), ('SELECTED', .02), ('ALL', -.01)]
                for i in range(5)]
        result = summarize(pd.DataFrame(rows)).set_index('feature_set')
        self.assertAlmostEqual(result.loc['SELECTED', 'delta_auc_vs_technical_pp'], 2)
        self.assertAlmostEqual(result.loc['ALL', 'delta_auc_vs_technical_pp'], -1)
        self.assertAlmostEqual(result.loc['TECHNICAL', 'roc_auc_std'], np.std(np.arange(5)*.01, ddof=1))


if __name__ == '__main__':
    unittest.main()
