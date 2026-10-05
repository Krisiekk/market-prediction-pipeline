import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features import intermarket


class IntermarketTests(unittest.TestCase):
    def test_merge_skips_missing_without_using_future(self):
        left = pd.DataFrame({'datetime': pd.date_range('2020-01-01', periods=5)})
        right = pd.DataFrame({
            'datetime': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-05']),
            'value': [1., np.nan, 2.],
        })
        result = intermarket._merge_last_known(left, right, 'value')
        self.assertTrue(pd.isna(result.value.iloc[0]))
        self.assertEqual(result.value.iloc[1:].tolist(), [1., 1., 1., 2.])

    def test_trends_volatility_and_no_future_dependency(self):
        dates = pd.date_range('2020-01-01', periods=100)
        x = np.arange(100, dtype=float)
        dxy = pd.DataFrame({'datetime': dates, 'dxy': 100 * np.exp(.002 * x)})

        def fred(path, column):
            values = -1 + .02 * x if column == 'dfii10' else 2 + .01 * x
            return pd.DataFrame({'datetime': dates, column: values})

        with patch.object(intermarket, '_load_dxy', return_value=dxy), \
                patch.object(intermarket, '_load_fred', side_effect=fred):
            full = intermarket.add_intermarket_features(pd.DataFrame({'datetime': dates}))
            prefix = intermarket.add_intermarket_features(pd.DataFrame({'datetime': dates[:75]}))
        pd.testing.assert_frame_equal(full.iloc[:75], prefix)
        self.assertAlmostEqual(full.dxy_return_20d.iloc[-1], np.exp(.04) - 1)
        self.assertAlmostEqual(full.dxy_volatility_20d.iloc[-1], 0)
        self.assertAlmostEqual(full.dxy_vs_sma50.iloc[-1],
                               full.dxy.iloc[-1] / full.dxy.iloc[-50:].mean() - 1)
        self.assertAlmostEqual(full.dxy_sma20_vs_sma50.iloc[-1],
                               full.dxy.iloc[-20:].mean() / full.dxy.iloc[-50:].mean() - 1)
        for column, step in [('dgs10', .01), ('dfii10', .02)]:
            self.assertAlmostEqual(full[f'{column}_change_20d'].iloc[-1], 20 * step)
            self.assertAlmostEqual(full[f'{column}_vs_sma20'].iloc[-1], 9.5 * step)
            self.assertAlmostEqual(full[f'{column}_sma20_vs_sma50'].iloc[-1], 15 * step)
            self.assertTrue(full[f'{column}_sma20_vs_sma50'].iloc[:49].isna().all())


if __name__ == '__main__':
    unittest.main()
