import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.features.calendar import filter_weekdays
from src.features import build_features as builder
from src.models.benchmark import prepare_dataset


class WeekdayCalendarTests(unittest.TestCase):
    def test_filter_copy_sort_idempotent_and_future_dates(self):
        raw = pd.DataFrame({'datetime': ['2030-05-06', '2030-05-04', '2030-05-03', '2030-05-05'], 'close': [4, 2, 1, 3]})
        saved = raw.copy(deep=True)
        result = filter_weekdays(raw)
        pd.testing.assert_frame_equal(raw, saved)
        self.assertEqual(result.datetime.dt.dayofweek.tolist(), [4, 0])
        self.assertTrue(result.datetime.is_monotonic_increasing)
        self.assertTrue(result.index.equals(pd.RangeIndex(2)))
        pd.testing.assert_frame_equal(result, filter_weekdays(result))
        with self.assertRaises(ValueError):
            filter_weekdays(pd.concat([raw, raw]))

    def test_weekend_excluded_before_rolling_lag_and_targets(self):
        raw = pd.DataFrame({'datetime': pd.date_range('2026-09-17', periods=12), 'close': np.arange(12, dtype=float)+100})
        raw.loc[raw.datetime.dt.dayofweek >= 5, 'close'] = 10000
        def indicators(df):
            self.assertTrue((df.datetime.dt.dayofweek < 5).all())
            return df.assign(rolling=df.close.rolling(2, min_periods=1).mean())
        def macro(df):
            self.assertTrue((df.datetime.dt.dayofweek < 5).all())
            return df.assign(macro=np.arange(len(df), dtype=float))
        with patch.object(builder.indicators, 'add_all_indicators', side_effect=indicators), patch.object(builder.intermarket, 'add_intermarket_features', side_effect=macro):
            h1 = builder.build_features(raw, horizon=1, intermarket_lag=0)
            h5 = builder.build_features(raw, horizon=5, intermarket_lag=0)
            lagged = builder.build_features(raw, horizon=1, intermarket_lag=1)
        self.assertEqual(h1.label_end_date.iloc[0], pd.Timestamp('2026-09-18'))
        self.assertEqual(h5.label_end_date.iloc[0], pd.Timestamp('2026-09-24'))
        self.assertEqual(h5.future_close.iloc[0], 107)
        self.assertEqual(h1.loc[h1.datetime.eq('2026-09-21'), 'rolling'].iloc[0], 102.5)
        self.assertEqual(lagged.macro.iloc[0], 0)
        self.assertEqual(lagged.datetime.iloc[0], pd.Timestamp('2026-09-18'))
        weekday = filter_weekdays(raw).set_index('datetime')
        for horizon, frame in [(1, h1), (5, h5)]:
            dates = pd.Series(weekday.index, index=weekday.index).shift(-horizon)
            self.assertEqual(frame.label_end_date.tolist(), frame.datetime.map(dates).tolist())

    def test_real_pipeline_diagnostics_not_in_X_and_prefix_invariance(self):
        # Real local inputs exercise all feature-generation stages without training.
        from src.config import DATA_RAW_DIR
        raw = pd.read_csv(DATA_RAW_DIR / 'xauusd_daily.csv')
        full, sets = prepare_dataset(raw, 5)
        for columns in sets.values():
            self.assertFalse(set(columns) & {'target', 'label_end_date', 'label_end', 'future_close', 'weekday'})
        self.assertTrue((full.datetime.dt.dayofweek < 5).all())
        self.assertTrue((full.label_end_date.dt.dayofweek < 5).all())
        self.assertFalse(full.datetime.duplicated().any())
        prefix, _ = prepare_dataset(raw.iloc[:-50], 5)
        pd.testing.assert_frame_equal(full.set_index('datetime').loc[prefix.datetime], prefix.set_index('datetime'))


if __name__ == '__main__':
    unittest.main()
