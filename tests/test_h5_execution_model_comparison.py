import unittest

import pandas as pd

from src.models.compare_h5_execution_models import add_ar_features, build_execution_labels


class H5ExecutionModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({
            "datetime": pd.bdate_range("2025-01-06", periods=8),
            "open": [100, 102, 103, 99, 105, 110, 108, 111],
            "close": [101, 104, 100, 106, 109, 115, 107, 112],
        })

    def test_execution_target_enters_next_open_and_exits_fifth_weekday_close(self):
        labels = build_execution_labels(self.raw)
        first = labels.iloc[0]
        expected_return = self.raw.loc[5, "close"] / self.raw.loc[1, "open"] - 1
        self.assertEqual(first.entry_date, self.raw.loc[1, "datetime"])
        self.assertEqual(first.label_end_date, self.raw.loc[5, "datetime"])
        self.assertAlmostEqual(first.forward_return, expected_return)
        self.assertEqual(first.target, int(expected_return > 0))

    def test_autoregressive_features_do_not_use_future_closes(self):
        altered = self.raw.copy()
        altered.loc[4, "close"] = 9999
        original_ar = add_ar_features(self.raw).set_index("datetime")
        altered_ar = add_ar_features(altered).set_index("datetime")
        date = self.raw.loc[3, "datetime"]
        pd.testing.assert_series_equal(original_ar.loc[date], altered_ar.loc[date])


if __name__ == "__main__":
    unittest.main()
