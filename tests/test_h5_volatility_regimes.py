import unittest

import numpy as np
import pandas as pd

from src.models.analyze_h5_volatility_regimes import assign_volatility_regimes


class H5VolatilityRegimeTests(unittest.TestCase):
    def test_future_prices_do_not_change_fold_training_cutoffs(self):
        dates = pd.bdate_range("2020-01-06", periods=100)
        returns = np.sin(np.arange(100) * 0.73) * 0.004
        close = 1500 * np.cumprod(1 + returns)
        raw = pd.DataFrame({"datetime": dates, "close": close})
        splits = pd.DataFrame([{
            "fold": 1,
            "train_start": dates[20],
            "train_end": dates[59],
            "test_start": dates[60],
            "test_end": dates[-1],
        }])
        _, original_cutoffs = assign_volatility_regimes(raw, splits)

        changed = raw.copy()
        future_returns = np.where(np.arange(40) % 2 == 0, 0.08, -0.07)
        changed.loc[60:, "close"] = close[59] * np.cumprod(1 + future_returns)
        _, changed_cutoffs = assign_volatility_regimes(changed, splits)
        np.testing.assert_allclose(
            original_cutoffs[["low_medium_cutoff", "medium_high_cutoff"]],
            changed_cutoffs[["low_medium_cutoff", "medium_high_cutoff"]],
        )


if __name__ == "__main__":
    unittest.main()
