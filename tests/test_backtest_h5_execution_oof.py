import unittest

import pandas as pd

from src.models.backtest_h5_execution_oof import _trade_stream


class H5ExecutionOOFBacktestTests(unittest.TestCase):
    def test_trades_do_not_overlap_and_cost_is_deducted_per_round_trip(self):
        dates = pd.bdate_range("2025-01-06", periods=16)
        predictions = pd.DataFrame({
            "date": dates[:11],
            "entry_date": dates[1:12],
            "label_end_date": dates[5:16],
            "fold": [1] * 5 + [2] * 5 + [3],
            "y_true": [1] * 11,
            "forward_return": [0.05] * 11,
            "AlwaysLong": [1] * 11,
        })
        prices = pd.DataFrame({"open": [100.0] * 16, "close": [105.0] * 16}, index=dates)
        trades, candidates, skipped, _, _ = _trade_stream(
            predictions, "AlwaysLong", "long_only", prices, 0.5,
        )
        self.assertEqual(candidates, 11)
        self.assertEqual(skipped, 8)
        self.assertEqual(len(trades), 3)
        self.assertTrue((trades.entry_date.iloc[1:].to_numpy() > trades.exit_date.iloc[:-1].to_numpy()).all())
        self.assertAlmostEqual(trades.net_return.iloc[0], 0.045)


if __name__ == "__main__":
    unittest.main()
