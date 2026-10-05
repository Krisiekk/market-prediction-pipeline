"""Analyze frozen H5 execution backtest trades by fold-local volatility regime."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.features.calendar import filter_weekdays

SPLITS_PATH = DATA_PROCESSED_DIR / "h5_execution_model_comparison" / "splits.csv"
TRADES_PATH = DATA_PROCESSED_DIR / "h5_execution_cost_backtest" / "trades.csv"
OUT = DATA_PROCESSED_DIR / "h5_execution_volatility_regimes"
WINDOW = 20


def assign_volatility_regimes(raw, splits, window=WINDOW):
    """Assign low/medium/high trailing volatility with train-only fold cutoffs."""
    weekday = filter_weekdays(raw).sort_values("datetime").drop_duplicates("datetime").copy()
    returns = pd.to_numeric(weekday.close, errors="coerce").pct_change(fill_method=None)
    vol = returns.rolling(window=window, min_periods=window).std()
    volatility = pd.DataFrame({"date": weekday.datetime, "realized_volatility": vol})
    records, cutoff_rows = [], []
    for split in splits.itertuples(index=False):
        train_start, train_end = pd.Timestamp(split.train_start), pd.Timestamp(split.train_end)
        test_start, test_end = pd.Timestamp(split.test_start), pd.Timestamp(split.test_end)
        train_vol = volatility.loc[
            volatility.date.between(train_start, train_end), "realized_volatility"
        ].dropna()
        if train_vol.empty:
            raise ValueError(f"Fold {split.fold} has no training volatility history")
        low_cut, high_cut = train_vol.quantile([1 / 3, 2 / 3]).to_numpy()
        if not np.isfinite(low_cut + high_cut) or low_cut >= high_cut:
            raise ValueError(f"Fold {split.fold} volatility cutoffs are invalid")
        test = volatility[volatility.date.between(test_start, test_end)].copy()
        if test.realized_volatility.isna().any():
            raise ValueError(f"Fold {split.fold} has missing test volatility")
        test["fold"] = int(split.fold)
        test["low_cutoff_train_only"] = low_cut
        test["high_cutoff_train_only"] = high_cut
        test["volatility_regime"] = np.select(
            [test.realized_volatility <= low_cut, test.realized_volatility <= high_cut],
            ["low", "medium"], default="high",
        )
        records.append(test)
        cutoff_rows.append({
            "fold": int(split.fold),
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "volatility_window_weekdays": window,
            "training_volatility_observations": int(len(train_vol)),
            "low_medium_cutoff": float(low_cut),
            "medium_high_cutoff": float(high_cut),
        })
    regimes = pd.concat(records, ignore_index=True)
    if regimes.date.duplicated().any():
        raise ValueError("A date was assigned to more than one outer test fold")
    return regimes, pd.DataFrame(cutoff_rows)


def summarize_by_regime(trades):
    """Report per-trade outcomes; do not compound disjoint regime subsets."""
    rows = []
    keys = ["model", "mode", "cost_usd_per_oz", "volatility_regime"]
    for values, part in trades.groupby(keys, sort=True):
        model, mode, cost, regime = values
        net = part.net_return.to_numpy(dtype=float)
        gains, losses = net[net > 0].sum(), -net[net < 0].sum()
        rows.append({
            "model": model,
            "mode": mode,
            "cost_usd_per_oz": cost,
            "volatility_regime": regime,
            "n_trades": len(part),
            "mean_gross_return_per_trade": part.gross_return.mean(),
            "mean_net_return_per_trade": part.net_return.mean(),
            "median_net_return_per_trade": part.net_return.median(),
            "win_rate": float(np.mean(net > 0)),
            "profit_factor": gains / losses if losses else np.inf,
            "net_pnl_usd_per_oz_sum": part.net_usd_per_oz.sum(),
        })
    return pd.DataFrame(rows)


def run(output_dir=OUT):
    raw = pd.read_csv(DATA_RAW_DIR / "xauusd_daily.csv", parse_dates=["datetime"])
    splits = pd.read_csv(SPLITS_PATH, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    trades = pd.read_csv(TRADES_PATH, parse_dates=["signal_date", "entry_date", "exit_date"])
    regimes, cutoffs = assign_volatility_regimes(raw, splits)
    joined = trades.merge(
        regimes[["date", "fold", "realized_volatility", "low_cutoff_train_only",
                 "high_cutoff_train_only", "volatility_regime"]],
        left_on=["signal_date", "fold"], right_on=["date", "fold"],
        how="left", validate="many_to_one",
    )
    if joined.volatility_regime.isna().any():
        raise ValueError("Some trades did not receive an outer-fold volatility regime")
    summary = summarize_by_regime(joined)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regimes.to_csv(output_dir / "oof_volatility_regimes.csv", index=False)
    cutoffs.to_csv(output_dir / "fold_cutoffs.csv", index=False)
    joined.to_csv(output_dir / "trades_with_regimes.csv", index=False)
    summary.to_csv(output_dir / "regime_summary.csv", index=False)
    metadata = {
        "experiment": "h5_execution_volatility_regimes_v1",
        "regime_feature": f"rolling standard deviation of weekday close returns, window={WINDOW}",
        "regime_cutoffs": "training-only per outer fold, terciles; applied to that fold's test period",
        "regime_known_at": "signal after close(t)",
        "statistics": "per-trade within regime; no compounding across disjoint subsets",
        "input_backtest": "non-overlapping execution-aligned OOF trades",
        "holdout": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary, regimes, cutoffs, joined


if __name__ == "__main__":
    result = run()
    sample = result[0].query("cost_usd_per_oz == 0.5 and mode == 'long_only'")
    print(sample.to_string(index=False))
