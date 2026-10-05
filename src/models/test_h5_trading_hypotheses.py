"""Run fixed, fold-safe regime/filter hypotheses on saved H5 OOF predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.features.calendar import filter_weekdays
from src.models.analyze_h5_volatility_regimes import assign_volatility_regimes
from src.models.backtest_h5_execution_oof import (
    _summarize_trades,
    build_prediction_table,
)

OUT = DATA_PROCESSED_DIR / "h5_trading_hypotheses"
COSTS = (0.0, 0.50, 1.0)
MODEL_CANDIDATES = (
    "LogisticRegression", "ExtraTreesClassifier", "Ensemble_HardVote_5", "ARX_Ridge",
)
FILTERS = {
    "no_filter": "No additional gate; baseline reference",
    "skip_low_volatility": "Trade only when fold-local 20-day volatility is medium/high",
    "bull_trend_sma50": "Trade only when close(t) is above trailing SMA50",
    "positive_momentum_20d": "Trade only when close(t)/close(t-20)-1 is positive",
}


def add_signal_gates(predictions, raw, splits):
    weekday = filter_weekdays(raw).sort_values("datetime").drop_duplicates("datetime").copy()
    close = pd.to_numeric(weekday.close, errors="coerce")
    weekday["realized_volatility_20"] = close.pct_change(fill_method=None).rolling(20, min_periods=20).std()
    weekday["sma_50"] = close.rolling(50, min_periods=50).mean()
    weekday["momentum_20d"] = close / close.shift(20) - 1
    regimes, _ = assign_volatility_regimes(raw, splits)
    gates = weekday[["datetime", "close", "realized_volatility_20", "sma_50", "momentum_20d"]].merge(
        regimes[["date", "volatility_regime"]], left_on="datetime", right_on="date",
        how="inner", validate="one_to_one",
    ).drop(columns="date")
    gates["skip_low_volatility"] = gates.volatility_regime.ne("low")
    gates["bull_trend_sma50"] = gates.close.gt(gates.sma_50)
    gates["positive_momentum_20d"] = gates.momentum_20d.gt(0)
    merged = predictions.merge(
        gates.rename(columns={"datetime": "date"}), on="date", how="left",
        validate="one_to_one",
    )
    if merged.volatility_regime.isna().any():
        raise ValueError("Missing signal-time regime/gate values")
    return merged


def _simulate(predictions, strategy, gate_name, prices, cost):
    gate_column = None if gate_name == "no_filter" else gate_name
    next_signal_allowed = pd.Timestamp.min
    trades = []
    candidates = skipped = 0
    for row in predictions.itertuples(index=False):
        if gate_column is not None and not bool(getattr(row, gate_column)):
            continue
        pred = 1 if strategy == "AlwaysLong" else int(getattr(row, strategy))
        if pred != 1:
            continue
        candidates += 1
        signal_date = pd.Timestamp(row.date)
        if signal_date < next_signal_allowed:
            skipped += 1
            continue
        entry_date, exit_date = pd.Timestamp(row.entry_date), pd.Timestamp(row.label_end_date)
        entry = float(prices.loc[entry_date, "open"])
        exit_price = float(prices.loc[exit_date, "close"])
        gross_usd_per_oz = exit_price - entry
        if not np.isclose(gross_usd_per_oz / entry, float(row.forward_return), rtol=1e-8, atol=1e-10):
            raise ValueError("OHLC trade return does not match execution-aligned target")
        trades.append({
            "signal_date": signal_date, "entry_date": entry_date, "exit_date": exit_date,
            "fold": int(row.fold), "model": strategy, "mode": "long_only",
            "side": 1, "entry_price": entry, "exit_price": exit_price,
            "gross_usd_per_oz": gross_usd_per_oz, "cost_usd_per_oz": cost,
            "net_usd_per_oz": gross_usd_per_oz - cost,
            "gross_return": gross_usd_per_oz / entry,
            "net_return": (gross_usd_per_oz - cost) / entry,
            "volatility_regime": row.volatility_regime,
        })
        next_signal_allowed = exit_date
    return pd.DataFrame(trades), candidates, skipped


def _summarize(trades, model, gate, cost, candidate_count, skipped):
    row = _summarize_trades(trades, model, "long_only", cost, candidate_count, skipped)
    row["filter"] = gate
    return row


def run(output_dir=OUT):
    raw = pd.read_csv(DATA_RAW_DIR / "xauusd_daily.csv", parse_dates=["datetime"])
    split_path = DATA_PROCESSED_DIR / "h5_execution_model_comparison" / "splits.csv"
    splits = pd.read_csv(split_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    predictions = build_prediction_table()
    predictions = add_signal_gates(predictions, raw, splits)
    weekday = filter_weekdays(raw).sort_values("datetime").drop_duplicates("datetime")
    prices = weekday.set_index("datetime")[["open", "close"]]

    all_trades, summary_rows, fold_rows = [], [], []
    results = [("AlwaysLong", filter_name) for filter_name in FILTERS]
    results += [(model, filter_name) for model in MODEL_CANDIDATES for filter_name in FILTERS]
    for model, gate in results:
        for cost in COSTS:
            trades, candidates, skipped = _simulate(predictions, model, gate, prices, cost)
            if not trades.empty:
                all_trades.append(trades.assign(filter=gate))
                ordered = trades.sort_values("signal_date")
                if not (ordered.entry_date.iloc[1:].to_numpy() > ordered.exit_date.iloc[:-1].to_numpy()).all():
                    raise ValueError(f"Overlapping positions for {model}/{gate}/cost={cost}")
            summary_rows.append(_summarize(trades, model, gate, cost, candidates, skipped))
            if not trades.empty:
                for fold, part in trades.groupby("fold", sort=True):
                    fold_row = _summarize(part, model, gate, cost, len(part), 0)
                    fold_row.pop("candidate_signals", None)
                    fold_row.pop("signals_skipped_while_in_trade", None)
                    fold_rows.append(fold_row | {"fold": int(fold)})

    summary = pd.DataFrame(summary_rows)
    baseline = summary[summary.model.eq("AlwaysLong")][
        ["filter", "cost_usd_per_oz", "net_compound_return", "max_drawdown_at_trade_exits", "n_trades"]
    ].rename(columns={
        "net_compound_return": "matched_always_long_net_return",
        "max_drawdown_at_trade_exits": "matched_always_long_max_drawdown",
        "n_trades": "matched_always_long_n_trades",
    })
    summary = summary.merge(baseline, on=["filter", "cost_usd_per_oz"], validate="many_to_one")
    summary["delta_net_return_vs_matched_always_long"] = (
        summary.net_compound_return - summary.matched_always_long_net_return
    )
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    fold_summary = pd.DataFrame(fold_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "signals_with_gates.csv", index=False)
    trades.to_csv(output_dir / "trades.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    fold_summary.to_csv(output_dir / "fold_summary.csv", index=False)
    metadata = {
        "experiment": "h5_fixed_trading_hypotheses_v1",
        "hypotheses": FILTERS,
        "models": ["AlwaysLong", *MODEL_CANDIDATES],
        "filters_are_signal_time_only": True,
        "volatility_cutoffs": "fold-local training terciles; 20 weekday return standard deviation",
        "trend_filter": "close above trailing SMA50",
        "momentum_filter": "20-observation close-to-close return > 0",
        "round_trip_cost_scenarios_usd_per_oz": list(COSTS),
        "selection_or_tuning": False,
        "holdout": False,
        "interpretation": "exploratory; hypotheses arose after inspecting development OOF results",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary, fold_summary, trades, predictions


if __name__ == "__main__":
    result = run()
    cols = ["model", "filter", "cost_usd_per_oz", "n_trades", "net_compound_return",
            "max_drawdown_at_trade_exits", "delta_net_return_vs_matched_always_long"]
    print(result[0].query("cost_usd_per_oz == 0.5")[cols].to_string(index=False))
