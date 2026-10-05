"""Cost-sensitivity backtest of frozen execution-aligned OOF predictions.

Predictions are generated after close(t), entered at open(t+1), and exited at
close(t+5). Trades are non-overlapping; hypothetical round-trip costs are
deducted in USD per ounce. This is a development diagnostic, not a holdout.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.features.calendar import filter_weekdays

MODEL_OOF = DATA_PROCESSED_DIR / "h5_execution_model_comparison" / "oof_predictions.csv"
FIVE_VOTER_OOF = DATA_PROCESSED_DIR / "h5_execution_ensemble_tuning" / "ensemble_oof_predictions.csv"
ARX_OOF = DATA_PROCESSED_DIR / "h5_execution_arx" / "arx_oof_predictions.csv"
OUT = DATA_PROCESSED_DIR / "h5_execution_cost_backtest"
COST_SCENARIOS_USD_PER_OZ = (0.0, 0.25, 0.50, 1.0)
CLASSIFIER_CANDIDATES = (
    "LogisticRegression", "ExtraTreesClassifier", "SVC",
    "Ensemble_RankEqual", "Ensemble_HardVote_5", "ARX_Ridge",
)


def build_prediction_table(model_path=MODEL_OOF, five_path=FIVE_VOTER_OOF, arx_path=ARX_OOF):
    base = pd.read_csv(model_path, parse_dates=["date", "entry_date", "label_end_date"])
    model_rows = base[base.model.isin(["LogisticRegression", "ExtraTreesClassifier", "SVC",
                                       "Ensemble_RankEqual"])].copy()
    if model_rows.duplicated(["date", "model"]).any():
        raise ValueError("Duplicate model/date rows in execution OOF predictions")
    if (model_rows.groupby("date").y_true.nunique().gt(1).any()
            or model_rows.groupby("date").forward_return.nunique().gt(1).any()):
        raise ValueError("Execution OOF targets/returns disagree between model rows")
    preds = model_rows.pivot(index="date", columns="model", values="y_pred").reset_index()
    meta = model_rows[model_rows.model.eq("LogisticRegression")][
        ["date", "entry_date", "label_end_date", "fold", "y_true", "forward_return"]
    ]
    preds = meta.merge(preds, on="date", validate="one_to_one")

    five = pd.read_csv(five_path, parse_dates=["date"])
    five = five[["date", "Ensemble_HardVote_5"]]
    arx = pd.read_csv(arx_path, parse_dates=["date"])
    arx = arx[["date", "arx_prediction"]].rename(columns={"arx_prediction": "ARX_Ridge"})
    preds = preds.merge(five, on="date", how="inner", validate="one_to_one")
    preds = preds.merge(arx, on="date", how="inner", validate="one_to_one")
    if len(preds) != len(meta):
        raise ValueError("OOF prediction files do not cover the same dates")
    for col in CLASSIFIER_CANDIDATES:
        if preds[col].isna().any():
            raise ValueError(f"Missing OOF predictions for {col}")
        preds[col] = preds[col].astype(int)
    if not preds.date.is_monotonic_increasing or preds.date.duplicated().any():
        raise ValueError("Prediction dates must be ordered and unique")
    return preds


def _trade_stream(predictions, model, mode, prices, cost_usd_per_oz):
    """Create a non-overlapping trade sequence using predictions only for entry."""
    next_signal_allowed = pd.Timestamp.min
    trades = []
    skipped_open_position = 0
    candidate_signals = 0
    skipped_by_fold = {}
    candidates_by_fold = {}
    for row in predictions.itertuples(index=False):
        signal_date = pd.Timestamp(row.date)
        pred = int(getattr(row, model)) if model != "AlwaysLong" else 1
        if mode == "long_only" and pred == 0:
            continue
        candidate_signals += 1
        candidates_by_fold[int(row.fold)] = candidates_by_fold.get(int(row.fold), 0) + 1
        if signal_date < next_signal_allowed:
            skipped_open_position += 1
            skipped_by_fold[int(row.fold)] = skipped_by_fold.get(int(row.fold), 0) + 1
            continue
        side = 1 if mode == "long_only" or pred == 1 else -1
        entry_date, exit_date = pd.Timestamp(row.entry_date), pd.Timestamp(row.label_end_date)
        try:
            entry = float(prices.loc[entry_date, "open"])
            exit_price = float(prices.loc[exit_date, "close"])
        except KeyError as exc:
            raise ValueError(f"Missing OHLC for scheduled trade date: {exc}") from exc
        if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0:
            raise ValueError("Trade prices must be finite and entry price positive")
        gross_usd_per_oz = side * (exit_price - entry)
        net_usd_per_oz = gross_usd_per_oz - cost_usd_per_oz
        gross_return = gross_usd_per_oz / entry
        net_return = net_usd_per_oz / entry
        if side == 1 and not np.isclose(gross_return, float(row.forward_return), rtol=1e-8, atol=1e-10):
            raise ValueError("OHLC-derived long return disagrees with the saved execution target")
        trades.append({
            "signal_date": signal_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "fold": int(row.fold),
            "model": model,
            "mode": mode,
            "side": side,
            "entry_price": entry,
            "exit_price": exit_price,
            "gross_usd_per_oz": gross_usd_per_oz,
            "cost_usd_per_oz": cost_usd_per_oz,
            "net_usd_per_oz": net_usd_per_oz,
            "gross_return": gross_return,
            "net_return": net_return,
            "y_true_for_audit_only": int(row.y_true),
            "predicted_direction": pred,
            "forward_return_for_audit_only": float(row.forward_return),
        })
        next_signal_allowed = exit_date
    return (pd.DataFrame(trades), candidate_signals, skipped_open_position,
            candidates_by_fold, skipped_by_fold)


def _summarize_trades(trades, model, mode, cost, candidate_signals, skipped):
    if trades.empty:
        return {
            "model": model, "mode": mode, "cost_usd_per_oz": cost,
            "n_trades": 0, "candidate_signals": candidate_signals,
            "signals_skipped_while_in_trade": skipped,
            "gross_compound_return": 0.0, "net_compound_return": 0.0,
            "max_drawdown_at_trade_exits": 0.0, "win_rate": np.nan,
            "profit_factor": np.nan, "trade_sharpe_annualized_approx": np.nan,
            "cagr_approx": np.nan,
        }
    gross = trades.gross_return.to_numpy(dtype=float)
    net = trades.net_return.to_numpy(dtype=float)
    gross_equity = np.cumprod(1 + gross)
    equity = np.cumprod(1 + net)
    equity_with_initial = np.r_[1.0, equity]
    peaks = np.maximum.accumulate(equity_with_initial)
    drawdowns = equity_with_initial / peaks - 1
    gains = net[net > 0].sum()
    losses = -net[net < 0].sum()
    std = np.std(net, ddof=1) if len(net) > 1 else np.nan
    sharpe = (np.mean(net) / std * np.sqrt(252 / 5)) if std and np.isfinite(std) else np.nan
    elapsed_days = max((trades.exit_date.max() - trades.entry_date.min()).days, 1)
    years = elapsed_days / 365.25
    cagr = equity[-1] ** (1 / years) - 1 if equity[-1] > 0 and years > 0 else np.nan
    return {
        "model": model, "mode": mode, "cost_usd_per_oz": cost,
        "n_trades": len(trades), "candidate_signals": candidate_signals,
        "signals_skipped_while_in_trade": skipped,
        "gross_compound_return": gross_equity[-1] - 1,
        "net_compound_return": equity[-1] - 1,
        "max_drawdown_at_trade_exits": drawdowns.min(),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": gains / losses if losses > 0 else np.inf,
        "trade_sharpe_annualized_approx": sharpe,
        "cagr_approx": cagr,
    }


def run(output_dir=OUT):
    raw_path = DATA_RAW_DIR / "xauusd_daily.csv"
    raw = pd.read_csv(raw_path, parse_dates=["datetime"])
    raw = filter_weekdays(raw).sort_values("datetime").drop_duplicates("datetime")
    prices = raw.set_index("datetime")[["open", "close"]]
    predictions = build_prediction_table()
    trades_all, summary_rows, fold_rows = [], [], []
    strategies = [(name, mode) for name in CLASSIFIER_CANDIDATES for mode in ("long_only", "long_short")]
    strategies.append(("AlwaysLong", "long_only"))

    for model, mode in strategies:
        for cost in COST_SCENARIOS_USD_PER_OZ:
            trades, candidate_signals, skipped, candidates_by_fold, skipped_by_fold = _trade_stream(
                predictions, model, mode, prices, cost,
            )
            if not trades.empty:
                trades_all.append(trades)
                ordered = trades.sort_values("signal_date")
                if not (ordered.entry_date.iloc[1:].to_numpy() >
                        ordered.exit_date.iloc[:-1].to_numpy()).all():
                    raise ValueError(f"Overlapping trades found for {model}/{mode}/cost={cost}")
            summary_rows.append(_summarize_trades(
                trades, model, mode, cost, candidate_signals, skipped,
            ))
            if not trades.empty:
                for fold, part in trades.groupby("fold", sort=True):
                    fold_rows.append(_summarize_trades(
                        part, model, mode, cost,
                        candidate_signals=candidates_by_fold.get(int(fold), 0),
                        skipped=skipped_by_fold.get(int(fold), 0),
                    ) | {"fold": int(fold)})

    trade_table = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    fold_summary = pd.DataFrame(fold_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "oof_prediction_table.csv", index=False)
    trade_table.to_csv(output_dir / "trades.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    fold_summary.to_csv(output_dir / "fold_summary.csv", index=False)
    metadata = {
        "experiment": "h5_execution_oof_cost_sensitivity_v1",
        "target": "signal after close(t); enter next weekday open; exit fifth subsequent weekday close",
        "calendar": "weekday proxy, no exchange holiday calendar",
        "positioning": "one unit of account notional, no leverage; no overlapping positions",
        "long_only": "UP opens long; DOWN remains in cash",
        "long_short": "UP long; DOWN short",
        "round_trip_cost_scenarios_usd_per_oz": list(COST_SCENARIOS_USD_PER_OZ),
        "costs_are_broker_quotes": False,
        "max_drawdown": "measured at trade exits only; no mark-to-market path available",
        "sharpe": "trade-return Sharpe annualized approximately with sqrt(252/5); not daily Sharpe",
        "selection_or_threshold_tuning": False,
        "holdout": False,
        "input_predictions_are_oof": True,
        "limitations": [
            "Development history has been used in prior model/feature selection.",
            "Cost scenarios are sensitivity analysis, not verified broker spread/slippage/commission.",
            "Weekdays are a session proxy and omit exchange holidays.",
            "Only trade-exit equity is used for drawdown; intratrade drawdown is unavailable.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary, fold_summary, trade_table, predictions


if __name__ == "__main__":
    result = run()
    print(result[0].to_string(index=False))
