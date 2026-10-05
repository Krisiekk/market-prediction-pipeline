import pandas as pd
import importlib

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.features import indicators, intermarket
from src.features.calendar import filter_weekdays

importlib.reload(indicators)
importlib.reload(intermarket)


def build_features(df, horizon=1, intermarket_lag=1):
    """Filter weekdays before all features, lag macro, then build targets before dropna.

    Default lag=1 matches benchmark and exploration; this is not a guarantee of FRED publication timing.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(intermarket_lag, int) or intermarket_lag < 0:
        raise ValueError("intermarket_lag must be a nonnegative integer")
    df = filter_weekdays(df)

    # Cechy techniczne
    df = indicators.add_all_indicators(df)

    # Cechy makro / intermarket
    technical_columns = set(df.columns)
    df = intermarket.add_intermarket_features(df)
    intermarket_columns = [c for c in df.columns if c not in technical_columns]
    if intermarket_lag:
        df[intermarket_columns] = df[intermarket_columns].shift(intermarket_lag)

    # Target
    future_close = df["close"].shift(-horizon)
    # Diagnostics only: explicitly excluded from model X.
    df["label_end_date"] = df["datetime"].shift(-horizon)
    df["future_close"] = future_close

    df["target"] = (
        future_close > df["close"]
    ).astype("Int64")

    df.loc[
        future_close.isna(),
        "target"
    ] = pd.NA

    df = df.dropna().reset_index(drop=True)

    return df


def rebuild_datasets():
    """Rebuild both standard ML CSVs from preserved RAW; report the calendar."""
    import json
    raw = pd.read_csv(DATA_RAW_DIR / "xauusd_daily.csv", parse_dates=["datetime"])
    weekdays = filter_weekdays(raw)
    report = {"calendar": "observed_weekdays_v1", "raw_rows": len(raw),
              "saturday_removed": int((raw.datetime.dt.dayofweek == 5).sum()),
              "sunday_removed": int((raw.datetime.dt.dayofweek == 6).sum()),
              "total_weekend_removed": len(raw) - len(weekdays),
              "rows_after_calendar_filter": len(weekdays),
              "raw_start": str(raw.datetime.min()), "raw_end": str(raw.datetime.max()),
              "filtered_start": str(weekdays.datetime.min()), "filtered_end": str(weekdays.datetime.max())}
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    for horizon, filename in [(1, "xauusd_daily_features.csv"), (5, "xauusd_daily_features_h5.csv")]:
        data = build_features(raw, horizon=horizon, intermarket_lag=1)
        data.to_csv(DATA_PROCESSED_DIR / filename, index=False)
        report[f"final_h{horizon}_observations"] = len(data)
        report[f"h{horizon}_start"] = str(data.datetime.min())
        report[f"h{horizon}_end"] = str(data.datetime.max())
        sample = data.loc[data.datetime.isin(pd.to_datetime(["2025-04-24", "2026-09-17", "2026-09-18"])),
                          ["datetime", "close", "label_end_date", "future_close", "target"]].copy()
        sample["horizon"] = horizon
        sample["weekday"] = sample.datetime.dt.day_name()
        sample["label_end_weekday"] = sample.label_end_date.dt.day_name()
        examples.append(sample)
    output = DATA_PROCESSED_DIR / "weekday_calendar"
    output.mkdir(parents=True, exist_ok=True)
    (output / "filtering_report.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.concat(examples).to_csv(output / "target_examples.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    rebuild_datasets()
