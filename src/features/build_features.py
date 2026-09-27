import pandas as pd
import importlib

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.features import indicators, intermarket

importlib.reload(indicators)
importlib.reload(intermarket)


def build_features(df, horizon=1):
    df = df.copy()

    # Cechy techniczne
    df = indicators.add_all_indicators(df)

    # Cechy makro / intermarket
    df = intermarket.add_intermarket_features(df)

    # Target
    future_close = df["close"].shift(-horizon)

    df["target"] = (
        future_close > df["close"]
    ).astype("Int64")

    df.loc[
        future_close.isna(),
        "target"
    ] = pd.NA

    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    horizon = 5

    input_path = DATA_RAW_DIR / "xauusd_daily.csv"
    output_path = (
        DATA_PROCESSED_DIR
        / f"xauusd_daily_features_h{horizon}.csv"
    )

    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    df_features = build_features(
        df,
        horizon=horizon
    )

    df_features.to_csv(
        output_path,
        index=False
    )

    print(
        f"saved {len(df_features)} rows to "
        f"{output_path}"
    )