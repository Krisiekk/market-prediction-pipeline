

import pandas as pd
import importlib
from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR

from src.features import indicators

importlib.reload(indicators)

def build_features(df, horizon =1):
    df = df.copy()

    df = indicators.add_all_indicators(df)

    feature_close = df["close"].shift(-horizon)
    df["target"]=(feature_close>df["close"]).astype("Int64")
    df.loc[feature_close.isna(), "target"] = pd.NA


    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    input_path = DATA_RAW_DIR / "xauusd_daily.csv"
    output_path = DATA_PROCESSED_DIR / "xauusd_daily_features.csv"

    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    df_features = build_features(df,horizon =1)

    df_features.to_csv(output_path,index=False)

    print(f"saved {len(df_features)} rowsto {output_path}")
