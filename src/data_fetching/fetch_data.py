import os
import requests
from dotenv import load_dotenv
import pandas as pd
from src.config import DATA_RAW_DIR


# =========================
# Ścieżki
# =========================

XAUUSD_OUTPUT_PATH = DATA_RAW_DIR / "xauusd_daily.csv"
DGS10_OUTPUT_PATH = DATA_RAW_DIR / "dgs10_daily.csv"
DFII10_OUTPUT_PATH = DATA_RAW_DIR / "dfii10_daily.csv"
USD_BROAD_OUTPUT_PATH = DATA_RAW_DIR / "usd_broad_daily.csv"
DXY_OUTPUT_PATH = DATA_RAW_DIR / "dxy_reconstructed_daily.csv"


# =========================
# API keys
# =========================

load_dotenv()

api_key = os.getenv("TWELVE_DATA_API_KEY")
fred_api_key = os.getenv("FRED_API_KEY")

if not api_key:
    raise ValueError("TWELVE_DATA_API_KEY environment variable not set")

if not fred_api_key:
    raise ValueError("FRED_API_KEY environment variable not set")

api_key = api_key.strip()
fred_api_key = fred_api_key.strip()


# =========================
# Twelve Data
# =========================

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"


def fetch_twelve_data_daily(symbol):
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 5000,
        "apikey": api_key,
    }

    response = requests.get(
        TWELVE_DATA_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise ValueError(
            f"Twelve Data error for {symbol}: {data}"
        )

    pair_df = pd.DataFrame(data["values"])

    pair_df["datetime"] = pd.to_datetime(
        pair_df["datetime"]
    )

    pair_df = pair_df.sort_values("datetime")

    return pair_df


# =========================
# XAU/USD
# =========================

df = fetch_twelve_data_daily("XAU/USD")

numeric_columns = [
    "open",
    "high",
    "low",
    "close"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df.to_csv(
    XAUUSD_OUTPUT_PATH,
    index=False
)

print(
    f"saved XAU/USD: {len(df)} rows -> "
    f"{XAUUSD_OUTPUT_PATH}"
)


# =========================
# DXY - rekonstrukcja
# =========================

DXY_PAIRS = {
    "EUR/USD": -0.576,
    "USD/JPY": 0.136,
    "GBP/USD": -0.119,
    "USD/CAD": 0.091,
    "USD/SEK": 0.042,
    "USD/CHF": 0.036,
}

dxy_df = None

for symbol, exponent in DXY_PAIRS.items():

    print(f"downloading {symbol}...")

    pair_df = fetch_twelve_data_daily(symbol)

    pair_df = pair_df[
        ["datetime", "close"]
    ].copy()

    pair_df["close"] = pd.to_numeric(
        pair_df["close"],
        errors="coerce"
    )

    column_name = symbol.replace("/", "_")

    pair_df = pair_df.rename(
        columns={"close": column_name}
    )

    if dxy_df is None:
        dxy_df = pair_df
    else:
        dxy_df = dxy_df.merge(
            pair_df,
            on="datetime",
            how="inner"
        )


# Formuła DXY

dxy_df["dxy"] = 50.14348112

for symbol, exponent in DXY_PAIRS.items():

    column_name = symbol.replace("/", "_")

    dxy_df["dxy"] *= (
        dxy_df[column_name] ** exponent
    )


dxy_df = dxy_df[
    ["datetime", "dxy"]
].sort_values("datetime")


dxy_df.to_csv(
    DXY_OUTPUT_PATH,
    index=False
)

print(
    f"saved reconstructed DXY: "
    f"{len(dxy_df)} rows -> "
    f"{DXY_OUTPUT_PATH}"
)


# =========================
# FRED
# =========================

FRED_BASE_URL = (
    "https://api.stlouisfed.org/"
    "fred/series/observations"
)


def fetch_fred_series(
    series_id,
    output_path,
    start_date="2007-01-01"
):

    params = {
        "series_id": series_id,
        "api_key": fred_api_key,
        "file_type": "json",
        "observation_start": start_date,
    }

    response = requests.get(
        FRED_BASE_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "observations" not in data:
        raise ValueError(
            f"FRED error for {series_id}: {data}"
        )

    observations = data["observations"]

    fred_df = pd.DataFrame(
        observations
    )[["date", "value"]]

    fred_df["date"] = pd.to_datetime(
        fred_df["date"]
    )

    fred_df["value"] = pd.to_numeric(
        fred_df["value"],
        errors="coerce"
    )

    fred_df = fred_df.sort_values("date")

    fred_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"saved {series_id}: "
        f"{len(fred_df)} rows -> "
        f"{output_path}"
    )

    return fred_df


# =========================
# Dane FRED
# =========================

# Nominalne US 10Y
fetch_fred_series(
    "DGS10",
    DGS10_OUTPUT_PATH
)

# Realne US 10Y
fetch_fred_series(
    "DFII10",
    DFII10_OUTPUT_PATH
)

# Broad Dollar Index
fetch_fred_series(
    "DTWEXBGS",
    USD_BROAD_OUTPUT_PATH
)