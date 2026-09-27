import pandas as pd

from src.config import DATA_RAW_DIR


DXY_PATH = DATA_RAW_DIR / "dxy_reconstructed_daily.csv"
DGS10_PATH = DATA_RAW_DIR / "dgs10_daily.csv"
DFII10_PATH = DATA_RAW_DIR / "dfii10_daily.csv"
USD_BROAD_PATH = DATA_RAW_DIR / "usd_broad_daily.csv"


def _load_dxy():
    df = pd.read_csv(DXY_PATH)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["dxy"] = pd.to_numeric(df["dxy"], errors="coerce")

    return df[["datetime", "dxy"]].sort_values("datetime")


def _load_fred(path, column_name):
    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "date": "datetime",
            "value": column_name,
        }
    )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df[column_name] = pd.to_numeric(
        df[column_name],
        errors="coerce",
    )

    return df[
        ["datetime", column_name]
    ].sort_values("datetime")


def _merge_last_known(left_df, right_df, value_column):
    return pd.merge_asof(
        left_df.sort_values("datetime"),
        right_df[
            ["datetime", value_column]
        ].sort_values("datetime"),
        on="datetime",
        direction="backward",
    )


def add_intermarket_features(df):
    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Wczytanie danych z innych rynków
    dxy = _load_dxy()

    dgs10 = _load_fred(
        DGS10_PATH,
        "dgs10",
    )

    dfii10 = _load_fred(
        DFII10_PATH,
        "dfii10",
    )

    usd_broad = _load_fred(
        USD_BROAD_PATH,
        "usd_broad",
    )

    # ========================================================
    # Łączenie danych z XAU/USD
    # ========================================================

    # Bierzemy ostatnią dostępną obserwację.
    # Nie pobieramy wartości z przyszłej daty.

    df = _merge_last_known(
        df,
        dxy,
        "dxy",
    )

    df = _merge_last_known(
        df,
        dgs10,
        "dgs10",
    )

    df = _merge_last_known(
        df,
        dfii10,
        "dfii10",
    )

    df = _merge_last_known(
        df,
        usd_broad,
        "usd_broad",
    )

    # ========================================================
    # DXY
    # ========================================================

    df["dxy_return_1d"] = (
        df["dxy"].pct_change(1)
    )

    df["dxy_return_5d"] = (
        df["dxy"].pct_change(5)
    )

    df["dxy_vs_sma20"] = (
        df["dxy"]
        / df["dxy"].rolling(20).mean()
        - 1
    )

    # ========================================================
    # US 10Y nominal yield
    # ========================================================

    # Rentowności są już wartościami procentowymi,
    # dlatego liczymy różnicę, a nie pct_change().

    df["dgs10_change_1d"] = (
        df["dgs10"].diff(1)
    )

    df["dgs10_change_5d"] = (
        df["dgs10"].diff(5)
    )

    # ========================================================
    # US 10Y real yield
    # ========================================================

    df["dfii10_change_1d"] = (
        df["dfii10"].diff(1)
    )

    df["dfii10_change_5d"] = (
        df["dfii10"].diff(5)
    )

    # ========================================================
    # 10Y breakeven inflation
    # ========================================================

    # Przybliżenie:
    #
    # nominalna rentowność - realna rentowność

    df["breakeven_10y"] = (
        df["dgs10"]
        - df["dfii10"]
    )

    df["breakeven_change_1d"] = (
        df["breakeven_10y"].diff(1)
    )

    df["breakeven_change_5d"] = (
        df["breakeven_10y"].diff(5)
    )

    # ========================================================
    # Broad U.S. Dollar Index
    # ========================================================

    df["usd_broad_return_1d"] = (
        df["usd_broad"].pct_change(1)
    )

    df["usd_broad_return_5d"] = (
        df["usd_broad"].pct_change(5)
    )

    return df