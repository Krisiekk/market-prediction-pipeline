"""Project calendar: observed Monday–Friday rows, not a holiday calendar."""
import pandas as pd


def filter_weekdays(df):
    """Preserve RAW; return sorted, unique weekday observations on a fresh index."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], errors='raise')
    if df.datetime.isna().any():
        raise ValueError('Missing observation date')
    df = df.sort_values('datetime')
    df = df.loc[df.datetime.dt.dayofweek < 5]
    if df.datetime.dt.normalize().duplicated().any():
        raise ValueError('Duplicate daily observation dates')
    return df.reset_index(drop=True)
