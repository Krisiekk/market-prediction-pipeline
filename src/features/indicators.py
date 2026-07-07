import pandas as pd
from pandas.core.interchange import column


def add_sma(df, column="close", window =20):
    df = df.copy()
    col_name = f"SMA_{window}"
    df[col_name]=df[column].rolling(window).mean()



    return df

def  add_rsi(df, column="close", window=14):
    df = df.copy()
    delta = df[column].diff()

    col_name = f"RSI_{window}"

    gain = delta.where(delta > 0, 0)
    lose = delta.where(delta < 0, 0).abs()

    avg_gain = gain.rolling(window).mean()
    avg_lose = lose.rolling(window).mean()

    rs = avg_gain / avg_lose


    df[col_name]= 100 - 100 / (1 + rs)

    return df




