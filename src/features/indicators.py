import pandas as pd



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
    loss = delta.where(delta < 0, 0).abs()

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss


    df[col_name]= 100 - 100 / (1 + rs)

    return df

def add_ema(df, column="close", window=20):
    df = df.copy()

    col_name = f"EMA_{window}"
    df[col_name]=df[column].ewm(span=window,adjust = False).mean()
    return df


def add_macd(df,column= "close",fast =12,slow =26, signal=9):
    df = df.copy()

    ema_fast =  df[column].ewm(span = fast,adjust = False).mean()
    ema_slow = df[column].ewm(span = slow,adjust = False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal,adjust = False).mean()

    df["MACD_HIST"]= df["MACD"] -df["MACD_Signal"]

    return df

def add_bollinger_bands(df,column="close", window=20,num_std =2):
    df = df.copy()

    middle_band = df[column].rolling(window).mean()
    rolling_band= df[column].rolling(window).std()

    df["BB_MIDDLE"] =middle_band
    df["BB_UPPER"] = middle_band + rolling_band * num_std
    df["BB_LOWER"] = middle_band - rolling_band * num_std

    return df

def add_all_indicators(df):

    df = df.copy()
    df= add_sma(df,window=20)
    df= add_rsi(df,window=14)
    df= add_ema(df,column="close",window=20)
    df = add_macd(df,column="close",fast=12,slow=26,signal=9)
    df = add_bollinger_bands(df,column="close",window=20)

    return df