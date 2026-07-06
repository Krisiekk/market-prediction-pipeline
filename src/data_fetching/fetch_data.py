import os
import requests
from dotenv import load_dotenv
import pandas as pd
from src.config import DATA_RAW_DIR

OUTPUT_PATH = DATA_RAW_DIR / "xauusd_daily.csv"

load_dotenv()

api_key = os.getenv("TWELVE_DATA_API_KEY").strip()


if api_key is None:
    raise ValueError("TWELVE_DATA_API_KEY environment variable not set")


BASE_URL = "https://api.twelvedata.com/time_series"

params = {
    "symbol":"XAU/USD",
    "interval":"1day",
    "outputsize":5000,
    "apikey":api_key


}

response = requests.get(BASE_URL, params=params)

if response.status_code != 200:
    print(f"request failed with status code {response.status_code}")
    print(response.text)

else:
    data = response.json()

    print(data.keys())

    data_list = data["values"]
    print(f"{len(data_list)} rows")
    print(data_list[0])


df = pd.DataFrame(data_list)



df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values(by="datetime")


numeric_columns =["open", "high", "low", "close"]

for col in numeric_columns:
    df[col]= pd.to_numeric(df[col], errors="coerce")


df.to_csv(OUTPUT_PATH, index=False)

print("saved  to data/raw/xauusd_daily.csv")