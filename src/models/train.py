import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import confusion_matrix
from src.config import DATA_PROCESSED_DIR
from src.validation.splits import create_time_series_splits


df = pd.read_csv(DATA_PROCESSED_DIR / "xauusd_daily_features.csv")
df["datetime"]= pd.to_datetime(df["datetime"])

feature_columns=[
    'SMA_20',
    'RSI_14',
    'EMA_20',
    'MACD',
    'MACD_Signal',
    'MACD_HIST',
    'BB_UPPER',
    'BB_LOWER'
]

splits = create_time_series_splits(df,n_splits=5)
for split_index,(train_index, test_index) in enumerate(splits):

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    X_train = train_df[feature_columns]
    y_train = train_df["target"]

    X_test = test_df[feature_columns]
    y_test = test_df["target"]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    coefficients = pd.Series(model.coef_[0],index = feature_columns).sort_values()

    print(
        f"Split {split_index}: "
        f"accuracy={accuracy:.4f}, "
        f"precision={precision:.4f}, "
        f"recall={recall:.4f}, "
        f"f1={f1:.4f}"
    )
    print("\n")
    print(coefficients)
    print("\n")

    print(f"Confusion matrix {split_index} :\n{cm}")

