import pandas as pd
from src.models.baseline import always_up_baseline, previous_day_direction_baseline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import confusion_matrix
from src.config import DATA_PROCESSED_DIR
from src.validation.splits import create_time_series_splits
from xgboost import XGBClassifier


def add_baseline_result(results,horizon,model_name,split_index,accuracy):
    results.append({
    "model": model_name,
    "horizon": horizon,
    "split": split_index,
    "accuracy": accuracy,
    "precision": None,
    "recall": None,
    "f1": None
    } )

def add_result(results, model_name, split_index, y_test, y_pred,horizon):
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "model": model_name,
        "horizon": horizon,
        "split": split_index,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    })

    return cm




results = []
results_h1 =[]

df = pd.read_csv(DATA_PROCESSED_DIR / "xauusd_daily_features_h5.csv")
df_h1 = pd.read_csv(DATA_PROCESSED_DIR / "xauusd_daily_features.csv")
df_h1["datetime"]= pd.to_datetime(df_h1["datetime"])
df["datetime"]= pd.to_datetime(df["datetime"])

feature_columns=[
    'SMA_20',
    'RSI_14',
    'EMA_20',
    'MACD',
    'MACD_Signal',
    'MACD_HIST',
    'BB_UPPER',
    'BB_LOWER',
    'PRICE_TO_SMA20',
    'PRICE_TO_EMA20',
    'BB_WIDTH',
    'BB_POSITION',
    'RETURN_1D',
    'RETURN_5D',
]


splits = create_time_series_splits(df,n_splits=5)
splits_h1= create_time_series_splits(df_h1,n_splits=5)
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

    cm = add_result(results,"LogisticRegression",split_index,y_test,y_pred,horizon=5)

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

    rf_model = RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced")

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)

    rf_accuracy = accuracy_score(y_test,rf_pred)
    rf_precision = precision_score(y_test,rf_pred)
    rf_recall = recall_score(y_test,rf_pred)
    rf_f1 = f1_score(y_test,rf_pred)

    rf_cm = add_result( results, "RandomForest",  split_index,y_test, rf_pred,horizon=5)
    print(

        f"Random Forest split {split_index}: "
        f"accuracy={rf_accuracy:.4f}, "
        f"precision={rf_precision:.4f}, "
        f"recall={rf_recall:.4f}, "
        f"f1={rf_f1:.4f}"

    )

for split_index,(train_index, test_index) in enumerate(splits_h1):
    train_df_h1 = df_h1.iloc[train_index]
    test_df_h1 =df_h1.iloc[test_index]

    X_train = train_df_h1[feature_columns]
    y_train = train_df_h1["target"]

    X_test = test_df_h1[feature_columns]
    y_test = test_df_h1["target"]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled
                           )
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = add_result(results_h1, "LogisticRegression",split_index,y_test,y_pred,horizon=1)

    rf_model = RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced")
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)

    rf_accuracy = accuracy_score(y_test,rf_pred)
    rf_precision = precision_score(y_test,rf_pred)
    rf_recall = recall_score(y_test,rf_pred)
    rf_f1 = f1_score(y_test,rf_pred)

    rf_cm = add_result(results_h1,"RandomForest",split_index,y_test,rf_pred,horizon=1)





results_baseline =[]
for split_index,(train_index, test_index) in enumerate(splits):

    test_h5_bs = df.iloc[test_index]
    y_test_h5 = test_h5_bs["target"]

    y_pred_h5_bs = always_up_baseline(len(test_h5_bs))
    y_pred_h5 =previous_day_direction_baseline(test_h5_bs)

    accuracy_h5_bs = accuracy_score(y_test_h5,y_pred_h5_bs)
    accuracy_h5 =accuracy_score(y_test_h5,y_pred_h5)

    add_result(results_baseline, "alwaysUpBaseline", split_index, y_test_h5, y_pred_h5_bs, horizon=5)
    add_result(results_baseline, "previousDayDirectionBaseline", split_index, y_test_h5, y_pred_h5, horizon=5)


for split_index,(train_index, test_index) in enumerate(splits_h1):

    test_h1_bs = df_h1.iloc[test_index]
    y_test_h1 = test_h1_bs["target"]

    y_pred_h1_bs = always_up_baseline(len(test_h1_bs))
    y_pred_h1 =previous_day_direction_baseline(test_h1_bs)

    accuracy_h1_bs = accuracy_score(y_test_h1,y_pred_h1_bs)
    accuracy_h1 =accuracy_score(y_test_h1,y_pred_h1)

    add_result(results_baseline, "alwaysUpBaseline", split_index, y_test_h1, y_pred_h1_bs, horizon=1)
    add_result(results_baseline, "previousDayDirectionBaseline", split_index, y_test_h1, y_pred_h1, horizon=1)







xg_model = XGBClassifier(
    n_estimators=200,
    max_depth=2,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,

)
for split_index,(train_index, test_index) in enumerate(splits_h1):
    test_h1 = df_h1.iloc[test_index]
    train_h1 = df_h1.iloc[train_index]

    X_train = train_h1[feature_columns]
    y_train = train_h1["target"]

    X_test = test_h1[feature_columns]
    y_test = test_h1["target"]

    xg_model.fit(X_train, y_train)
    y_pred = xg_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    add_result(results_h1, "XGBoost", split_index, y_test, y_pred, horizon=1)


for split_index,(train_index, test_index) in enumerate(splits):

    test = df.iloc[test_index]
    train = df.iloc[train_index]

    X_train = train[feature_columns]
    y_train = train["target"]

    X_test = test[feature_columns]
    y_test = test["target"]

    xg_model.fit(X_train, y_train)
    y_pred = xg_model.predict(X_test)

    add_result(results, "XGBoost", split_index, y_test, y_pred, horizon=5)





results_bs_df = pd.DataFrame(results_baseline)
results_df_path = DATA_PROCESSED_DIR / "baseline_results.csv"
results_bs_df.to_csv(results_df_path, index=False)

results_df_h1 =pd.DataFrame(results_h1)
results_df = pd.DataFrame(results)
results_path_h1 =DATA_PROCESSED_DIR / "model_results_after_relative_features.csv"
results_path = DATA_PROCESSED_DIR / "model_results_after_relative_feature_h5.csv"
results_df.to_csv(results_path, index=False)
results_df_h1.to_csv(results_path_h1, index=False)