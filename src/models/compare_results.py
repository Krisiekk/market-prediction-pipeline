from pathlib import Path
import pandas as pd
from src.config import DATA_RAW_DIR,DATA_PROCESSED_DIR

baseline_results =pd.read_csv(DATA_PROCESSED_DIR / "baseline_results.csv")

models_h1_results =pd.read_csv(DATA_PROCESSED_DIR / "model_results_after_relative_features.csv")

models_h5_results =pd.read_csv(DATA_PROCESSED_DIR / "model_results_after_relative_feature_h5.csv")

print(baseline_results)
print(models_h1_results)
print(models_h5_results)

all_results =pd.concat([baseline_results,models_h1_results,models_h5_results],ignore_index=True)

all_results =all_results.groupby(["horizon","model"]).agg(
    mean_accuracy=("accuracy","mean"),
    mean_precision=("precision","mean"),
    mean_recall=("recall","mean"),
    mean_f1=("f1","mean"),
    std_accuracy=("accuracy","std")
)

print(all_results.to_string())

summary_path = DATA_PROCESSED_DIR / "summary_result.csv"
all_results.to_csv(summary_path)





