"""Evaluate a regularized ARX-style return regression and its ensemble vote.

The model predicts the continuous execution-aligned return using lagged XAU
returns (autoregressive inputs) and contemporaneous technical/intermarket
features (exogenous inputs). Ridge alpha is selected inside each outer fold.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.models.compare_h5_execution_models import (
    GAP,
    N_SPLITS,
    prepare_execution_dataset,
)
from src.models.tune_h5_execution_ensemble import _inner_splits
from src.validation.splits import create_time_series_splits

BASE_RESULTS = DATA_PROCESSED_DIR / "h5_execution_ensemble_tuning"
OUT = DATA_PROCESSED_DIR / "h5_execution_arx"
BASE_VOTERS = (
    "LogisticRegression", "ExtraTreesClassifier", "SVC",
    "RandomForestClassifier", "HistGradientBoostingClassifier",
)
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def _regression_metrics(y_return, y_pred_return):
    y_direction = (np.asarray(y_return) > 0).astype(int)
    y_pred_direction = (np.asarray(y_pred_return) > 0).astype(int)
    return {
        "roc_auc": roc_auc_score(y_direction, y_pred_return),
        "directional_accuracy": accuracy_score(y_direction, y_pred_direction),
        "balanced_accuracy": balanced_accuracy_score(y_direction, y_pred_direction),
        "mae": mean_absolute_error(y_return, y_pred_return),
        "rmse": float(np.sqrt(mean_squared_error(y_return, y_pred_return))),
        "r2": r2_score(y_return, y_pred_return),
    }


def _classifier_metrics(y, prediction):
    return {
        "roc_auc": np.nan,
        "directional_accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
    }


def _load_five_voter_predictions(path, expected_dates):
    frame = pd.read_csv(path, parse_dates=["date"])
    needed = {"date", "fold", "y_true", "forward_return", *BASE_VOTERS,
              "Ensemble_HardVote_5"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"Five-voter OOF file is missing columns: {sorted(missing)}")
    if frame.duplicated("date").any():
        raise ValueError("Five-voter OOF file has duplicate dates")
    if set(frame.date) != set(expected_dates):
        raise ValueError("ARX outer-fold dates do not match the five-voter OOF dates")
    return frame


def _combined_vote(frame):
    base_votes = frame[list(BASE_VOTERS)].to_numpy(dtype=int)
    arx_vote = frame.arx_prediction.to_numpy(dtype=int)
    votes = np.column_stack([base_votes, arx_vote])
    count_up = votes.sum(axis=1)
    # With six equally weighted voters, a 3-3 tie is an abstention/no-trade.
    combined = np.where(count_up > 3, 1, np.where(count_up < 3, 0, np.nan))
    return count_up, combined, count_up != 3


def run(base_results=BASE_RESULTS, output_dir=OUT):
    raw_path = DATA_RAW_DIR / "xauusd_daily.csv"
    raw = pd.read_csv(raw_path, parse_dates=["datetime"])
    data, features, ar_features, _ = prepare_execution_dataset(raw)
    model_features = [*features, *ar_features]
    splits = create_time_series_splits(data, n_splits=N_SPLITS, gap=GAP)
    expected_test_dates = [data.iloc[test_idx].datetime for _, test_idx in splits]
    expected_dates = pd.concat(expected_test_dates, ignore_index=True)
    five = _load_five_voter_predictions(
        Path(base_results) / "ensemble_oof_predictions.csv", expected_dates,
    )

    predictions, tuning_rows = [], []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train = data.iloc[train_idx].reset_index(drop=True)
        test = data.iloc[test_idx].reset_index(drop=True)
        if not train.label_end_date.max() < test.datetime.min():
            raise ValueError(f"Outer fold {fold}: training labels overlap validation dates")
        inner = _inner_splits(train)
        estimator = make_pipeline(StandardScaler(), Ridge())
        search = GridSearchCV(
            estimator,
            {"ridge__alpha": RIDGE_ALPHAS},
            scoring="neg_mean_squared_error",
            cv=inner,
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(train[model_features], train.forward_return)
        forecast = search.best_estimator_.predict(test[model_features])
        score = _regression_metrics(test.forward_return.to_numpy(), forecast)
        tuning_rows.append({
            "fold": fold,
            "best_alpha": float(search.best_params_["ridge__alpha"]),
            "best_inner_mean_mse": float(-search.best_score_),
            "n_inner_folds": len(inner),
            "n_train": len(train),
            "n_test": len(test),
            "train_label_end": str(train.label_end_date.max().date()),
            "test_start": str(test.datetime.min().date()),
            **score,
        })
        predictions.append(pd.DataFrame({
            "date": test.datetime,
            "fold": fold,
            "y_true": test.target,
            "forward_return": test.forward_return,
            "arx_predicted_return": forecast,
            "arx_prediction": (forecast > 0).astype(int),
        }))
        print(f"fold={fold} ARX alpha={search.best_params_['ridge__alpha']} "
              f"outer_auc={score['roc_auc']:.4f} outer_mae={score['mae']:.6f}", flush=True)

    arx = pd.concat(predictions, ignore_index=True)
    joined = five.merge(arx, on=["date", "fold", "y_true"],
                        how="inner", suffixes=("_base", "_arx"), validate="one_to_one")
    if len(joined) != len(arx):
        raise ValueError("ARX and five-voter OOF predictions did not align")
    if not np.allclose(joined.forward_return_base, joined.forward_return_arx,
                       rtol=1e-9, atol=1e-12):
        raise ValueError("Forward returns disagree between ARX and base OOF predictions")
    joined["forward_return"] = joined.forward_return_arx
    count_up, combined, _ = _combined_vote(joined)
    joined["six_voter_up_count"] = count_up
    joined["Ensemble_HardVote_6"] = combined

    fold_rows = []
    for fold, part in joined.groupby("fold", sort=True):
        y = part.y_true.to_numpy(dtype=int)
        fold_rows.append({"fold": int(fold), "model": "ARX_Ridge", "n_test": len(part),
                          **_regression_metrics(part.forward_return, part.arx_predicted_return)})
        fold_rows.append({"fold": int(fold), "model": "Ensemble_HardVote_5", "n_test": len(part),
                          **_classifier_metrics(y, part.Ensemble_HardVote_5)})
        mask = part.six_voter_up_count.to_numpy() != 3
        if mask.any():
            strict_pred = part.Ensemble_HardVote_6.to_numpy()[mask].astype(int)
            fold_rows.append({"fold": int(fold), "model": "Ensemble_HardVote_6_abstain_on_tie",
                              "n_test": int(mask.sum()), "coverage": float(mask.mean()),
                              **_classifier_metrics(y[mask], strict_pred)})
            fold_rows.append({"fold": int(fold), "model": "Ensemble_HardVote_5_same_coverage",
                              "n_test": int(mask.sum()), "coverage": float(mask.mean()),
                              **_classifier_metrics(y[mask], part.Ensemble_HardVote_5.to_numpy()[mask])})
            fold_rows.append({"fold": int(fold), "model": "AlwaysUp_same_coverage",
                              "n_test": int(mask.sum()), "coverage": float(mask.mean()),
                              **_classifier_metrics(y[mask], np.ones(mask.sum(), dtype=int))})

    fold_metrics = pd.DataFrame(fold_rows)
    rows = []
    for model, group in fold_metrics.groupby("model", sort=False):
        rows.append({
            "model": model,
            "mean_auc": group.roc_auc.mean(),
            "std_auc": group.roc_auc.std(),
            "folds_auc_above_05": int((group.roc_auc > 0.5).sum()),
            "mean_directional_accuracy": group.directional_accuracy.mean(),
            "mean_balanced_accuracy": group.balanced_accuracy.mean(),
            "mean_mae": group.mae.mean(),
            "mean_rmse": group.rmse.mean(),
            "mean_r2": group.r2.mean(),
            "mean_coverage": group.coverage.mean() if "coverage" in group else 1.0,
        })
    summary = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arx.to_csv(output_dir / "arx_oof_predictions.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(output_dir / "fold_tuning_results.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    joined.to_csv(output_dir / "ensemble_oof_predictions.csv", index=False)
    metadata = {
        "experiment": "h5_execution_arx_ridge_v1",
        "model": "Ridge regression with autoregressive lag returns and exogenous technical/intermarket features",
        "target": "continuous open(t+1) to close(t+5) return; signal after close(t)",
        "feature_set": "TECHNICAL + A+C + weekday return lags",
        "alpha_candidates": RIDGE_ALPHAS,
        "inner_selection_metric": "mean squared error on expanding TimeSeriesSplit, gap=5",
        "outer_folds": N_SPLITS,
        "base_voters": list(BASE_VOTERS),
        "tie_rule": "a 3-3 tie abstains; compare against five-voter and AlwaysUp on the same covered dates",
        "holdout": False,
        "transaction_costs": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary, fold_metrics, arx, pd.DataFrame(tuning_rows), joined


if __name__ == "__main__":
    result = run()
    print(result[0].to_string(index=False))
