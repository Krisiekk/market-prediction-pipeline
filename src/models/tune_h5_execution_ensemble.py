"""Nested walk-forward tuning of two extra H5 execution-target voters.

The existing execution-target OOF predictions for Logistic Regression,
ExtraTrees, and SVC are reused. Random Forest and HistGradientBoosting are
tuned only inside each outer training fold, then evaluated on its untouched
outer validation fold. No holdout or transaction-cost backtest is performed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.models.compare_h5_execution_models import (
    GAP,
    N_SPLITS,
    prepare_execution_dataset,
)
from src.validation.splits import create_time_series_splits

SOURCE = DATA_PROCESSED_DIR / "h5_execution_model_comparison" / "oof_predictions.csv"
OUT = DATA_PROCESSED_DIR / "h5_execution_ensemble_tuning"
BASE_VOTERS = ("LogisticRegression", "ExtraTreesClassifier", "SVC")
TUNED_VOTERS = ("RandomForestClassifier", "HistGradientBoostingClassifier")


def make_searches():
    """Return deliberately small, predeclared parameter searches."""
    return {
        "RandomForestClassifier": (
            RandomForestClassifier(random_state=42, n_jobs=1),
            {
                "n_estimators": [200, 400],
                "max_depth": [8, None],
                "min_samples_leaf": [5, 10],
                "max_features": ["sqrt", 0.5],
            },
        ),
        "HistGradientBoostingClassifier": (
            HistGradientBoostingClassifier(random_state=42, early_stopping=False),
            {
                "learning_rate": [0.03, 0.05],
                "max_iter": [100, 200],
                "max_leaf_nodes": [7, 15],
                "min_samples_leaf": [10, 20],
                "l2_regularization": [0.1, 1.0],
            },
        ),
    }


def _inner_splits(train):
    """Chronological CV splits with an explicit label-overlap purge check."""
    splitter = TimeSeriesSplit(n_splits=3, gap=GAP)
    splits = list(splitter.split(train))
    for inner_fold, (fit_idx, valid_idx) in enumerate(splits, start=1):
        if not train.iloc[fit_idx].label_end_date.max() < train.iloc[valid_idx].datetime.min():
            raise ValueError(f"Inner fold {inner_fold}: training labels overlap validation dates")
    return splits


def _metrics(y, score, prediction):
    return {
        "roc_auc": roc_auc_score(y, score),
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
    }


def majority_vote(predictions):
    """Return binary majority decisions for an odd number of voter columns."""
    votes = np.asarray(predictions, dtype=int)
    if votes.ndim != 2 or votes.shape[1] % 2 != 1:
        raise ValueError("Majority vote requires a 2D array with an odd voter count")
    if not np.isin(votes, [0, 1]).all():
        raise ValueError("Voter predictions must be binary")
    return (votes.sum(axis=1) >= (votes.shape[1] // 2 + 1)).astype(int)


def _base_predictions(path, expected_dates):
    pred = pd.read_csv(path, parse_dates=["date"])
    pred = pred[pred.model.isin(BASE_VOTERS)].copy()
    if pred.empty:
        raise FileNotFoundError(f"No saved base-voter OOF predictions in {path}")
    if set(pred.model) != set(BASE_VOTERS):
        raise ValueError("Saved OOF predictions do not contain all three base voters")
    if pred.duplicated(["date", "model"]).any():
        raise ValueError("Duplicate date/model rows in base OOF predictions")
    if set(pred.date) != set(expected_dates):
        raise ValueError("Base OOF dates do not match the execution-target outer folds")
    wide = pred.pivot(index="date", columns="model", values="y_pred").reset_index()
    meta = pred[pred.model.eq(BASE_VOTERS[0])][["date", "fold", "y_true", "forward_return"]]
    wide = wide.merge(meta, on="date", how="inner", validate="one_to_one")
    return wide


def run(source=SOURCE, output_dir=OUT):
    raw_path = DATA_RAW_DIR / "xauusd_daily.csv"
    raw = pd.read_csv(raw_path, parse_dates=["datetime"])
    data, features, _, _ = prepare_execution_dataset(raw)
    outer_splits = create_time_series_splits(data, n_splits=N_SPLITS, gap=GAP)
    oof_parts = []
    tune_rows = []
    search_spaces = make_searches()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        train, test = data.iloc[train_idx].reset_index(drop=True), data.iloc[test_idx].reset_index(drop=True)
        if not train.label_end_date.max() < test.datetime.min():
            raise ValueError(f"Outer fold {outer_fold}: training labels overlap validation dates")
        inner_cv = _inner_splits(train)
        x_train, y_train = train[features], train.target
        x_test, y_test = test[features], test.target

        for model_name, (estimator, param_grid) in search_spaces.items():
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=inner_cv,
                refit=True,
                n_jobs=1,
                return_train_score=False,
                error_score="raise",
            )
            search.fit(x_train, y_train)
            fitted = search.best_estimator_
            class_one = int(np.flatnonzero(fitted.classes_ == 1)[0])
            probability = fitted.predict_proba(x_test)[:, class_one]
            prediction = fitted.predict(x_test).astype(int)
            score = np.log(np.clip(probability, 1e-6, 1 - 1e-6) /
                           (1 - np.clip(probability, 1e-6, 1 - 1e-6)))
            row = pd.DataFrame({
                "date": test.datetime,
                "entry_date": test.entry_date,
                "label_end_date": test.label_end_date,
                "fold": outer_fold,
                "model": model_name,
                "y_true": y_test,
                "score": score,
                "probability": probability,
                "y_pred": prediction,
                "forward_return": test.forward_return,
            })
            oof_parts.append(row)
            tune_rows.append({
                "fold": outer_fold,
                "model": model_name,
                "n_inner_folds": len(inner_cv),
                "best_inner_mean_auc": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "n_candidates": len(search.cv_results_["params"]),
                "n_train": len(train),
                "n_test": len(test),
                "train_label_end": str(train.label_end_date.max().date()),
                "test_start": str(test.datetime.min().date()),
                **_metrics(y_test, probability, prediction),
            })
            print(f"fold={outer_fold} model={model_name} best_inner_auc={search.best_score_:.4f} "
                  f"outer_auc={roc_auc_score(y_test, probability):.4f}", flush=True)

    tuned = pd.concat(oof_parts, ignore_index=True)
    base = _base_predictions(source, data.iloc[[i for _, test in outer_splits for i in test]].datetime)
    joined = base.merge(
        tuned.pivot(index="date", columns="model", values="y_pred").reset_index(),
        on="date", how="inner", validate="one_to_one",
    )
    for model_name in TUNED_VOTERS:
        if joined[model_name].isna().any():
            raise ValueError(f"Missing tuned voter predictions: {model_name}")

    joined["Ensemble_HardVote_3"] = majority_vote(joined[list(BASE_VOTERS)])
    joined["Ensemble_HardVote_5"] = majority_vote(joined[[*BASE_VOTERS, *TUNED_VOTERS]])

    # Scores for hard-vote ensembles are the fraction of UP votes. AUC is
    # descriptive ranking only; the actionable decision remains majority vote.
    score_frames = [
        ("Ensemble_HardVote_3", joined[list(BASE_VOTERS)].astype(float).mean(axis=1)),
        ("Ensemble_HardVote_5", joined[[*BASE_VOTERS, *TUNED_VOTERS]].astype(float).mean(axis=1)),
    ]
    for ensemble_name, vote_fraction in score_frames:
        joined[f"{ensemble_name}_up_fraction"] = vote_fraction

    ensemble_rows = []
    for fold, frame in joined.groupby("fold", sort=True):
        for name in ("Ensemble_HardVote_3", "Ensemble_HardVote_5"):
            ensemble_rows.append({
                "fold": int(fold), "model": name, "n_test": len(frame),
                **_metrics(frame.y_true, frame[f"{name}_up_fraction"], frame[name]),
                "up_vote_fraction_mean": float(frame[f"{name}_up_fraction"].mean()),
            })
    metrics_rows = []
    for frame in tuned.groupby("model", sort=False):
        name, predictions = frame
        for fold, part in predictions.groupby("fold", sort=True):
            metrics_rows.append({
                "fold": int(fold), "model": name, "n_test": len(part),
                **_metrics(part.y_true, part.probability, part.y_pred),
            })
    fold_metrics = pd.DataFrame(metrics_rows + ensemble_rows)
    summary = fold_metrics.groupby("model", sort=False).agg(
        mean_auc=("roc_auc", "mean"), std_auc=("roc_auc", "std"),
        min_auc=("roc_auc", "min"), folds_above_05=("roc_auc", lambda x: int((x > 0.5).sum())),
        mean_accuracy=("accuracy", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
    ).reset_index()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tuned.to_csv(output_dir / "tuned_model_oof_predictions.csv", index=False)
    pd.DataFrame(tune_rows).to_csv(output_dir / "fold_tuning_results.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    joined.to_csv(output_dir / "ensemble_oof_predictions.csv", index=False)
    metadata = {
        "experiment": "nested_h5_execution_ensemble_tuning_v1",
        "target": "open(t+1) to close(t+5), direction; signal after close(t)",
        "calendar": "weekdays proxy; no exchange holiday calendar",
        "feature_set": "TECHNICAL + A+C",
        "outer_folds": N_SPLITS,
        "inner_folds_per_outer_train": 3,
        "gap": GAP,
        "inner_selection_metric": "mean ROC AUC",
        "base_voters": list(BASE_VOTERS),
        "tuned_voters": list(TUNED_VOTERS),
        "ensemble_rules": {"3 models": "at least 2 UP votes", "5 models": "at least 3 UP votes"},
        "all_tuning_inside_outer_train": True,
        "holdout": False,
        "backtest_costs": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary, fold_metrics, tuned, pd.DataFrame(tune_rows), joined


if __name__ == "__main__":
    results = run()
    print(results[0].to_string(index=False))
