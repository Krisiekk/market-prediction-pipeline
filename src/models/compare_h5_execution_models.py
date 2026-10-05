"""Compare fixed H5 models on an executable weekday target.

The signal is formed after close(t), entered at open(t+1), and exited at
close(t+5). This is a development walk-forward experiment, not a holdout.
Run with: python -m src.models.compare_h5_execution_models
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.features.calendar import filter_weekdays
from src.models.benchmark import SELECTED, TECHNICAL, prepare_dataset
from src.validation.splits import create_time_series_splits

OUT = DATA_PROCESSED_DIR / "h5_execution_model_comparison"
HORIZON = 5
N_SPLITS = 5
GAP = 5
INTERMARKET_FEATURES = [SELECTED[5][0], SELECTED[5][2]]  # A+C
ENSEMBLE_MODELS = ("LogisticRegression", "ExtraTreesClassifier", "SVC")
AR_LAGS = (0, 1, 2, 4, 9)


def build_execution_labels(raw, horizon=HORIZON):
    """Create labels on the weekday grid before feature rows are dropped."""
    weekday = filter_weekdays(raw).sort_values("datetime").reset_index(drop=True)
    if weekday.datetime.isna().any() or weekday.datetime.duplicated().any():
        raise ValueError("XAU dates must be non-null and unique")
    entry_open = pd.to_numeric(weekday["open"], errors="coerce").shift(-1)
    exit_close = pd.to_numeric(weekday["close"], errors="coerce").shift(-horizon)
    label_end_date = weekday["datetime"].shift(-horizon)
    forward_return = exit_close / entry_open - 1.0
    labels = pd.DataFrame({
        "datetime": weekday["datetime"],
        "entry_date": weekday["datetime"].shift(-1),
        "label_end_date": label_end_date,
        "entry_open": entry_open,
        "exit_close": exit_close,
        "forward_return": forward_return,
        "target": (forward_return > 0).astype("Int64"),
    })
    labels.loc[forward_return.isna(), "target"] = pd.NA
    return labels


def add_ar_features(weekday):
    """Use only close-to-close returns observable by the signal at close(t)."""
    frame = filter_weekdays(weekday).sort_values("datetime").copy()
    daily_return = pd.to_numeric(frame["close"], errors="coerce").pct_change(fill_method=None)
    for lag in AR_LAGS:
        frame[f"ar_return_lag_{lag}"] = daily_return.shift(lag)
    return frame[["datetime", *[f"ar_return_lag_{lag}" for lag in AR_LAGS]]]


def prepare_execution_dataset(raw):
    """Build the fixed A+C feature sample and attach the executable H5 label."""
    features, feature_sets = prepare_dataset(raw, HORIZON)
    feature_columns = list(TECHNICAL) + INTERMARKET_FEATURES
    labels = build_execution_labels(raw)
    ar_features = add_ar_features(raw)
    data = features.drop(
        columns=["target", "label_end", "label_end_date", "future_close"], errors="ignore"
    )
    data = data.merge(labels, on="datetime", how="inner", validate="one_to_one")
    data = data.merge(ar_features, on="datetime", how="left", validate="one_to_one")
    ar_columns = [f"ar_return_lag_{lag}" for lag in AR_LAGS]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=feature_columns + ar_columns + ["target", "label_end_date", "entry_open", "exit_close"])
    data = data.sort_values("datetime").reset_index(drop=True)
    data["target"] = data.target.astype(int)
    if not data.datetime.is_monotonic_increasing or data.datetime.duplicated().any():
        raise ValueError("Prepared sample dates must be ordered and unique")
    if not (data.label_end_date.dt.dayofweek < 5).all():
        raise ValueError("H5 label end must remain on a weekday")
    if not data.entry_date.lt(data.label_end_date).all():
        raise ValueError("Entry must precede exit")
    if set(feature_columns + ar_columns) & {
        "target", "entry_open", "exit_close", "forward_return", "label_end_date", "entry_date"
    }:
        raise ValueError("A diagnostic or target column leaked into model features")
    return data, feature_columns, ar_columns, feature_sets


def make_models():
    """Use fixed, previously evaluated settings; no parameter search."""
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.003, max_iter=3000, random_state=42),
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            max_features=0.5, random_state=42, n_jobs=1,
        ),
        "SVC": make_pipeline(
            StandardScaler(),
            SVC(C=0.01, gamma=0.001, kernel="rbf", probability=False),
        ),
        "AR_Logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42),
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(16,), alpha=0.01, solver="lbfgs",
                          max_iter=2000, random_state=42),
        ),
    }


def _raw_score(estimator, x):
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(x), dtype=float)
    probability = np.asarray(estimator.predict_proba(x), dtype=float)
    class_one = int(np.flatnonzero(estimator.classes_ == 1)[0])
    p = np.clip(probability[:, class_one], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid_calibration(estimator, x_train, y_train, x_test, seed=42):
    """Fit a sigmoid on chronological inner OOF scores, then score the outer fold."""
    splitter = TimeSeriesSplit(n_splits=3, gap=GAP)
    oof_scores, oof_targets = [], []
    for inner_train, inner_valid in splitter.split(x_train):
        model = clone(estimator)
        model.fit(x_train.iloc[inner_train], y_train.iloc[inner_train])
        oof_scores.append(_raw_score(model, x_train.iloc[inner_valid]))
        oof_targets.append(y_train.iloc[inner_valid].to_numpy())
    scores = np.concatenate(oof_scores)
    targets = np.concatenate(oof_targets)
    calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    calibrator.fit(scores.reshape(-1, 1), targets)
    final_model = clone(estimator)
    final_model.fit(x_train, y_train)
    calibrated = calibrator.predict_proba(_raw_score(final_model, x_test).reshape(-1, 1))[:, 1]
    train_reference = scores
    return final_model, calibrated, train_reference


def _empirical_rank(reference, values):
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    return np.searchsorted(reference, values, side="right") / max(len(reference), 1)


def _metrics(y, score, prediction, probability=None):
    result = {
        "roc_auc": roc_auc_score(y, score),
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
    }
    if probability is not None:
        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        result["probability_auc"] = roc_auc_score(y, probability)
        result["brier"] = brier_score_loss(y, probability)
        result["log_loss"] = log_loss(y, probability, labels=[0, 1])
    else:
        result["probability_auc"] = np.nan
        result["brier"] = np.nan
        result["log_loss"] = np.nan
    return result


def run(output_dir=OUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_RAW_DIR / "xauusd_daily.csv"
    raw = pd.read_csv(raw_path, parse_dates=["datetime"])
    data, features, ar_features, feature_sets = prepare_execution_dataset(raw)
    models = make_models()
    splits = create_time_series_splits(data, n_splits=N_SPLITS, gap=GAP)

    fold_rows, prediction_frames, split_rows = [], [], []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        if not train.label_end_date.max() < test.datetime.min():
            raise ValueError(f"Fold {fold}: training labels overlap validation dates")
        split_rows.append({
            "fold": fold, "n_train": len(train), "n_test": len(test), "gap": GAP,
            "train_start": train.datetime.min(), "train_end": train.datetime.max(),
            "train_label_end": train.label_end_date.max(),
            "test_start": test.datetime.min(), "test_end": test.datetime.max(),
            "test_label_end": test.label_end_date.max(),
        })

        x_by_model = {
            "LogisticRegression": (train[features], test[features]),
            "ExtraTreesClassifier": (train[features], test[features]),
            "SVC": (train[features], test[features]),
            "AR_Logistic": (train[ar_features], test[ar_features]),
            "MLP": (train[features], test[features]),
        }
        fitted, raw_scores, probabilities, references, predictions = {}, {}, {}, {}, {}

        for name, estimator in models.items():
            x_train, x_test = x_by_model[name]
            if name in ENSEMBLE_MODELS:
                model, calibrated_probability, reference = _sigmoid_calibration(
                    estimator, x_train, train.target, x_test,
                )
                fitted[name] = model
                raw_scores[name] = _raw_score(model, x_test)
                references[name] = reference
                probabilities[name] = calibrated_probability
                predictions[name] = model.predict(x_test).astype(int)
            else:
                model = clone(estimator)
                model.fit(x_train, train.target)
                fitted[name] = model
                raw_scores[name] = _raw_score(model, x_test)
                if hasattr(model, "predict_proba"):
                    class_one = int(np.flatnonzero(model.classes_ == 1)[0])
                    probabilities[name] = model.predict_proba(x_test)[:, class_one]
                predictions[name] = model.predict(x_test).astype(int)

            probability = probabilities.get(name)
            score = raw_scores[name]
            metrics = _metrics(test.target, score, predictions[name], probability)
            extra = {}
            if name == "MLP":
                mlp = fitted[name].named_steps["mlpclassifier"]
                extra = {"mlp_iterations": int(mlp.n_iter_), "mlp_converged": mlp.n_iter_ < mlp.max_iter}
            fold_rows.append({"fold": fold, "model": name, "n_test": len(test), **metrics, **extra})
            prediction_frames.append(pd.DataFrame({
                "date": test.datetime.to_numpy(), "entry_date": test.entry_date.to_numpy(),
                "label_end_date": test.label_end_date.to_numpy(), "fold": fold,
                "model": name, "y_true": test.target.to_numpy(), "score": score,
                "probability": probability if probability is not None else np.nan,
                "y_pred": predictions[name], "forward_return": test.forward_return.to_numpy(),
            }))

        # Three distinct, fixed ensemble rules from the three classifier families.
        votes = np.column_stack([predictions[name] for name in ENSEMBLE_MODELS])
        hard_fraction = votes.mean(axis=1)
        hard_prediction = (hard_fraction >= 0.5).astype(int)
        soft_probability = np.mean([probabilities[name] for name in ENSEMBLE_MODELS], axis=0)
        rank_scores = np.mean([
            _empirical_rank(references[name], raw_scores[name]) for name in ENSEMBLE_MODELS
        ], axis=0)
        ensemble_defs = [
            ("Ensemble_HardVote", hard_fraction, hard_prediction, None),
            ("Ensemble_SoftEqual", soft_probability, (soft_probability >= 0.5).astype(int), soft_probability),
            ("Ensemble_RankEqual", rank_scores, (rank_scores >= 0.5).astype(int), None),
        ]
        for name, score, prediction, probability in ensemble_defs:
            metrics = _metrics(test.target, score, prediction, probability)
            fold_rows.append({"fold": fold, "model": name, "n_test": len(test), **metrics})
            prediction_frames.append(pd.DataFrame({
                "date": test.datetime.to_numpy(), "entry_date": test.entry_date.to_numpy(),
                "label_end_date": test.label_end_date.to_numpy(), "fold": fold,
                "model": name, "y_true": test.target.to_numpy(), "score": score,
                "probability": probability if probability is not None else np.nan,
                "y_pred": prediction, "forward_return": test.forward_return.to_numpy(),
            }))

        always_up = np.ones(len(test), dtype=int)
        fold_rows.append({
            "fold": fold, "model": "AlwaysUp", "n_test": len(test),
            "roc_auc": np.nan, "accuracy": accuracy_score(test.target, always_up),
            "balanced_accuracy": balanced_accuracy_score(test.target, always_up),
            "probability_auc": np.nan, "brier": np.nan, "log_loss": np.nan,
        })
        prediction_frames.append(pd.DataFrame({
            "date": test.datetime.to_numpy(), "entry_date": test.entry_date.to_numpy(),
            "label_end_date": test.label_end_date.to_numpy(), "fold": fold,
            "model": "AlwaysUp", "y_true": test.target.to_numpy(), "score": np.nan,
            "probability": np.nan, "y_pred": always_up,
            "forward_return": test.forward_return.to_numpy(),
        }))

    folds = pd.DataFrame(fold_rows)
    summary = folds.groupby("model", sort=False).agg(
        mean_auc=("roc_auc", "mean"), std_auc=("roc_auc", "std"),
        min_auc=("roc_auc", "min"), folds_above_05=("roc_auc", lambda x: int((x > 0.5).sum())),
        mean_accuracy=("accuracy", "mean"), mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_probability_auc=("probability_auc", "mean"),
        mean_brier=("brier", "mean"), mean_log_loss=("log_loss", "mean"),
    ).reset_index()
    summary.insert(1, "folds", N_SPLITS)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    split_table = pd.DataFrame(split_rows)

    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    split_table.to_csv(output_dir / "splits.csv", index=False)
    metadata = {
        "experiment": "h5_execution_model_comparison_v1",
        "target": "open(t+1) to close(t+5), direction; signal after close(t)",
        "calendar": "observed weekdays proxy; no exchange holiday calendar",
        "horizon": HORIZON, "n_splits": N_SPLITS, "gap": GAP,
        "n_rows": len(data), "n_oof_predictions_per_model": int(len(data) - split_table.n_train.iloc[0] - GAP),
        "start_date": str(data.datetime.min().date()), "end_date": str(data.datetime.max().date()),
        "positive_target_rate": float(data.target.mean()),
        "feature_set": "TECHNICAL + A+C",
        "features": features, "ar_features": ar_features,
        "models": list(models), "ensembles": ["hard majority vote", "equal calibrated probability", "equal rank"],
        "calibration": "sigmoid logistic calibration on inner chronological OOF scores; 3 splits, gap=5",
        "selection_or_tuning": False, "holdout": False, "backtest_costs": False,
        "input_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "versions": {pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "pandas", "scikit-learn", "xgboost")},
        "notes": [
            "Development walk-forward only; this history has been used in prior experiments.",
            "Soft probabilities are calibrated inside each outer training fold using chronological inner OOF scores.",
            "AR_Logistic uses only lagged weekday close-to-close returns observable at signal time.",
            "MLP is a small feed-forward network on technical and A+C features; not an LSTM/Transformer.",
            "No spread, slippage, commission, threshold selection, or holdout is included.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return summary, folds, predictions, split_table


if __name__ == "__main__":
    result = run()
    print(result[0].to_string(index=False))
