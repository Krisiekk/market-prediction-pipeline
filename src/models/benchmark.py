"""Reproducible, expanding-window XAU/USD family benchmark (no tuning).

Run: python -m src.models.benchmark
All feature sets use the same complete-case sample and frozen temporal folds.
Prediction is after XAU close. Macro lag=1 observation matches exploration,
not a point-in-time publication guarantee. No untouched holdout is used here.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.features.build_features import build_features
from src.features.calendar import filter_weekdays
from src.features.indicators import add_all_indicators
from src.models.baseline import always_up_baseline
from src.validation.splits import create_time_series_splits

# Same 14 inputs as train.py; BB_MIDDLE duplicates SMA_20 and was not used there.
TECHNICAL = ['SMA_20', 'RSI_14', 'EMA_20', 'MACD', 'MACD_Signal', 'MACD_HIST',
             'BB_UPPER', 'BB_LOWER', 'PRICE_TO_SMA20', 'PRICE_TO_EMA20',
             'BB_WIDTH', 'BB_POSITION', 'RETURN_1D', 'RETURN_5D']
SELECTED = {1: ['dgs10_vs_sma20', 'dxy_sma20_vs_sma50'],
            5: ['dfii10_vs_sma20', 'dxy_vs_sma50', 'dxy_sma20_vs_sma50']}
METRICS = ['roc_auc', 'balanced_accuracy', 'accuracy']
KEYS = ['horizon', 'feature_set', 'model']


def make_models():
    """Fixed base parameters, identical across horizons and feature sets."""
    return {
        'LogisticRegression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=42)),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=1),
        'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=1),
        'SVC': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale', probability=False)),
        # Disable random internal validation/early stopping. Fit only on train.
        'HistGradientBoostingClassifier': HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15,
            learning_rate=.05, l2_regularization=1., early_stopping=False, random_state=42),
        'XGBClassifier': XGBClassifier(n_estimators=200, max_depth=2, learning_rate=.03,
            subsample=.8, colsample_bytree=.8, min_child_weight=5, reg_alpha=.1,
            reg_lambda=1., eval_metric='logloss', random_state=42, n_jobs=1),
    }


def prepare_dataset(raw, horizon):
    raw = filter_weekdays(raw)
    if raw.datetime.isna().any() or raw.datetime.duplicated().any():
        raise ValueError('Invalid or duplicate XAU dates')
    # Detect every column introduced by intermarket, including four raw levels.
    before_macro = set(add_all_indicators(raw).columns)
    data = build_features(raw, horizon=horizon, intermarket_lag=1)
    macro = [c for c in data if c not in before_macro and c not in {'target', 'label_end_date', 'future_close', 'label_end'}]
    sets = {'TECHNICAL': TECHNICAL.copy(),
            'SELECTED': TECHNICAL + SELECTED[horizon], 'ALL': TECHNICAL + macro}
    if not set(SELECTED[horizon]).issubset(macro):
        raise ValueError('Selected intermarket features missing from pipeline')
    if any(set(cols) & {'target', 'label_end_date', 'future_close', 'label_end', 'weekday'} for cols in sets.values()):
        raise ValueError('Target must never be included in X')
    # Reuse the target's weekday-grid date, computed before dropna.
    data['label_end'] = data['label_end_date']
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=sets['ALL'] + ['target', 'label_end'])
    data = data.reset_index(drop=True)
    data['target'] = data.target.astype(int)
    return data, sets


def score_predictions(y, pred, scores=None):
    return {'roc_auc': roc_auc_score(y, scores) if scores is not None else np.nan,
            'balanced_accuracy': balanced_accuracy_score(y, pred),
            'accuracy': accuracy_score(y, pred)}


def summarize(folds):
    summary = folds.groupby(KEYS).agg(**{
        f'{metric}_{stat}': (metric, stat) for metric in METRICS for stat in ('mean', 'std')
    }).reset_index()
    reference = summary.query("feature_set == 'TECHNICAL'")[['horizon', 'model', 'roc_auc_mean']]
    reference = reference.rename(columns={'roc_auc_mean': 'technical_auc'})
    summary = summary.merge(reference, on=['horizon', 'model'], validate='many_to_one')
    summary['delta_auc_vs_technical_pp'] = 100 * (summary.roc_auc_mean - summary.technical_auc)
    return summary.drop(columns='technical_auc').sort_values(['horizon', 'roc_auc_mean'], ascending=[True, False])


def run_benchmark(raw, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = make_models()
    records, baselines, predictions, split_records, coverage = [], [], [], [], []
    feature_lists = {}
    for horizon in (1, 5):
        data, sets = prepare_dataset(raw, horizon)
        feature_lists[str(horizon)] = sets
        coverage.append(dict(horizon=horizon, raw_rows=len(raw), common_rows=len(data),
                             start=data.datetime.min(), end=data.datetime.max()))
        splits = create_time_series_splits(data, n_splits=5, gap=horizon)
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            train, test = data.iloc[train_idx], data.iloc[test_idx]
            if train.label_end.max() >= test.datetime.min():
                raise ValueError('Training labels overlap the test interval')
            if train.target.nunique() != 2 or test.target.nunique() != 2:
                raise ValueError('Both classes are required in every train/test fold')
            dates = dict(horizon=horizon, fold=fold, gap=horizon, n_train=len(train), n_test=len(test),
                         train_start=train.datetime.min(), train_end=train.datetime.max(),
                         train_label_end=train.label_end.max(), test_start=test.datetime.min(), test_end=test.datetime.max())
            split_records.append(dates)
            baselines.append({**dates, 'model': 'AlwaysUp',
                              **score_predictions(test.target, always_up_baseline(len(test)))})
            for feature_set, columns in sets.items():
                for name, prototype in models.items():
                    fitted = clone(prototype).fit(train[columns], train.target)
                    pred = fitted.predict(test[columns])
                    # SVC uses ranking scores; no random probability calibration CV.
                    scores = (fitted.decision_function(test[columns]) if hasattr(fitted, 'decision_function')
                              else fitted.predict_proba(test[columns])[:, 1])
                    keys = dict(horizon=horizon, fold=fold, feature_set=feature_set, model=name)
                    records.append({**dates, **keys, 'n_features': len(columns),
                                    **score_predictions(test.target, pred, scores)})
                    predictions.append(pd.DataFrame({**keys, 'datetime': test.datetime.to_numpy(),
                        'target': test.target.to_numpy(), 'prediction': pred, 'score': scores}))
            print(f'H{horizon}, fold {fold}/5: 18 fits completed', flush=True)
    folds = pd.DataFrame(records)
    summary = summarize(folds)
    baseline_folds = pd.DataFrame(baselines)
    baseline_summary = baseline_folds.groupby(['horizon', 'model']).agg(**{
        f'{metric}_{stat}': (metric, stat) for metric in METRICS[1:] for stat in ('mean', 'std')
    }).reset_index()
    auc_folds = folds.pivot(index=KEYS, columns='fold', values='roc_auc').add_prefix('fold_').reset_index()
    stability = summary.merge(auc_folds, on=KEYS, validate='one_to_one')
    stability['folds_auc_above_half'] = (stability[[f'fold_{i}' for i in range(1, 6)]] > .5).sum(axis=1)
    reference = folds.query("feature_set == 'TECHNICAL'")[['horizon', 'model', 'fold', 'roc_auc']]
    paired = folds.merge(reference, on=['horizon', 'model', 'fold'], suffixes=('', '_technical'), validate='many_to_one')
    paired['delta_auc_vs_technical_pp'] = 100 * (paired.roc_auc - paired.roc_auc_technical)
    paired['improved'] = paired.delta_auc_vs_technical_pp > 0
    paired_summary = paired.groupby(KEYS).agg(delta_auc_vs_technical_pp=('delta_auc_vs_technical_pp', 'mean'),
        delta_std_pp=('delta_auc_vs_technical_pp', 'std'), improved_folds=('improved', 'sum')).reset_index()
    tables = dict(summary=summary, folds=folds, baseline_folds=baseline_folds,
                  baseline_summary=baseline_summary, stability=stability,
                  feature_comparison=paired_summary, paired_folds=paired,
                  predictions=pd.concat(predictions, ignore_index=True),
                  splits=pd.DataFrame(split_records), coverage=pd.DataFrame(coverage))
    for name, table in tables.items():
        table.to_csv(output_dir / f'{name}.csv', index=False)
    metadata = {'calendar': 'observed_weekdays_v1', 'seed': 42, 'n_splits': 5, 'gap': {'1': 1, '5': 5}, 'intermarket_lag': 1,
        'feature_sets': feature_lists, 'std_ddof': 1,
        'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'xgboost')},
        'model_parameters': {name: m.get_params(deep=True) for name, m in models.items()},
        'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA_RAW_DIR.glob('*.csv'))},
        'assumptions': ['After-close prediction; horizons mean successive Monday-Friday source observations; no holiday calendar.',
            'Backward asof and macro lag 1; no publication timestamps or historical vintages available.',
            'ALL includes raw intermarket levels; TECHNICAL uses the 14 historical train.py inputs.',
            'Selected features came from prior exploration of this history: CV is not untouched evaluation.',
            'No tuning, holdout, feature removal or ensemble. H5 labels overlap within test folds.']}
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    write_report(tables, output_dir)
    return tables


def write_report(tables, output_dir):
    def table(frame):
        return frame.to_html(index=False, float_format=lambda v: f'{v:.4f}', border=0)
    content = '''<html><meta charset="utf-8"><style>body{font:14px sans-serif;margin:30px}
    table{border-collapse:collapse;margin-bottom:24px}td,th{padding:7px;border:1px solid #ddd}
    th{background:#eef2f7}h2{margin-top:32px}</style><h1>Benchmark XAU/USD — H1 / H5</h1>
    <p>36 konfiguracji, 180 dopasowań, 5 foldów czasowych. Gap = horyzont.
    Te same daty dla wszystkich konfiguracji danego horyzontu. Std próbki (ddof=1).</p>
    <p>Intermarket opóźnione o 1 obserwację przed dropna; brak gwarancji rzeczywistych godzin publikacji.
    ALL zawiera także surowe poziomy intermarket. H1/H5 odnoszą się do kolejnych obserwacji poniedziałek–piątek.
    Selected wybrane wcześniej na tej historii — wyniki pozostają eksploracyjne.
    Potrzebny będzie całkowicie nietknięty holdout. AUC 0,51–0,52 nie potwierdza przewagi tradingowej.</p>'''
    for title, name in [('Wspólna próba', 'coverage'), ('Always Up (bez AUC)', 'baseline_summary'),
                        ('Wszystkie konfiguracje', 'summary'), ('Wpływ zestawów cech (pp AUC)', 'feature_comparison')]:
        content += f'<h2>{title}</h2>' + table(tables[name])
    content += '<h2>Top 3 średnie AUC per horyzont — stabilność, nie wybór zwycięzcy</h2>'
    content += table(tables['stability'].groupby('horizon', sort=False).head(3))
    content += '<h2>Daty foldów</h2>' + table(tables['splits']) + '</html>'
    (output_dir / 'report.html').write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DATA_PROCESSED_DIR / 'benchmark_weekdays')
    args = parser.parse_args()
    raw = pd.read_csv(DATA_RAW_DIR / 'xauusd_daily.csv', parse_dates=['datetime'])
    tables = run_benchmark(raw, args.output_dir)
    print(tables['summary'].to_string(index=False))
    print(f'Results: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
