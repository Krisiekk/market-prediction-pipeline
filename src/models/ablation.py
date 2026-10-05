"""H5 LogisticRegression ablation; reuses benchmark data, models and metrics.

python -m src.models.ablation
Requires the existing benchmark artifacts and verifies their reproduction.
No pipeline, target, feature generation or model parameters are changed.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.models.benchmark import (SELECTED, make_models, prepare_dataset,
                                  score_predictions, summarize)
from src.validation.splits import create_time_series_splits

LABELS = dict(zip('ABC', SELECTED[5]))


def make_variants(technical):
    variants = {'TECHNICAL': list(technical)}
    for size in (1, 2, 3):
        for letters in combinations(LABELS, size):
            variants['TECHNICAL + ' + '+'.join(letters)] = list(technical) + [LABELS[c] for c in letters]
    return variants


def run_ablation(output_dir, benchmark_dir):
    output_dir, benchmark_dir = Path(output_dir), Path(benchmark_dir)
    if output_dir.resolve() == benchmark_dir.resolve():
        raise ValueError('Ablation output must not overwrite benchmark results')
    metadata = json.loads((benchmark_dir / 'metadata.json').read_text())
    if metadata.get('calendar') != 'observed_weekdays_v1':
        raise ValueError('Run the weekday benchmark first; historical calendar is incompatible')
    for name, expected in metadata['input_sha256'].items():
        actual = hashlib.sha256((DATA_RAW_DIR / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Input changed since benchmark: {name}')
    raw = pd.read_csv(DATA_RAW_DIR / 'xauusd_daily.csv', parse_dates=['datetime'])
    # Same complete-case sample as benchmark ALL, not separate dropna per variant.
    data, sets = prepare_dataset(raw, horizon=5)
    variants = make_variants(sets['TECHNICAL'])
    if variants['TECHNICAL + A+B+C'] != metadata['feature_sets']['5']['SELECTED']:
        raise ValueError('Selected feature list changed since benchmark')
    prototype = make_models()['LogisticRegression']
    splits = create_time_series_splits(data, n_splits=5, gap=5)
    old_splits = pd.read_csv(benchmark_dir / 'splits.csv').query('horizon == 5').set_index('fold')
    old_predictions = pd.read_csv(benchmark_dir / 'predictions.csv', float_precision='round_trip')
    old_predictions = old_predictions.query("horizon == 5 and model == 'LogisticRegression'")
    records, predictions, index_records = [], [], []
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        if train.label_end.max() >= test.datetime.min():
            raise ValueError('Training labels overlap test')
        previous = old_splits.loc[fold]
        for name, actual in {'train_start': train.datetime.min(), 'train_end': train.datetime.max(),
                             'test_start': test.datetime.min(), 'test_end': test.datetime.max(),
                             'train_label_end': train.label_end.max()}.items():
            if actual != pd.Timestamp(previous[name]):
                raise ValueError(f'Benchmark date mismatch: fold {fold}, {name}')
        if (len(train), len(test), 5) != (previous.n_train, previous.n_test, previous.gap):
            raise ValueError('Benchmark split size/gap mismatch')
        index_records.append(dict(fold=fold, train_indices=train_idx.tolist(), test_indices=test_idx.tolist()))
        for variant, columns in variants.items():
            model = clone(prototype).fit(train[columns], train.target)
            pred = model.predict(test[columns])
            scores = model.decision_function(test[columns])
            if variant in ('TECHNICAL', 'TECHNICAL + A+B+C'):
                feature_set = 'TECHNICAL' if variant == 'TECHNICAL' else 'SELECTED'
                old = old_predictions[(old_predictions.fold == fold) & (old_predictions.feature_set == feature_set)]
                np.testing.assert_array_equal(pd.to_datetime(old.datetime).to_numpy(), test.datetime.to_numpy())
                np.testing.assert_array_equal(old.target.to_numpy(), test.target.to_numpy())
                np.testing.assert_array_equal(old.prediction.to_numpy(), pred)
                np.testing.assert_allclose(old.score.to_numpy(), scores, rtol=1e-10, atol=1e-12)
            keys = dict(horizon=5, model='LogisticRegression', feature_set=variant, fold=fold)
            records.append({**keys, **score_predictions(test.target, pred, scores)})
            predictions.append(pd.DataFrame({**keys, 'datetime': test.datetime.to_numpy(),
                'target': test.target.to_numpy(), 'prediction': pred, 'score': scores}))
        print(f'Fold {fold}/5: 8 variants; benchmark controls reproduced', flush=True)
    folds = pd.DataFrame(records)
    summary = summarize(folds)
    reference = folds.query("feature_set == 'TECHNICAL'").set_index('fold').roc_auc
    folds['delta_vs_technical_pp'] = 100 * (folds.roc_auc - folds.fold.map(reference))
    positive = folds.assign(positive=folds.delta_vs_technical_pp > 0).groupby('feature_set').positive.sum()
    auc_folds = folds.pivot(index='feature_set', columns='fold', values='roc_auc').add_prefix('fold_')
    summary = summary.join(auc_folds, on='feature_set')
    summary['positive_fold_delta_count'] = summary.feature_set.map(positive)
    summary = summary.set_index('feature_set').loc[list(variants)].reset_index()
    summary = summary.rename(columns={'feature_set': 'features', 'roc_auc_mean': 'auc_mean',
        'roc_auc_std': 'auc_std', 'delta_auc_vs_technical_pp': 'delta_vs_technical_pp'})
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    folds.to_csv(output_dir / 'folds.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / 'predictions.csv', index=False)
    (output_dir / 'splits.json').write_text(json.dumps(index_records, indent=2) + '\n')
    (output_dir / 'metadata.json').write_text(json.dumps({
        'calendar': 'observed_weekdays_v1', 'horizon': 5, 'gap': 5, 'intermarket_lag': 1, 'common_rows': len(data),
        'labels': LABELS, 'variants': variants, 'benchmark_dir': str(benchmark_dir.resolve()),
        'benchmark_controls_reproduced': True, 'input_sha256': metadata['input_sha256'],
        'model_parameters': prototype.get_params(deep=True),
        'limitations': 'Exploratory selected features on previously inspected history; no holdout or tuning.'
    }, indent=2, default=str) + '\n')
    columns = ['features', 'auc_mean', 'auc_std'] + [f'fold_{i}' for i in range(1, 6)] + [
        'delta_vs_technical_pp', 'positive_fold_delta_count', 'balanced_accuracy_mean', 'accuracy_mean']
    html = '<html><meta charset="utf-8"><h1>H5 LogisticRegression ablation</h1><p>'
    html += 'A=dfii10_vs_sma20; B=dxy_vs_sma50; C=dxy_sma20_vs_sma50. '
    html += '5 foldów, gap=5, lag=1, identyczna próba benchmarku. Std: ddof=1. '
    html += 'To eksploracja znanej historii, nie potwierdzenie przewagi; wymagany nietknięty holdout.</p>'
    html += summary[columns].to_html(index=False, float_format=lambda v: f'{v:.4f}') + '</html>'
    (output_dir / 'report.html').write_text(html, encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DATA_PROCESSED_DIR / 'ablation_h5_weekdays')
    parser.add_argument('--benchmark-dir', type=Path, default=DATA_PROCESSED_DIR / 'benchmark_weekdays')
    args = parser.parse_args()
    print(run_ablation(args.output_dir, args.benchmark_dir).to_string(index=False))


if __name__ == '__main__':
    main()
