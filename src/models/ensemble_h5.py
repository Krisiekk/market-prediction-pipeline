"""Fixed H5 A+C ensemble diagnostics from the saved tuning OOF predictions.

Run with ``python -m src.models.ensemble_h5``. No models are refit here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from src.config import DATA_PROCESSED_DIR

SOURCE = DATA_PROCESSED_DIR / 'tuning_h5_ac'
OUT = DATA_PROCESSED_DIR / 'ensemble_h5_ac'
MODELS = ['LogisticRegression', 'ExtraTreesClassifier', 'SVC']
EXPECTED_CONFIGS = {
    # This sklearn version records the benchmark's default L2 behavior as
    # penalty='deprecated'; the fitted baseline config is retained verbatim.
    'LogisticRegression': {'logisticregression__C': 0.003, 'logisticregression__class_weight': None, 'logisticregression__max_iter': 3000, 'logisticregression__random_state': 42},
    'ExtraTreesClassifier': {'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 5, 'max_features': 0.5, 'class_weight': None, 'random_state': 42},
    'SVC': {'svc__C': 0.01, 'svc__gamma': 0.001, 'svc__kernel': 'rbf', 'svc__class_weight': None, 'svc__probability': False},
}


def _chosen_predictions(source=SOURCE):
    pred = pd.read_csv(source / 'oof_predictions.csv', parse_dates=['date'])
    top = pd.read_csv(source / 'top_configs.csv')
    chosen = {}
    for name in MODELS:
        rows = top[top.model.eq(name) & ~top.is_baseline.astype(bool)]
        if len(rows) != 1:
            raise ValueError(f'Expected exactly one tuned winner for {name}, got {len(rows)}')
        config_id = rows.iloc[0].config_id
        selected = pred[pred.model.eq(name) & pred.config_id.eq(config_id)].copy()
        if selected.empty:
            raise ValueError(f'Missing saved OOF predictions: {name}/{config_id}')
        chosen[name] = selected
    return chosen


def _validate(chosen):
    ref = chosen[MODELS[0]].sort_values('date').reset_index(drop=True)
    assert len(ref) == 3985 and ref.date.nunique() == 3985
    assert not ref.date.duplicated().any()
    for name, frame in chosen.items():
        frame = frame.sort_values('date').reset_index(drop=True)
        assert len(frame) == 3985 and frame.date.nunique() == 3985
        assert frame.date.equals(ref.date), f'OOF date mismatch: {name}'
        assert frame.fold.equals(ref.fold), f'OOF fold mismatch: {name}'
        assert frame.y_true.equals(ref.y_true), f'y_true mismatch: {name}'
        assert not frame.date.duplicated().any()
        assert frame.groupby('fold').size().to_dict() == {i: 797 for i in range(1, 6)}
        assert frame.score_type.nunique() == 1
    for name, expected in EXPECTED_CONFIGS.items():
        row = pd.read_csv(SOURCE / 'all_configs.csv').query('model == @name and is_baseline == False').sort_values('mean_roc_auc', ascending=False).iloc[0]
        params = json.loads(row.params_json)
        for key, value in expected.items():
            assert params[key] == value, f'Frozen parameter mismatch: {name}/{key}'
    assert chosen['LogisticRegression'].probability_up.notna().all()
    assert chosen['ExtraTreesClassifier'].probability_up.notna().all()
    assert chosen['SVC'].score_type.eq('decision_function').all()
    assert chosen['SVC'].probability_up.isna().all()
    return ref


def _rank_in_fold(values):
    values = pd.Series(values)
    n = len(values)
    if n < 2:
        return np.full(n, 0.5)
    # Average ranks preserve ties and map the fold's minimum/maximum to 0/1.
    return (values.rank(method='average').to_numpy() - 1.0) / (n - 1.0)


def build_oof(chosen):
    frames = {name: frame.sort_values('date').reset_index(drop=True) for name, frame in chosen.items()}
    base = frames['LogisticRegression'][['date', 'fold', 'y_true']].copy()
    base['logistic_score'] = frames['LogisticRegression'].score.to_numpy()
    base['logistic_probability'] = frames['LogisticRegression'].probability_up.to_numpy()
    base['extratrees_probability'] = frames['ExtraTreesClassifier'].probability_up.to_numpy()
    base['svc_decision_score'] = frames['SVC'].score.to_numpy()
    base['logistic_rank'] = np.nan
    base['extratrees_rank'] = np.nan
    base['svc_rank'] = np.nan
    # Normalize independently inside each validation fold; ranks are not probabilities.
    for _, idx in base.groupby('fold', sort=True).groups.items():
        base.loc[idx, 'logistic_rank'] = _rank_in_fold(base.loc[idx, 'logistic_probability'])
        base.loc[idx, 'extratrees_rank'] = _rank_in_fold(base.loc[idx, 'extratrees_probability'])
        base.loc[idx, 'svc_rank'] = _rank_in_fold(base.loc[idx, 'svc_decision_score'])
    base['primary_probability_ensemble'] = 0.5 * base.logistic_probability + 0.5 * base.extratrees_probability
    base['primary_rank_ensemble'] = 0.5 * base.logistic_rank + 0.5 * base.extratrees_rank
    base['logistic_extratrees_svc_rank_ensemble'] = (base.logistic_rank + base.extratrees_rank + base.svc_rank) / 3
    base['extratrees_svc_rank_ensemble'] = 0.5 * base.extratrees_rank + 0.5 * base.svc_rank
    base['logistic_svc_rank_ensemble'] = 0.5 * base.logistic_rank + 0.5 * base.svc_rank
    base['logistic_y_pred'] = frames['LogisticRegression'].y_pred.to_numpy()
    base['extratrees_y_pred'] = frames['ExtraTreesClassifier'].y_pred.to_numpy()
    base['primary_probability_y_pred'] = (base.primary_probability_ensemble >= 0.5).astype(int)
    return base


VARIANTS = {
    'Logistic': ('logistic_probability', 'probability', ['logistic_rank']),
    'ExtraTrees': ('extratrees_probability', 'probability', ['extratrees_rank']),
    'SVC': ('svc_decision_score', 'score', ['svc_rank']),
    'Logistic+ExtraTrees probability 50/50': ('primary_probability_ensemble', 'probability', ['logistic_rank', 'extratrees_rank']),
    'Logistic+ExtraTrees rank 50/50': ('primary_rank_ensemble', 'rank', ['logistic_rank', 'extratrees_rank']),
    'Logistic+ExtraTrees+SVC rank equal': ('logistic_extratrees_svc_rank_ensemble', 'rank', ['logistic_rank', 'extratrees_rank', 'svc_rank']),
    'ExtraTrees+SVC rank 50/50': ('extratrees_svc_rank_ensemble', 'rank', ['extratrees_rank', 'svc_rank']),
    'Logistic+SVC rank 50/50': ('logistic_svc_rank_ensemble', 'rank', ['logistic_rank', 'svc_rank']),
}


def summarize(oof):
    folds = []
    for variant, (score_col, kind, _) in VARIANTS.items():
        for fold, frame in oof.groupby('fold', sort=True):
            score = frame[score_col].to_numpy()
            y = frame.y_true.to_numpy()
            row = {'variant': variant, 'fold': int(fold), 'n': len(frame), 'roc_auc': roc_auc_score(y, score), 'score_kind': kind}
            if kind == 'probability':
                pred = (score >= .5).astype(int)
                row.update(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred))
            else:
                row.update(accuracy=np.nan, balanced_accuracy=np.nan)
            folds.append(row)
    fold_df = pd.DataFrame(folds)
    summaries=[]
    for variant, group in fold_df.groupby('variant', sort=False):
        aucs=group.sort_values('fold').roc_auc.to_numpy()
        vals={'variant':variant,'mean_auc':aucs.mean(),'median_auc':np.median(aucs),'std_auc':aucs.std(ddof=1),'min_auc':aucs.min(),'max_auc':aucs.max(),
              'folds_above_05':int((aucs>.5).sum()),'mean_without_fold4':np.mean([aucs[i-1] for i in [1,2,3,5]])}
        for i,x in enumerate(aucs,1): vals[f'fold{i}']=x
        metric=group[['accuracy','balanced_accuracy']].mean(numeric_only=True)
        vals['mean_accuracy']=metric.get('accuracy',np.nan); vals['mean_balanced_accuracy']=metric.get('balanced_accuracy',np.nan)
        summaries.append(vals)
    return pd.DataFrame(summaries),fold_df


def diversity(oof):
    from scipy.stats import pearsonr, spearmanr
    a=oof.logistic_probability.to_numpy(); b=oof.extratrees_probability.to_numpy()
    pa=oof.logistic_y_pred.to_numpy().astype(int); pb=oof.extratrees_y_pred.to_numpy().astype(int); y=oof.y_true.to_numpy().astype(int)
    corr=pd.DataFrame([{'model_a':'LogisticRegression','model_b':'ExtraTreesClassifier','pearson_score_corr':pearsonr(a,b).statistic,
      'spearman_score_corr':spearmanr(a,b).statistic,'prediction_agreement_rate':np.mean(pa==pb)}])
    ca=pa==y; cb=pb==y
    correctness=pd.DataFrame([{'both_correct':int(np.sum(ca&cb)),'logistic_only_correct':int(np.sum(ca&~cb)),
      'extratrees_only_correct':int(np.sum(~ca&cb)),'both_wrong':int(np.sum(~ca&~cb))}])
    return corr,correctness


def write_report(out, summary, folds, corr, correctness):
    cols=['variant','mean_auc','median_auc','std_auc','min_auc','fold1','fold2','fold3','fold4','fold5','folds_above_05','mean_without_fold4']
    table=summary[cols].to_html(index=False,float_format=lambda x:f'{x:.6f}',border=0)
    report=f'''<!doctype html><html><meta charset="utf-8"><style>body{{font:15px system-ui;max-width:1200px;margin:32px auto;color:#17202a}}table{{border-collapse:collapse;margin:18px 0}}td,th{{padding:7px;border:1px solid #ddd}}th{{background:#eef2f7}}.note{{background:#f4f6f7;padding:12px}}</style><body>
<h1>H5 A+C — OOF ensemble diagnostics</h1><p>Weekday-only calendar, H5, intermarket lag 1, time-series CV gap 5. Uses the saved 3,985 common development OOF rows; no model refits.</p>
<p class="note">All scores are development OOF. Variants containing SVC and the diagnostic primary rank ensemble use fold-wise percentile ranks, not probabilities. No weights were optimized; no calibration, threshold tuning, holdout, or backtest was performed.</p>
<h2>OOF AUC summary</h2>{table}<h2>Fold results</h2>{folds.to_html(index=False,float_format=lambda x:f'{x:.6f}',border=0)}
<h2>Logistic–ExtraTrees diversity</h2>{corr.to_html(index=False,float_format=lambda x:f'{x:.6f}',border=0)}{correctness.to_html(index=False,border=0)}
</body></html>'''
    (out/'report.html').write_text(report)


def run(source=SOURCE, out=OUT):
    global SOURCE
    SOURCE=Path(source); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    chosen=_chosen_predictions(SOURCE); ref=_validate(chosen); oof=build_oof(chosen); summary,folds=summarize(oof); corr,correctness=diversity(oof)
    assert oof.groupby('fold').size().to_dict()=={i:797 for i in range(1,6)}
    assert np.allclose(oof.primary_probability_ensemble,.5*oof.logistic_probability+.5*oof.extratrees_probability)
    assert oof.groupby('fold').logistic_rank.min().eq(0).all() and oof.groupby('fold').logistic_rank.max().eq(1).all()
    assert oof.groupby('fold').svc_rank.min().eq(0).all() and oof.groupby('fold').svc_rank.max().eq(1).all()
    oof.to_csv(out/'ensemble_oof_predictions.csv',index=False)
    summary.to_csv(out/'ensemble_summary.csv',index=False); folds.to_csv(out/'ensemble_fold_results.csv',index=False)
    corr.to_csv(out/'diversity_summary.csv',index=False)
    correctness.to_csv(out/'diversity_correctness.csv',index=False)
    metadata={'horizon':5,'calendar':'weekdays only','intermarket_lag':1,'gap':5,'shuffle':False,'n_oof':len(oof),'n_folds':5,
      'rows_per_fold':oof.groupby('fold').size().to_dict(),'models':['LogisticRegression','ExtraTreesClassifier','SVC'],
      'source':'saved tuning_h5_ac/oof_predictions.csv','model_refits':0,'weights':{'Logistic+ExtraTrees probability': [0.5,0.5],
      'Logistic+ExtraTrees rank':[0.5,0.5],'Logistic+ExtraTrees+SVC rank':[1/3,1/3,1/3],'ExtraTrees+SVC rank':[0.5,0.5],'Logistic+SVC rank':[0.5,0.5]},
      'rank_normalization':'per validation fold; average ranks mapped to [0,1]; not probabilities','holdout':False,'ensemble_fit':False}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    write_report(out,summary,folds,corr,correctness)
    return summary,folds,oof,corr,correctness


if __name__=='__main__':
    result=run()
    print(result[0].to_string(index=False))
    print('\nDiversity:')
    print(result[3].to_string(index=False))
    print('\nCorrectness overlap:')
    print(result[4].to_string(index=False))
