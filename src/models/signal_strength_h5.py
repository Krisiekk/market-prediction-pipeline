"""Fold-local H5 signal-strength/abstention diagnostics from saved OOF scores.

No refits, model tuning, threshold optimization, or trading backtest is done.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.models.benchmark import prepare_dataset

SOURCE = DATA_PROCESSED_DIR / 'ensemble_h5_ac' / 'ensemble_oof_predictions.csv'
OUT = DATA_PROCESSED_DIR / 'signal_strength_h5'
COVERAGES = (1.0, .8, .6, .4, .2)
VARIANTS = {
    'ExtraTrees': ('extratrees_probability', 'probability_like'),
    'Logistic': ('logistic_probability', 'probability_like'),
    'ExtraTrees+SVC rank': ('extratrees_svc_rank_ensemble', 'fold_rank'),
}


def load_and_validate(source=SOURCE):
    frame = pd.read_csv(source, parse_dates=['date'])
    required = {'date','fold','y_true','logistic_probability','extratrees_probability','extratrees_svc_rank_ensemble'}
    assert required.issubset(frame.columns)
    assert len(frame) == 3985 and frame.date.nunique() == 3985
    assert not frame.date.duplicated().any()
    assert frame.groupby('fold').size().to_dict() == {i:797 for i in range(1,6)}
    assert frame.y_true.isin([0,1]).all()
    assert frame.extratrees_svc_rank_ensemble.between(0,1).all()
    for _, fold in frame.groupby('fold'):
        expected=(fold.extratrees_rank.to_numpy()+fold.svc_rank.to_numpy())/2
        assert np.allclose(fold.extratrees_svc_rank_ensemble.to_numpy(),expected)
        for rank_col in ('extratrees_rank','svc_rank'):
            assert np.isclose(fold[rank_col].min(),0.) and np.isclose(fold[rank_col].max(),1.)
    return frame.sort_values(['fold','date']).reset_index(drop=True)


def attach_future_return(oof, raw_path=DATA_RAW_DIR / 'xauusd_daily.csv'):
    raw = pd.read_csv(raw_path, parse_dates=['datetime'])
    dataset, _ = prepare_dataset(raw, 5)
    labels = dataset[['datetime','close','future_close','target','label_end_date']].copy()
    labels['future_return'] = labels.future_close / labels.close - 1.0
    labels = labels.rename(columns={'datetime':'date','target':'dataset_target'})
    assert not labels.date.duplicated().any()
    merged = oof.merge(labels, on='date', how='left', validate='one_to_one')
    assert merged[['future_return','future_close','label_end_date']].notna().all().all()
    assert np.array_equal(merged.y_true.to_numpy(), merged.dataset_target.to_numpy())
    assert (merged.label_end_date > merged.date).all()
    return merged.drop(columns='dataset_target')


def _selection(frame, score_col, coverage):
    """Choose top strength rows without access to labels or future returns."""
    strength = (frame[score_col].astype(float) - .5).abs()
    n = len(frame)
    n_select = int(np.ceil(coverage * n))
    # Stable date tiebreak gives deterministic counts when strengths tie.
    order = pd.DataFrame({'strength':strength, 'date':frame.date}).sort_values(
        ['strength','date'], ascending=[False,True], kind='mergesort').index
    selected = pd.Series(False, index=frame.index)
    selected.loc[order[:n_select]] = True
    return selected, strength


def _direction_metrics(y, pred):
    valid = np.asarray(pred) >= 0
    y = np.asarray(y)[valid]
    pred = np.asarray(pred)[valid]
    if len(y) == 0:
        return {'n_signals':0,'directional_accuracy':np.nan,'balanced_accuracy':np.nan,
                'up_precision':np.nan,'down_precision':np.nan}
    correct = pred == y
    recalls=[]
    for klass in (0,1):
        mask = y == klass
        recalls.append(float(np.mean(pred[mask] == klass)) if mask.any() else np.nan)
    bacc = float(np.nanmean(recalls)) if not np.isnan(recalls).all() else np.nan
    return {'n_signals':len(y),'directional_accuracy':float(correct.mean()),'balanced_accuracy':bacc,
            'up_precision':float(precision_score(y,pred,pos_label=1,zero_division=0)),
            'down_precision':float(precision_score(y,pred,pos_label=0,zero_division=0))}


def _signal_metrics(frame, pred):
    pred = np.asarray(pred, dtype=int)
    active = pred >= 0
    up = active & (pred == 1); down = active & (pred == 0)
    directional = _direction_metrics(frame.y_true.to_numpy(), pred)
    return {
        **directional,
        'up_signals':int(up.sum()),'down_signals':int(down.sum()),
        'no_trade':int((~active).sum()),
        'mean_h5_return_up':float(frame.loc[up,'future_return'].mean()) if up.any() else np.nan,
        'median_h5_return_up':float(frame.loc[up,'future_return'].median()) if up.any() else np.nan,
        'mean_h5_return_down':float(frame.loc[down,'future_return'].mean()) if down.any() else np.nan,
        'median_h5_return_down':float(frame.loc[down,'future_return'].median()) if down.any() else np.nan,
        'positive_rate':float(frame.y_true.mean()),
    }


def analyze_coverage(data):
    signal_rows=[]; fold_rows=[]
    for variant,(score_col,score_kind) in VARIANTS.items():
        for fold, part0 in data.groupby('fold',sort=True):
            part=part0.reset_index(drop=True).copy()
            part['score_kind']=score_kind
            part['centered_strength']=(part[score_col]-.5).abs()
            # Assignment comes only from the score, centered value, and fold-local dates.
            for coverage in COVERAGES:
                selected,strength=_selection(part,score_col,coverage)
                chosen=part.loc[selected].copy()
                scores=chosen[score_col].to_numpy()
                pred=np.where(scores>.5,1,np.where(scores<.5,0,-1))
                metrics=_signal_metrics(chosen,pred)
                n_total=len(part)
                metrics['no_trade']=n_total-metrics['n_signals']
                fold_rows.append({'variant':variant,'score_kind':score_kind,'fold':int(fold),'coverage_target':coverage,
                    'n_total':n_total,'n_selected':int(selected.sum()),'actual_selected_coverage':float(selected.mean()),
                    'actual_coverage':metrics['n_signals']/n_total,**metrics})
                row_pred=np.full(n_total,-1,dtype=int)
                row_pred[np.flatnonzero(selected.to_numpy())]=pred
                full=part.copy()
                signal_rows.append(pd.DataFrame({'date':full.date,'fold':int(fold),'y_true':full.y_true,
                    'future_return':full.future_return,'score':full[score_col],'score_kind':score_kind,
                    'centered_strength':strength.to_numpy(),'variant':variant,
                    'coverage_target':coverage,'selected':selected.to_numpy(),'prediction':row_pred,
                    'is_signal':row_pred>=0}))
    fold_df=pd.DataFrame(fold_rows)
    all_signals=pd.concat(signal_rows,ignore_index=True)
    # Pooled across OOF folds; fold metrics remain available separately.
    summary=[]
    for (variant,coverage), group in all_signals.groupby(['variant','coverage_target'],sort=False):
        # A given row appears at each coverage. Aggregate only active predictions,
        # but retain selected count to show exact realized coverage.
        selected=group[group.selected]
        pred=selected.prediction.to_numpy()
        metrics=_signal_metrics(selected,pred)
        n_total=797*5
        metrics['no_trade']=n_total-metrics['n_signals']
        summary.append({'variant':variant,'score_kind':selected.score_kind.iloc[0],
            'coverage_target':coverage,'n_total':n_total,'n_selected':len(selected),
            'actual_selected_coverage':len(selected)/n_total,
            'actual_coverage':metrics['n_signals']/n_total,**metrics,
            'pooled_balanced_accuracy':metrics['balanced_accuracy'],
            'mean_fold_directional_accuracy':fold_df.query('variant == @variant and coverage_target == @coverage').directional_accuracy.mean(),
            'mean_fold_balanced_accuracy':fold_df.query('variant == @variant and coverage_target == @coverage').balanced_accuracy.mean()})
    return pd.DataFrame(summary),fold_df,all_signals


def analyze_quantiles(data):
    fold_rows=[]
    for variant,(score_col,score_kind) in VARIANTS.items():
        for fold, part0 in data.groupby('fold',sort=True):
            part=part0.reset_index(drop=True).copy()
            strength=(part[score_col]-.5).abs()
            # Stable score-only ordering creates five near-equal fold-local bins;
            # Q1 is weakest and Q5 strongest. Labels/returns do not affect bins.
            order=pd.DataFrame({'strength':strength,'date':part.date}).sort_values(
                ['strength','date'],ascending=[True,True],kind='mergesort').index.to_list()
            bins=np.empty(len(part),dtype=int)
            for q,indices in enumerate(np.array_split(np.asarray(order),5),1): bins[indices]=q
            part['strength']=strength;part['quantile']=bins
            pred=np.where(part[score_col]>.5,1,np.where(part[score_col]<.5,0,-1))
            part['prediction']=pred
            for q, selected in part.groupby('quantile',sort=True):
                met=_signal_metrics(selected,selected.prediction.to_numpy())
                row={'variant':variant,'score_kind':score_kind,'fold':int(fold),'quantile':f'Q{q}',
                    'n':len(selected),'directional_accuracy':met['directional_accuracy'],
                    'positive_rate':met['positive_rate'],'mean_h5_return':float(selected.future_return.mean()),
                    'median_h5_return':float(selected.future_return.median()),'mean_strength':float(selected.strength.mean()),
                    'n_directional_signals':met['n_signals']}
                fold_rows.append(row)
    fold_df=pd.DataFrame(fold_rows)
    summary=[]
    for (variant,q),group in fold_df.groupby(['variant','quantile'],sort=False):
        # Pooled counts / rates are based on all five folds, not average fold rates.
        summary.append({'variant':variant,'score_kind':group.score_kind.iloc[0],'quantile':q,'n':int(group.n.sum()),
            'directional_accuracy':np.average(group.directional_accuracy.dropna(),weights=group.loc[group.directional_accuracy.notna(),'n_directional_signals']) if group.directional_accuracy.notna().any() else np.nan,
            'positive_rate':float(np.average(group.positive_rate,weights=group.n)),'mean_h5_return':float(np.average(group.mean_h5_return,weights=group.n)),
            'median_h5_return':np.nan,
            'mean_strength':float(np.average(group.mean_strength,weights=group.n)),'n_directional_signals':int(group.n_directional_signals.sum())})
        # Median H5 return is computed directly from all observations in each bin below.
    # Recompute exact pooled medians from a score-only fold assignment frame.
    for row in summary:
        variant=row['variant'];score_col=VARIANTS[variant][0];q=int(row['quantile'][1:]); selected_returns=[]
        for fold,part0 in data.groupby('fold',sort=True):
            part=part0.sort_values('date').reset_index(drop=True)
            strength=(part[score_col]-.5).abs()
            order=pd.DataFrame({'strength':strength,'date':part.date}).sort_values(['strength','date'],ascending=[True,True],kind='mergesort').index.to_list()
            bins=np.empty(len(part),dtype=int)
            for qi,indices in enumerate(np.array_split(np.asarray(order),5),1): bins[indices]=qi
            selected_returns.extend(part.loc[bins==q,'future_return'].tolist())
        row['median_h5_return']=float(np.median(selected_returns))
    return pd.DataFrame(summary),fold_df


def write_report(out,coverage,folds,quantiles,quantile_folds,auc_context):
    def html(df):return df.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)
    main=coverage[['variant','coverage_target','n_signals','directional_accuracy','balanced_accuracy','up_precision','down_precision']]
    report=f'''<!doctype html><html><meta charset="utf-8"><style>body{{font:15px system-ui;max-width:1400px;margin:32px auto;color:#17202a}}table{{border-collapse:collapse;margin:16px 0;font-size:13px}}td,th{{padding:6px;border:1px solid #ddd}}th{{background:#eef2f7}}.note{{background:#f4f6f7;padding:12px}}</style><body>
<h1>H5 signal strength / abstention diagnostics</h1>
<p class="note">Development OOF only (3,985 observations; 797 per fold). No refits, threshold search, calibration, backtest, or holdout. Coverage selection and quantile bins are fold-local and use only score strength. Future returns are diagnostic only. ExtraTrees+SVC uses rank score, not probability. An exact center score (0.5) is NO TRADE.</p>
<h2>Saved OOF AUC context</h2>{html(auc_context)}
<h2>Coverage summary (pooled OOF)</h2>{html(main)}<h2>Fold coverage results</h2>{html(folds)}
<h2>Strength quantiles (Q1 weakest, Q5 strongest)</h2>{html(quantiles)}<h2>Quantiles by fold</h2>{html(quantile_folds)}
</body></html>'''
    (out/'report.html').write_text(report)


def run(source=SOURCE,out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    oof=load_and_validate(source)
    data=attach_future_return(oof)
    summary,folds,signals=analyze_coverage(data)
    quantiles,quantile_folds=analyze_quantiles(data)
    auc_source=pd.read_csv(DATA_PROCESSED_DIR/'ensemble_h5_ac'/'ensemble_summary.csv')
    auc_names={'ExtraTrees':'ExtraTrees','Logistic':'Logistic','ExtraTrees+SVC rank':'ExtraTrees+SVC rank 50/50'}
    auc_context=auc_source[auc_source.variant.isin(auc_names.values())][['variant','mean_auc','median_auc','std_auc','min_auc','folds_above_05','fold1','fold2','fold3','fold4','fold5']].copy()
    expected_rows=len(VARIANTS)*len(oof)*len(COVERAGES)
    assert len(signals)==expected_rows
    assert folds.groupby(['variant','coverage_target']).size().eq(5).all()
    assert quantile_folds.groupby(['variant','quantile']).size().eq(5).all()
    assert not signals.duplicated(['variant','coverage_target','date']).any()
    summary.to_csv(out/'coverage_summary.csv',index=False);folds.to_csv(out/'coverage_fold_results.csv',index=False)
    quantiles.to_csv(out/'quantile_summary.csv',index=False);quantile_folds.to_csv(out/'quantile_fold_results.csv',index=False)
    signals.to_csv(out/'signal_oof.csv',index=False)
    auc_context.to_csv(out/'auc_context.csv',index=False)
    metadata={'horizon':5,'calendar':'weekdays only','intermarket_lag':1,'gap':5,'shuffle':False,
      'n_common_oof':len(oof),'rows_per_fold':oof.groupby('fold').size().to_dict(),'variants':VARIANTS,
      'coverage_levels':list(COVERAGES),'fold_local_selection':True,'selection_inputs':['score strength=abs(score-0.5)','fold-local date tiebreak'],
      'coverage_rounding':'ceil(target coverage * 797) selected observations per fold; actual coverage differs by at most rounding and exact-center abstentions',
      'target_and_future_return_used_for_selection':False,'rank_score_is_probability':False,'model_refits':0,
      'future_return_definition':'future_close / close - 1 from frozen H5 prepared dataset; diagnostic only',
      'holdout':False,'trading_backtest':False}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    write_report(out,summary,folds,quantiles,quantile_folds,auc_context)
    return summary,folds,quantiles,quantile_folds,signals


if __name__=='__main__':
    res=run()
    print(res[0][['variant','coverage_target','n_signals','directional_accuracy','balanced_accuracy','up_precision','down_precision']].to_string(index=False))
    print('\nQuantiles:')
    print(res[2].to_string(index=False))
