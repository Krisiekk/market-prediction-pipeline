"""Descriptive diagnostics for existing H5 LogisticRegression A+C OOF artifacts.

Run: python -m src.validation.stability_h5_ac
No estimator is fitted; features are rebuilt from unchanged RAW through the
existing weekday-aware pipeline solely to join diagnostics to saved OOF dates.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.models.benchmark import prepare_dataset

A = 'dfii10_vs_sma20'
C = 'dxy_sma20_vs_sma50'
OUT = DATA_PROCESSED_DIR / 'stability_h5_ac'
BENCHMARK = DATA_PROCESSED_DIR / 'benchmark_weekdays'
ABLATION = DATA_PROCESSED_DIR / 'ablation_h5_weekdays'


def sample_std(x):
    return x.std(ddof=1)


def analyze(output_dir=OUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_meta = pd.read_csv(BENCHMARK / 'splits.csv').query('horizon == 5').sort_values('fold')
    oof_path = ABLATION / 'predictions.csv'
    oof = pd.read_csv(oof_path, float_precision='round_trip').query("feature_set == 'TECHNICAL + A+C'").copy()
    oof['datetime'] = pd.to_datetime(oof.datetime)
    oof = oof.sort_values('datetime').reset_index(drop=True)
    if oof.empty or oof.datetime.duplicated().any():
        raise ValueError('Expected one saved OOF prediction per H5 date')
    # Saved `score` is LogisticRegression.decision_function, verified from prior
    # ablation against benchmark. Sigmoid equals binary LogisticRegression UP probability.
    oof['probability_up'] = expit(oof.score)
    raw = pd.read_csv(DATA_RAW_DIR / 'xauusd_daily.csv', parse_dates=['datetime'])
    data, sets = prepare_dataset(raw, 5)
    if A not in sets['SELECTED'] or C not in sets['SELECTED']:
        raise ValueError('A+C features unavailable')
    # Existing builder carries target-end diagnostics; not model features.
    joined = oof.merge(data, on='datetime', suffixes=('', '_feature'), validate='one_to_one')
    if len(joined) != len(oof):
        raise ValueError('OOF dates do not match rebuilt processed features')
    if not (joined.target == joined.target_feature).all():
        raise ValueError('Saved OOF target differs from rebuilt weekday H5 target')
    if not (joined.label_end_date == joined.label_end).all():
        raise ValueError('Target end date differs from H5 label end')
    joined['future_return_h5'] = joined.future_close / joined.close - 1
    joined['xau_return_1d_diag'] = joined.close.pct_change()
    joined['xau_volatility_20d_diag'] = joined.xau_return_1d_diag.rolling(20).std()
    joined['year'] = joined.datetime.dt.year

    fold_rows, regime_rows, score_rows, feature_rows, comparison_rows = [], [], [], [], []
    technical = pd.read_csv(BENCHMARK / 'predictions.csv', float_precision='round_trip')
    technical = technical.query("horizon == 5 and feature_set == 'TECHNICAL' and model == 'LogisticRegression'")
    technical['datetime'] = pd.to_datetime(technical.datetime)
    technical_auc = technical.groupby('fold').apply(lambda x: roc_auc_score(x.target, x.score), include_groups=False)
    for _, meta in folds_meta.iterrows():
        fold = int(meta.fold)
        part = joined[joined.fold == fold].copy()
        y, pred, score = part.target, part.prediction, part.score
        if len(part) != int(meta.n_test) or part.datetime.min() != pd.Timestamp(meta.test_start) or part.datetime.max() != pd.Timestamp(meta.test_end):
            raise ValueError(f'OOF fold {fold} does not match benchmark dates')
        auc = roc_auc_score(y, score)
        fold_rows.append(dict(fold=fold, validation_start=part.datetime.min(), validation_end=part.datetime.max(),
            n_validation=len(part), positive_rate=y.mean(), down_rate=1-y.mean(), auc=auc,
            balanced_accuracy=balanced_accuracy_score(y,pred), accuracy=accuracy_score(y,pred),
            mean_future_return=part.future_return_h5.mean(), median_future_return=part.future_return_h5.median(),
            std_future_return=sample_std(part.future_return_h5), mean_xau_daily_return=part.xau_return_1d_diag.mean(),
            xau_volatility_daily=sample_std(part.xau_return_1d_diag), mean_rolling_xau_volatility_20d=part.xau_volatility_20d_diag.mean()))
        market_cols = ['dxy','dxy_return_1d','dxy_return_5d','dxy_return_20d','dxy_vs_sma50',C,
                       'dgs10','dgs10_change_1d','dgs10_change_5d','dgs10_change_20d',
                       'dfii10','dfii10_change_1d','dfii10_change_5d','dfii10_change_20d',A]
        row = {'fold':fold,'start':part.datetime.min(),'end':part.datetime.max(),'n':len(part),
               'up_rate':y.mean(),'down_rate':1-y.mean(),
               'mean_xau_return_1d':part.xau_return_1d_diag.mean(),
               'std_xau_return_1d':sample_std(part.xau_return_1d_diag),
               'mean_xau_return_h5':part.future_return_h5.mean(),
               'std_xau_return_h5':sample_std(part.future_return_h5)}
        for column in market_cols:
            row[f'{column}_mean'] = part[column].mean()
            row[f'{column}_std'] = sample_std(part[column])
        row['A_C_correlation'] = part[[A,C]].corr().iloc[0,1]
        row['A_target_pearson'] = part[A].corr(y)
        row['C_target_pearson'] = part[C].corr(y)
        row['A_future_return_pearson'] = part[A].corr(part.future_return_h5)
        row['C_future_return_pearson'] = part[C].corr(part.future_return_h5)
        regime_rows.append(row)
        scores = part.probability_up
        score_rows.append(dict(fold=fold, score_type='probability_up (sigmoid of saved decision_function)',
            mean=scores.mean(), std=sample_std(scores), min=scores.min(), p25=scores.quantile(.25),
            median=scores.median(), p75=scores.quantile(.75), max=scores.max(),
            decision_mean=score.mean(), decision_std=sample_std(score), decision_min=score.min(), decision_max=score.max()))
        for feature in [A,C]:
            feature_rows.append(dict(fold=fold,feature=feature,n=part[feature].count(),mean=part[feature].mean(),
                std=sample_std(part[feature]),median=part[feature].median(),min=part[feature].min(),max=part[feature].max(),
                pearson_target=part[feature].corr(y),pearson_future_return=part[feature].corr(part.future_return_h5)))
        technical_part=technical[technical.fold==fold]
        comparison_rows.append(dict(fold=fold,date_start=part.datetime.min(),date_end=part.datetime.max(),
            technical_auc=technical_auc.loc[fold],ac_auc=auc,delta_pp=100*(auc-technical_auc.loc[fold])))

    fold_summary=pd.DataFrame(fold_rows)
    regime=pd.DataFrame(regime_rows)
    feature_summary=pd.DataFrame(feature_rows)
    comparison=pd.DataFrame(comparison_rows)
    # Descriptive fold 4/5 vs pooled other folds; not an inferential test.
    regime_values=['up_rate','mean_xau_return_1d','std_xau_return_1d','mean_xau_return_h5','std_xau_return_h5']
    for col in market_cols:
        regime_values += [f'{col}_mean',f'{col}_std']
    regime_values += ['A_C_correlation','A_target_pearson','C_target_pearson','A_future_return_pearson','C_future_return_pearson']
    contrast=[]
    for fold in (4,5):
        this=joined[joined.fold==fold];other=joined[joined.fold!=fold]
        for col in regime_values:
            source_col=col.removesuffix('_mean') if col.endswith('_mean') and col.removesuffix('_mean') in this else col
            if source_col in this:
                a=this[source_col].mean();b=other[source_col].mean();sd=other[source_col].std(ddof=1)
            elif col in regime:
                a=regime.loc[regime.fold==fold,col].iloc[0];b=regime.loc[regime.fold!=fold,col].mean();sd=regime.loc[regime.fold!=fold,col].std(ddof=1)
            else:continue
            contrast.append(dict(fold=fold,diagnostic=col,fold_value=a,other_folds_value=b,
                                 difference=a-b,standardized_difference=(a-b)/sd if pd.notna(sd) and sd else np.nan))
    contrast_df=pd.DataFrame(contrast)

    # Purely diagnostic mean without fold 4; not a new CV estimate.
    ac_aucs=comparison.ac_auc
    performance=pd.DataFrame([dict(metric='AUC mean all folds',value=ac_aucs.mean()),
        dict(metric='AUC median all folds',value=ac_aucs.median()),
        dict(metric='AUC sample std all folds',value=ac_aucs.std(ddof=1)),
        dict(metric='AUC mean folds 1,2,3,5 (diagnostic only)',value=comparison.loc[comparison.fold!=4,'ac_auc'].mean()),
        dict(metric='Fold 4 contribution to overall mean (pp)',value=100*(ac_aucs.mean()-comparison.loc[comparison.fold!=4,'ac_auc'].mean()))])

    yearly=[]
    for year,part in joined.groupby('year'):
        yearly.append(dict(year=year,n=len(part),auc=roc_auc_score(part.target,part.score) if part.target.nunique()==2 else np.nan,
            balanced_accuracy=balanced_accuracy_score(part.target,part.prediction),accuracy=accuracy_score(part.target,part.prediction),
            positive_rate=part.target.mean(),low_sample=len(part)<100))
    year_summary=pd.DataFrame(yearly)
    rolling_tables={}
    for window in (250,500):
        rows=[]
        if len(joined)>=window:
            for end in range(window,len(joined)+1):
                part=joined.iloc[end-window:end]
                if part.target.nunique()==2:
                    rows.append(dict(window_end_date=part.datetime.iloc[-1],rolling_auc=roc_auc_score(part.target,part.score),n=len(part)))
            rolling_tables[window]=pd.DataFrame(rows)
        else:rolling_tables[window]=pd.DataFrame(columns=['window_end_date','rolling_auc','n'])

    outputs={'fold_summary':fold_summary,'technical_vs_ac':comparison,'fold_market_summary':regime,
        'feature_regime_summary':feature_summary,'feature_correlations':feature_summary[['fold','feature','pearson_target','pearson_future_return']],
        'score_distribution':pd.DataFrame(score_rows),'year_summary':year_summary,'fold4_fold5_contrast':contrast_df,
        'performance_diagnostic':performance,'predictions_diagnostics':joined[['datetime','fold','target','score','probability_up','prediction','close','future_close','future_return_h5','label_end_date',A,C]]}
    for name,table in outputs.items():table.to_csv(output_dir/f'{name}.csv',index=False)
    for window,table in rolling_tables.items():table.to_csv(output_dir/f'rolling_auc_{window}.csv',index=False)
    metadata={'source':'existing OOF predictions, no model refits','folds':'benchmark_weekdays/splits.csv',
        'predictions':'ablation_h5_weekdays/predictions.csv; A+C rows only',
        'score':'saved decision_function; probability_up=sigmoid(score), equal to LogisticRegression binary predict_proba',
        'xau_realized_volatility':'sample std of 1-observation close returns per fold',
        'low_sample_year_threshold':100,'no_selection_or_tuning':True}
    (output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    write_report(output_dir,fold_summary,comparison,regime,feature_summary,year_summary,
                 pd.DataFrame(score_rows),performance,rolling_tables,contrast_df)
    return outputs,rolling_tables


def write_report(out,folds,comparison,regime,features,years,scores,performance,rolling,contrast):
    def html(df):return df.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)
    report='''<html><meta charset="utf-8"><style>body{font:14px sans-serif;margin:32px;max-width:1500px}table{border-collapse:collapse;margin:18px 0}td,th{padding:6px;border:1px solid #ddd}th{background:#eef2f7}h2{margin-top:32px}</style>
    <h1>Stability H5 LogisticRegression TECHNICAL + A+C</h1><p>Wyłącznie istniejące OOF predictions; żadnych refitów. Zbudowano opisowe cechy z RAW przez obecny weekday pipeline.</p>
    <p>Score zapisany w OOF to decision_function. Probability UP w tabeli to sigmoid(score), równoważny score’owi predict_proba dla binarnej regresji logistycznej. AUC liczono z surowego score.</p>
    <p>Opisowa diagnostyka może wskazać różnice rozkładów; nie dowodzi przyczyn ekonomicznych, leakage ani zmiany reżimu.</p>'''
    for title,table in [('Granice foldów i outcome',folds),('TECHNICAL vs A+C',comparison),('Reżimy foldów: ceny i rynki',regime),
        ('Rozkład A i C',features),('Rozkład score',scores),('AUC / balanced accuracy / accuracy per year',years),
        ('Mean bez fold 4 — diagnostyka, nie nowa CV',performance),('Fold 4/5 vs pozostałe foldy (opisowo)',contrast)]:
        report+=f'<h2>{title}</h2>'+html(table)
    for w,table in rolling.items():
        report+=f'<h2>Rolling AUC window={w}</h2>'+html(table.tail(100))
    report+='''<h2>Interpretacja / ograniczenia</h2><p>Nie strojono modelu, nie zmieniano cech ani progu. OOF obejmuje lata, na których wcześniej wybierano A+C. H5 etykiety nachodzą na siebie w obrębie bloków testowych; okna rolling są silnie zależne. A/C korelacje i reżimowe zestawienia są opisowe. Brak znanych publikacji FRED point-in-time pozostaje ograniczeniem.</p></html>'''
    (out/'report.html').write_text(report,encoding='utf-8')


if __name__=='__main__':
    tables, rolling=analyze()
    print(tables['fold_summary'].to_string(index=False))
    print(tables['technical_vs_ac'].to_string(index=False))
    print(tables['performance_diagnostic'].to_string(index=False))
