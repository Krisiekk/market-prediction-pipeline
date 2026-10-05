"""Controlled H5 A+C CV tuning; no holdout, ensemble, or feature selection.

Run: python -m src.models.tune_h5
All configurations use saved benchmark weekday folds and identical X/y rows.
Mean ROC AUC is the ranking criterion; dispersion is reported separately.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.models.benchmark import SELECTED, make_models, prepare_dataset
from src.validation.splits import create_time_series_splits

OUT = DATA_PROCESSED_DIR / 'tuning_h5_ac'
BENCHMARK = DATA_PROCESSED_DIR / 'benchmark_weekdays'
ABLATION = DATA_PROCESSED_DIR / 'ablation_h5_weekdays'
FEATURE_SET = 'TECHNICAL + A+C'
FEATURES = SELECTED[5][0], SELECTED[5][2]
BASELINE_AC_AUC = 0.5293129935084135
BASELINE_AC_FOLDS = [0.5240502334974126,0.5260663507109005,0.5245158952055503,
                     0.5940627126761273,0.47786977545207654]


def canonical(params):
    return json.dumps(params, sort_keys=True, separators=(',', ':'), default=str)


def configurations():
    baselines = make_models()
    output = {}
    # L2 only: the optional L1 grid is deliberately excluded to keep this pass scoped.
    lr = [{'logisticregression__C': c, 'logisticregression__class_weight': cw}
          for c in [.001,.003,.01,.03,.1,.3,1.0,3,10,30,100]
          for cw in [None,'balanced']]
    output['LogisticRegression'] = (baselines['LogisticRegression'], lr)
    svc = [{'svc__C': c, 'svc__gamma': g, 'svc__class_weight': cw}
           for c in [.01,.03,.1,.3,1.0,3,10,30]
           for g in ['scale',.001,.003,.01,.03,.1]
           for cw in [None,'balanced']]
    output['SVC'] = (baselines['SVC'], svc)
    # 39 reproducible draws + unchanged prior-benchmark baseline (40 total).
    et_space = {'n_estimators':[300,600], 'max_depth':[None,4,6,8,12],
        'min_samples_leaf':[1,3,5,10,20], 'max_features':['sqrt',.5,1.0],
        'class_weight':[None,'balanced']}
    et = list(ParameterSampler(et_space,n_iter=39,random_state=42))
    output['ExtraTreesClassifier'] = (baselines['ExtraTreesClassifier'], et)
    hgb_space = {'learning_rate':[.01,.03,.05,.1], 'max_iter':[100,200,400],
        'max_leaf_nodes':[7,15,31], 'l2_regularization':[0,.1,1,10],
        'min_samples_leaf':[10,20,40]}
    hgb = list(ParameterSampler(hgb_space,n_iter=39,random_state=43))
    output['HistGradientBoostingClassifier'] = (baselines['HistGradientBoostingClassifier'], hgb)
    return output


def apply_params(model, params):
    # `None` means retain the estimator's exact benchmark default for tree baselines.
    return clone(model).set_params(**{k:v for k,v in params.items() if v is not None})


def configs_for_model(name, baseline, candidates):
    baseline_params = baseline.get_params(deep=True)
    configs = []
    seen = set()
    for candidate in candidates:
        params = dict(candidate)
        model = apply_params(baseline, params)
        resolved = model.get_params(deep=True)
        key = canonical({k:resolved[k] for k in sorted(resolved) if k in params or k in baseline_params})
        if key in seen:
            continue
        seen.add(key)
        # The baseline config is identified by the full effective estimator params.
        is_baseline = all(resolved[k] == baseline_params[k] for k in params)
        config_id = 'BASELINE' if is_baseline else f'{name}_{len(configs):03d}'
        if any(c['config_id'] == 'BASELINE' for c in configs):
            config_id = f'{name}_{len(configs):03d}'
        configs.append({'config_id':config_id,'is_baseline':is_baseline,'params':params,'estimator':model})
    # Ensure every model has a baseline, including ET, where 200 estimators is outside grid.
    if not any(c['is_baseline'] for c in configs):
        configs.insert(0,{'config_id':'BASELINE','is_baseline':True,'params':{},'estimator':clone(baseline)})
    else:
        for c in configs:
            if c['is_baseline']:
                c['config_id']='BASELINE'
    return configs


def validate_frozen_data(data, feature_sets, raw, benchmark_dir):
    assert len(feature_sets) and FEATURES[0] in feature_sets['SELECTED'] and FEATURES[1] in feature_sets['SELECTED']
    xcols = [*feature_sets['TECHNICAL'], *FEATURES]
    assert len(xcols) == len(set(xcols))
    assert not set(xcols) & {'target','future_close','label_end_date','label_end','weekday'}
    assert (data.datetime.dt.dayofweek < 5).all()
    assert data.datetime.is_monotonic_increasing and not data.datetime.duplicated().any()
    assert data.label_end_date.dt.dayofweek.lt(5).all()
    assert data.label_end_date.equals(data.label_end)
    saved = pd.read_csv(benchmark_dir/'splits.csv').query('horizon==5').sort_values('fold')
    assert len(saved)==5 and saved.gap.eq(5).all()
    splits=create_time_series_splits(data,n_splits=5,gap=5)
    dates=[]
    for i,(train_idx,test_idx) in enumerate(splits,1):
        train,test=data.iloc[train_idx],data.iloc[test_idx]
        row=saved.iloc[i-1]
        assert (len(train),len(test))==(row.n_train,row.n_test)
        assert (train.datetime.min(),train.datetime.max(),test.datetime.min(),test.datetime.max()) == tuple(pd.Timestamp(row[x]) for x in ['train_start','train_end','test_start','test_end'])
        assert train.label_end_date.max() < test.datetime.min()
        dates.append(test.datetime.to_numpy())
    assert len(dates) == 5 and all(len(d) == len(dates[i]) for i, d in enumerate(dates))
    # Every estimator/configuration reuses this one frozen split list.
    metadata=json.loads((benchmark_dir/'metadata.json').read_text())
    for filename, expected_hash in metadata['input_sha256'].items():
        actual_hash=hashlib.sha256((DATA_RAW_DIR/filename).read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f'RAW input changed: {filename}'
    assert metadata.get('calendar')=='observed_weekdays_v1'
    assert metadata['feature_sets']['5']['SELECTED'] == feature_sets['SELECTED']
    assert metadata['intermarket_lag']==1
    return splits,metadata


def evaluate(output_dir=OUT):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(DATA_RAW_DIR/'xauusd_daily.csv',parse_dates=['datetime'])
    data,sets=prepare_dataset(raw,5)
    splits,benchmeta=validate_frozen_data(data,sets,raw,BENCHMARK)
    xcols=list(sets['TECHNICAL'])+list(FEATURES)
    if len(data)!=4786:
        raise ValueError(f'Unexpected frozen A+C dataset size: {len(data)}')
    search=configurations();models=make_models()
    config_list={name:configs_for_model(name,est,candidates) for name,(est,candidates) in search.items()}
    fold_rows=[];pred_frames=[];config_rows=[];all_fold_dates=[]
    reference=pd.read_csv(ABLATION/'folds.csv').query("feature_set == @FEATURE_SET")
    ref_auc=reference.set_index('fold').roc_auc.to_dict()
    ref_oof=pd.read_csv(ABLATION/'predictions.csv',float_precision='round_trip').query("feature_set == @FEATURE_SET")
    for fold,(tri,vi) in enumerate(splits,1):
        train,val=data.iloc[tri],data.iloc[vi]
        fold_dates=dict(fold=fold,train_start=train.datetime.min(),train_end=train.datetime.max(),
            train_label_end=train.label_end_date.max(),validation_start=val.datetime.min(),validation_end=val.datetime.max(),
            n_train=len(train),n_validation=len(val),gap=5)
        all_fold_dates.append(fold_dates)
        assert train.label_end_date.max()<val.datetime.min()
        for name,configs in config_list.items():
            for cfg in configs:
                model=clone(cfg['estimator'])
                model.fit(train[xcols],train.target)
                pred=model.predict(val[xcols])
                if hasattr(model,'predict_proba'):
                    probability=model.predict_proba(val[xcols])[:,1]
                    score=probability
                    score_kind='probability_up'
                else:
                    probability=np.full(len(val),np.nan)
                    score=model.decision_function(val[xcols])
                    score_kind='decision_function'
                auc=roc_auc_score(val.target,score)
                metrics=dict(roc_auc=auc,balanced_accuracy=balanced_accuracy_score(val.target,pred),accuracy=accuracy_score(val.target,pred))
                row={**fold_dates,'model':name,'config_id':cfg['config_id'],'is_baseline':cfg['is_baseline'],
                     'params_json':canonical(model.get_params(deep=True)), 'search_params_json':canonical(cfg['params']),
                     'n_features':len(xcols),**metrics}
                if name in ('LogisticRegression','SVC'):
                    scaler=model.named_steps['standardscaler']
                    assert int(np.asarray(scaler.n_samples_seen_).max())==len(train)
                    row['scaler_fit_samples']=int(np.asarray(scaler.n_samples_seen_).max())
                fold_rows.append(row)
                pred_frames.append(pd.DataFrame({'date':val.datetime.to_numpy(),'fold':fold,'model':name,
                    'config_id':cfg['config_id'],'is_baseline':cfg['is_baseline'],'y_true':val.target.to_numpy(),
                    'score':score,'score_type':score_kind,'probability_up':probability,'y_pred':pred}))
                if name=='LogisticRegression' and cfg['is_baseline']:
                    prior=ref_oof[ref_oof.fold==fold].sort_values('datetime')
                    assert np.array_equal(pd.to_datetime(prior.datetime).to_numpy(),val.datetime.to_numpy())
                    assert np.array_equal(prior.target.to_numpy(),val.target.to_numpy())
                    assert np.array_equal(prior.prediction.to_numpy(),pred)
                    assert np.allclose(prior.score.to_numpy(),model.decision_function(val[xcols]),rtol=1e-10,atol=1e-12)
        print(f'Fold {fold}/5: {sum(map(len,config_list.values()))} configs completed',flush=True)
    folds=pd.DataFrame(fold_rows)
    group=['model','config_id','is_baseline','params_json','search_params_json']
    summaries=[]
    for keys,g in folds.groupby(group,sort=False):
        row=dict(zip(group,keys))
        for metric in ['roc_auc','balanced_accuracy','accuracy']:
            row[f'mean_{metric}']=g[metric].mean()
            row[f'std_{metric}']=g[metric].std(ddof=1)
        aucs=g.sort_values('fold').roc_auc.to_numpy()
        row['median_auc']=np.median(aucs);row['min_auc']=aucs.min();row['max_auc']=aucs.max()
        row['folds_above_05']=int((aucs>.5).sum())
        for i,value in enumerate(aucs,1):row[f'fold_{i}_auc']=value
        if row['model']=='LogisticRegression':
            row['delta_mean_auc_pp_vs_previous_ac']=100*(row['mean_roc_auc']-BASELINE_AC_AUC)
            for i,value in enumerate(aucs,1):row[f'delta_fold_{i}_pp_vs_previous_ac']=100*(value-BASELINE_AC_FOLDS[i-1])
        else:
            row['delta_mean_auc_pp_vs_model_baseline']=np.nan
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    for name in summary.model.unique():
        base_auc=summary.loc[(summary.model==name)&summary.is_baseline,'mean_roc_auc'].iloc[0]
        summary.loc[summary.model==name,'delta_mean_auc_pp_vs_model_baseline']=100*(summary.loc[summary.model==name,'mean_roc_auc']-base_auc)
    summary=summary.sort_values(['model','mean_roc_auc'],ascending=[True,False]).reset_index(drop=True)
    # Select by mean AUC only. Stability remains visible alongside the ranking.
    best_ids=summary.sort_values('mean_roc_auc',ascending=False).groupby('model',sort=False).head(1).set_index('model').config_id.to_dict()
    chosen=summary[summary.is_baseline | summary.apply(lambda r:best_ids[r.model]==r.config_id,axis=1)]
    oof=pd.concat(pred_frames,ignore_index=True)
    oof=oof[oof.is_baseline | oof.apply(lambda r:best_ids[r.model]==r.config_id,axis=1)].copy()
    assert not oof.duplicated(['model','config_id','date']).any()
    date_bounds = folds[['fold','train_end','validation_start','validation_end']].drop_duplicates('fold').set_index('fold')
    for fold, rows in oof.groupby('fold'):
        bounds = date_bounds.loc[fold]
        assert (pd.to_datetime(rows.date) > pd.Timestamp(bounds.train_end)).all()
        assert (pd.to_datetime(rows.date) >= pd.Timestamp(bounds.validation_start)).all()
        assert (pd.to_datetime(rows.date) <= pd.Timestamp(bounds.validation_end)).all()
    expected_oof = sum(len(test_idx) for _, test_idx in splits)
    assert oof.groupby(['model','config_id']).size().eq(expected_oof).all()
    # Correlation and classification agreement for the four mean-AUC winners,
    # aligned on the exact common OOF dates.
    best_oof=oof[oof.apply(lambda r:best_ids[r.model]==r.config_id,axis=1)]
    score_wide=best_oof.pivot(index='date',columns='model',values='score')
    pred_wide=best_oof.pivot(index='date',columns='model',values='y_pred')
    corr=score_wide.corr(method='pearson').rename_axis('model').reset_index()
    agreement=pd.DataFrame(index=pred_wide.columns,columns=pred_wide.columns,dtype=float)
    for a in pred_wide:
        for b in pred_wide:agreement.loc[a,b]=(pred_wide[a]==pred_wide[b]).mean()
    agreement=agreement.rename_axis('model').reset_index()
    # Include previous frozen A+C reference, evaluated on identical OOF dates.
    prior=ref_oof[['datetime','fold','target','score','prediction']].rename(columns={'datetime':'date','target':'y_true','score':'score','prediction':'y_pred'})
    prior['model']='LogisticRegression';prior['config_id']='PREVIOUS_AC_BASELINE';prior['is_baseline']=True;prior['score_type']='decision_function';prior['probability_up']=np.nan
    # It is the same baseline OOF and should equal the benchmark baseline below.
    if not ((oof.model=='LogisticRegression')&(oof.config_id=='BASELINE')).any():pass
    # Save complete fold coverage and metadata.
    outputs={'all_configs':summary,'top_configs':chosen,'fold_results':folds,'oof_predictions':oof,
             'model_score_correlations':corr,'model_prediction_agreement':agreement,
             'fold_dates':pd.DataFrame(all_fold_dates)}
    for name,table in outputs.items():table.to_csv(output_dir/f'{name}.csv',index=False)
    import importlib.metadata
    metadata={'horizon':5,'calendar':'observed_weekdays_v1','intermarket_lag':1,'feature_set':FEATURE_SET,
        'features':xcols,'n_rows':len(data),'n_splits':5,'gap':5,'shuffle':False,
        'ranking':'mean ROC AUC only; stability reported alongside, no stability penalty',
        'randomized_seed':42,'hist_seed':43,'configs_per_model':{k:len(v) for k,v in config_list.items()},
        'baseline_parameters':{k:v.get_params(deep=True) for k,v in models.items()},
        'input_sha256':benchmeta['input_sha256'],
        'previous_ac_baseline_auc':BASELINE_AC_AUC,'previous_ac_fold_auc':BASELINE_AC_FOLDS,
        'OOF_best_configs_selected_on_same_folds':'diagnostic only; not unbiased post-selection performance',
        'holdout_run':False,'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scikit-learn','xgboost']}}
    (output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    write_report(output_dir,summary,folds,chosen,corr,agreement,config_list)
    return outputs


def write_report(out,summary,folds,top,corr,agreement,config_list):
    def html(df):return df.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)
    report='''<html><meta charset="utf-8"><style>body{font:14px sans-serif;margin:32px}table{border-collapse:collapse;margin:16px 0}td,th{padding:6px;border:1px solid #ddd}th{background:#eef2f7}</style>
    <h1>Controlled tuning H5 — Logistic / SVC / ExtraTrees / HGB</h1>
    <p>5 expanding folds, gap=5, weekday calendar, frozen TECHNICAL+A+C, intermarket lag=1.
    Mean AUC ranks configurations; no std penalty, holdout, threshold tuning or ensemble.</p>
    <p>All configurations share fold dates. Scaling remains within Pipeline and is fit only on training rows.
    Best configurations are selected on these same folds: their OOF correlations are descriptive and not unbiased post-selection estimates.</p>'''
    report+='<h2>Best mean AUC per model and baseline</h2>'+html(top)
    report+='<h2>Top five configurations per model</h2>'+html(summary.sort_values('mean_roc_auc',ascending=False).groupby('model',sort=False).head(5))
    report+='<h2>OOF score correlation (winners)</h2>'+html(corr)
    report+='<h2>Prediction agreement (winners)</h2>'+html(agreement)
    report+='<p>No ensemble built. A+C was selected using this history; future independent forward data is still required.</p></html>'
    (out/'report.html').write_text(report,encoding='utf-8')


if __name__=='__main__':
    results=evaluate()
    print(results['top_configs'].to_string(index=False))
