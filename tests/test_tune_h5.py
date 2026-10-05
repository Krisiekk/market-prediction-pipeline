import unittest
from pathlib import Path
import pandas as pd
from src.config import DATA_RAW_DIR
from src.models.tune_h5 import validate_frozen_data
from src.models.benchmark import prepare_dataset
from src.models.tune_h5 import configurations, configs_for_model, FEATURES
from src.models.benchmark import make_models


class TuneH5Tests(unittest.TestCase):

    def test_frozen_weekday_data_and_fold_purge(self):
        raw=pd.read_csv(DATA_RAW_DIR/'xauusd_daily.csv',parse_dates=['datetime'])
        data,sets=prepare_dataset(raw,5)
        splits,_=validate_frozen_data(data,sets,raw,Path('data/processed/benchmark_weekdays'))
        self.assertEqual(len(splits),5)
        self.assertTrue((data.datetime.dt.dayofweek<5).all())
        self.assertTrue(data.label_end_date.dt.dayofweek.lt(5).all())
        columns=sets['TECHNICAL']+['dfii10_vs_sma20','dxy_sma20_vs_sma50']
        self.assertFalse(set(columns)&{'target','future_close','label_end_date','label_end','weekday'})
        for tr,va in splits:
            self.assertEqual(va[0]-tr[-1]-1,5)
            self.assertLess(data.iloc[tr].label_end_date.max(),data.iloc[va].datetime.min())

    def test_bounded_grids_and_benchmark_baselines(self):
        grids=configurations();models=make_models()
        self.assertEqual(set(grids), {'LogisticRegression','SVC','ExtraTreesClassifier','HistGradientBoostingClassifier'})
        built={name:configs_for_model(name,base,candidates) for name,(base,candidates) in grids.items()}
        self.assertEqual([len(built[x]) for x in grids], [22,96,40,40])
        self.assertTrue(all(sum(c['is_baseline'] for c in cfgs)==1 for cfgs in built.values()))
        self.assertEqual(FEATURES, ('dfii10_vs_sma20','dxy_sma20_vs_sma50'))
        lr=next(x for x in built['LogisticRegression'] if x['is_baseline'])
        self.assertEqual(lr['estimator'].named_steps['logisticregression'].get_params(),models['LogisticRegression'].named_steps['logisticregression'].get_params())
        svc=next(x for x in built['SVC'] if x['is_baseline'])
        self.assertEqual(svc['estimator'].named_steps['svc'].get_params(),models['SVC'].named_steps['svc'].get_params())
        self.assertFalse(svc['estimator'].named_steps['svc'].probability)
        self.assertFalse(built['HistGradientBoostingClassifier'][0]['estimator'].early_stopping)


if __name__ == '__main__':
    unittest.main()
