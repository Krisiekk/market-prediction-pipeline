"""Read-only calendar/target audit. Never trains a model or creates a holdout.

Run: python -m src.validation.calendar_audit
Weekday counts are a diagnostic proxy, NOT a verified venue/session calendar.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.models.benchmark import prepare_dataset
from src.features.calendar import filter_weekdays

OHLC = ['open', 'high', 'low', 'close']
DAYS = ['poniedziałek', 'wtorek', 'środa', 'czwartek', 'piątek', 'sobota', 'niedziela']


def calendar_stats(data):
    counts = data.datetime.dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
    delta = data.datetime.diff().dt.total_seconds() / 86400
    gaps = {str(i): int(delta.eq(i).sum()) for i in (1, 2, 3)}
    gaps['other'] = int((delta.notna() & ~delta.isin([1, 2, 3])).sum())
    return {'start': str(data.datetime.min()), 'end': str(data.datetime.max()), 'rows': len(data),
            'duplicates': int(data.datetime.duplicated().sum()),
            'weekday_counts': dict(zip(DAYS, counts.astype(int).tolist())),
            'gaps_days': gaps, 'exact_gaps_days': {str(k): int(v) for k, v in delta.value_counts().items()}}


def target_audit(raw, horizon=5):
    rows = []
    for i in range(len(raw) - horizon):
        following = raw.iloc[i+1:i+horizon+1]
        rows.append(dict(datetime=raw.datetime.iloc[i], future_date=raw.datetime.iloc[i+horizon],
                         close=raw.close.iloc[i], future_close=raw.close.iloc[i+horizon],
                         calendar_days=(raw.datetime.iloc[i+horizon]-raw.datetime.iloc[i]).days,
                         weekday_records=int((following.datetime.dt.dayofweek < 5).sum()),
                         weekend_records=int((following.datetime.dt.dayofweek >= 5).sum())))
    return pd.DataFrame(rows)


def run_audit(output_dir=DATA_PROCESSED_DIR / 'calendar_audit_weekdays'):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_RAW_DIR / 'xauusd_daily.csv', parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
    processed, sets = prepare_dataset(raw, horizon=5)
    stats = {'raw': calendar_stats(raw), 'processed_h5_lag1': calendar_stats(processed)}
    # Also inspect the legacy materialized feature file without overwriting it.
    legacy_path = DATA_PROCESSED_DIR / 'xauusd_daily_features_h5.csv'
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path, parse_dates=['datetime']).sort_values('datetime')
        stats['legacy_h5_csv_not_used_by_benchmark'] = calendar_stats(legacy)
    weekend = raw.loc[raw.datetime.dt.dayofweek >= 5].copy()
    previous = raw[OHLC].shift(1)
    weekend['same_ohlc_as_previous'] = raw[OHLC].eq(previous).all(axis=1).loc[weekend.index]
    weekend['same_close_as_previous'] = raw.close.eq(raw.close.shift()).loc[weekend.index]
    weekend['flat_bar'] = weekend[OHLC].nunique(axis=1).eq(1)
    weekend['valid_ohlc'] = (np.isfinite(weekend[OHLC]).all(axis=1) & weekend[OHLC].gt(0).all(axis=1)
        & weekend.high.ge(weekend[['open', 'close', 'low']].max(axis=1))
        & weekend.low.le(weekend[['open', 'close', 'high']].min(axis=1)))
    friday = raw.loc[raw.datetime.dt.dayofweek == 4, ['datetime'] + OHLC]
    aligned = pd.merge_asof(weekend[['datetime']], friday.rename(columns={'datetime': 'friday_date'}),
                            left_on='datetime', right_on='friday_date', direction='backward')
    weekend['friday_date'] = aligned.friday_date.to_numpy()
    weekend['same_ohlc_as_friday'] = (weekend[OHLC].to_numpy() == aligned[OHLC].to_numpy()).all(axis=1)
    weekend['same_close_as_friday'] = weekend.close.to_numpy() == aligned.close.to_numpy()
    weekend['range_pct'] = 100 * (weekend.high - weekend.low) / weekend.close
    raw_range = 100 * (raw.high - raw.low) / raw.close
    stats['weekend_checks'] = {c: int(weekend[c].sum()) for c in [
        'same_ohlc_as_previous', 'same_close_as_previous', 'flat_bar', 'valid_ohlc',
        'same_ohlc_as_friday', 'same_close_as_friday']}
    sunday = weekend[weekend.datetime.dt.dayofweek == 6]
    stats['weekend_checks']['sunday_with_previous_saturday'] = int(
        ((raw.datetime - raw.datetime.shift()).loc[sunday.index].dt.days == 1).sum())
    stats['weekend_checks']['sunday_same_close_as_previous'] = int(sunday.same_close_as_previous.sum())
    stats['range_pct_median'] = {'weekday': float(raw_range[raw.datetime.dt.dayofweek < 5].median()),
                               'saturday': float(weekend.loc[weekend.datetime.dt.dayofweek == 5, 'range_pct'].median()),
                               'sunday': float(sunday.range_pct.median())}
    examples = []
    for date in [weekend.datetime.iloc[0], sunday.datetime.iloc[0], sunday.datetime.iloc[-2]]:
        friday_date = date - pd.Timedelta(days=date.dayofweek - 4)
        frame = raw.set_index('datetime').reindex(pd.date_range(friday_date, periods=4)).rename_axis('datetime').reset_index()
        frame.insert(0, 'example_friday', friday_date)
        frame['weekday'] = frame.datetime.dt.dayofweek.map(dict(enumerate(DAYS)))
        examples.append(frame)
    targets = target_audit(filter_weekdays(raw))
    check = processed.merge(targets, on='datetime', suffixes=('', '_audit'), validate='one_to_one')
    assert (check.label_end == check.future_date).all()
    assert (check.target == (check.future_close > check.close).astype(int)).all()
    affected = targets[targets.weekend_records > 0]
    stats['h5'] = {'definition': '5 successive Monday-Friday rows, before dropna',
                   'total_labelled_raw_rows': len(targets), 'windows_with_weekends': len(affected),
                   'processed_windows_with_weekends': int((check.weekend_records > 0).sum()),
                   'first_affected_input': str(affected.datetime.min()),
                   'actual_trading_session_count': 'unverified: no source timezone/session calendar in CSV'}
    # Report possible tail dates without selecting a boundary or running a model.
    candidates = []
    old_predictions = pd.read_csv(DATA_PROCESSED_DIR / 'benchmark/predictions.csv', usecols=['horizon', 'datetime'])
    seen = set(pd.to_datetime(old_predictions.loc[old_predictions.horizon == 5, 'datetime']))
    for fraction in (.15, .20):
        tail = processed.iloc[int(np.floor(len(processed) * (1-fraction))):]
        candidates.append(dict(fraction=fraction, start=tail.datetime.min(), end=tail.datetime.max(),
                               rows=len(tail), already_evaluated_cv_rows=int(tail.datetime.isin(seen).sum())))
    source_checks = []
    for filename, date_col, value_col in [
        ('dxy_reconstructed_daily.csv', 'datetime', 'dxy'),
        ('dgs10_daily.csv', 'date', 'value'), ('dfii10_daily.csv', 'date', 'value'),
        ('usd_broad_daily.csv', 'date', 'value'),
    ]:
        source = pd.read_csv(DATA_RAW_DIR / filename, parse_dates=[date_col]).dropna(subset=[value_col])
        source = source[[date_col]].rename(columns={date_col: 'source_date'}).sort_values('source_date')
        matched = pd.merge_asof(raw[['datetime']], source, left_on='datetime', right_on='source_date', direction='backward')
        used = matched.source_date.shift(1)
        assert (used.dropna() <= matched.loc[used.notna(), 'datetime']).all()
        source_checks.append(dict(source=filename, future_source_dates=int((used > matched.datetime).sum()),
                                  max_age_days=int((matched.datetime - used).dt.days.max())))
    stats['decision'] = {'holdout_run': False, 'status': 'STOP_AFTER_AUDIT',
        'reason': 'Weekday calendar active. No holdout: latest history already used by CV/ablation.'}
    tables = {'intermarket_date_checks': pd.DataFrame(source_checks), 'weekend_records': weekend, 'weekend_examples': pd.concat(examples),
              'target_h5_mapping': targets, 'target_h5_affected_examples': affected.head(12),
              'tail_candidates_not_selected': pd.DataFrame(candidates),
              'weekends_by_year': weekend.groupby(weekend.datetime.dt.year).size().rename('rows').reset_index()}
    for name, table in tables.items():
        table.to_csv(out / f'{name}.csv', index=False)
    (out / 'audit.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
    html = '<html><meta charset="utf-8"><h1>Audyt kalendarza — STOP przed holdoutem</h1>'
    html += '<p>H5 = pięć kolejnych rekordów poniedziałek–piątek. Zmienne weekendowe OHLC nie dowodzą autentyczności sesji. Brak strefy czasowej i identyfikatora sesji.</p>'
    html += '<p>Brak dowodu prostego forward-fill nie jest dowodem poprawnego kalendarza. Najnowsze 15–20% historii już oceniano w CV; nie stanowi nietkniętego holdoutu.</p>'
    html += '<pre>' + json.dumps(stats, ensure_ascii=False, indent=2) + '</pre>'
    for name in ['weekend_examples', 'target_h5_affected_examples', 'tail_candidates_not_selected', 'weekends_by_year']:
        html += f'<h2>{name}</h2>' + tables[name].to_html(index=False)
    (out / 'report.html').write_text(html + '</html>', encoding='utf-8')
    return stats, tables


if __name__ == '__main__':
    stats, tables = run_audit()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    for key in ['weekend_examples', 'target_h5_affected_examples', 'tail_candidates_not_selected']:
        print(key, '\n', tables[key].to_string(index=False))
