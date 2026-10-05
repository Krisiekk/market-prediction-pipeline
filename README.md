# Market prediction pipeline

## Raport końcowy projektu badawczego

[Raport końcowy](FINAL_PROJECT_REPORT.md) zbiera metodologię, wyniki modeli,
ensemble, backtestu po kosztach oraz testów reżimów zmienności. Zawiera też
instrukcję odtworzenia eksperymentów i ograniczenia badania.
[Tabele źródłowe raportu](reports/final/README.md) są małym snapshotem wyników,
przeznaczonym do wersjonowania razem z raportem.

Status: zakończona seria eksperymentów badawczych; brak niezależnego holdoutu
i potwierdzonej przewagi inwestycyjnej. Plan budowy agenta transakcyjnego
pozostaje szerszym kierunkiem rozwoju, a nie listą ukończonych funkcji.

## Benchmark kierunku XAU/USD

Uruchom z katalogu projektu (środowisko wymaga numpy, pandas, scikit-learn i xgboost):

```bash
.venv/bin/python -m src.models.benchmark
.venv/bin/python -m unittest discover -s tests -v
```

Opcjonalnie: `--output-dir /ścieżka/do/wyników` (domyślnie `data/processed/benchmark_weekdays`).
Benchmark korzysta z istniejących surowych CSV, `build_features`, `indicators`,
`intermarket` i `create_time_series_splits`. Nie pobiera danych z sieci.
Ponowne uruchomienie nadpisuje tylko pliki benchmarku w wybranym katalogu.

- H1 i H5, 5 expanding-window foldów, gap odpowiednio 1 i 5.
- 3 zestawy × 6 rodzin = 36 konfiguracji / 180 dopasowań.
- TECHNICAL: 14 cech z dotychczasowego `train.py` (bez powielającego SMA20 `BB_MIDDLE`).
- SELECTED: zadane z góry 2 cechy intermarket H1 / 3 cechy H5.
- ALL: wszystkie kolumny dodane przez `intermarket.py`, **włącznie z poziomami**
  `dxy`, `dgs10`, `dfii10`, `usd_broad`. Listy zapisane w `metadata.json`.
- Wspólna kompletna próba dla wszystkich konfiguracji danego horyzontu.
  TECHNICAL również oceniany tylko na datach dostępnych dla ALL.
- LR/SVC skalowane w Pipeline fitowanym wyłącznie na treningu. SVC AUC z
  `decision_function`, bez kalibracji. HGB bez wewnętrznego losowego early stopping.
- Stałe parametry, bez tuningu i bez ensemble. RF/ExtraTrees mają 200 drzew,
  min_samples_leaf=5. Wszystkie parametry i wersje bibliotek zapisane w metadanych.

### Pliki wynikowe

`report.html` — czytelne tabele; `summary.csv` — średnie i odchylenia standardowe
próbki (ddof=1); `folds.csv` — wszystkie wyniki foldów; `stability.csv` — AUC
każdego foldu; `feature_comparison.csv` / `paired_folds.csv` — różnice względem
TECHNICAL w pp AUC; `baseline_summary.csv` / `baseline_folds.csv` — Always Up bez
AUC; `splits.csv` / `coverage.csv` — daty i liczebności; `predictions.csv` —
predykcje out-of-fold (score SVC to margines, nie prawdopodobieństwo);
`metadata.json` — cechy, parametry, wersje i SHA256 wejściowych CSV.

Raport `BENCHMARK_RESULTS.md` dokumentuje historyczny przebieg sprzed filtra
weekendów. Aktualny przebieg na kalendarzu poniedziałek–piątek opisuje
`WEEKDAY_RESULTS.md`.

### Ograniczenia interpretacji

Predykcja po zamknięciu XAU. Intermarket opóźniony o jedną obserwację przed
usuwaniem braków, zgodnie z exploration. Backward asof nie bierze przyszłych dat,
ale brak godzin publikacji i historycznych wersji FRED oznacza, że to nie jest
pełna rekonstrukcja danych point-in-time. Dłuższa przerwa źródła może pozostawić
nieaktualną wartość. Horyzonty oznaczają kolejne dostępne obserwacje poniedziałek–piątek, bez dodatkowego
kalendarza świąt. Weekend jest usuwany przed feature engineering i targetem.

SELECTED wybrano wcześniej na tej historii, więc CV nie jest niezależną walidacją
selekcji. Etykiety H5 w obrębie testu nakładają się. Std foldów nie jest przedziałem
ufności; AUC 0,51–0,52 nie oznacza potwierdzonej przewagi tradingowej.
Po wyborze konfiguracji konieczny jest całkowicie nietknięty okres holdout.

## Kontrolowane ablation H5 LogisticRegression

```bash
.venv/bin/python -m src.models.ablation
```

Osiem kombinacji trzech cech SELECTED, na dokładnie tej samej kompletnej próbie
co benchmark. Wymaga zapisanych wyników benchmarku; kontroluje hashe wejść,
daty foldów oraz odtworzenie predykcji TECHNICAL i SELECTED. Wyniki trafiają do
`data/processed/ablation_h5_weekdays` (opcje `--output-dir`, `--benchmark-dir`).
Historyczny raport z poprzednim kalendarzem: [ABLATION_H5_RESULTS.md](ABLATION_H5_RESULTS.md).
Aktualny wynik na kalendarzu poniedziałek–piątek: [WEEKDAY_RESULTS.md](WEEKDAY_RESULTS.md).

## Audyt kalendarza przed holdoutem

```bash
.venv/bin/python -m src.validation.calendar_audit
```

Audyt nie trenuje modelu i nie zmienia danych. Raport: [CALENDAR_AUDIT.md](CALENDAR_AUDIT.md),
pliki szczegółowe: `data/processed/calendar_audit/`. Historyczny audyt dotyczył kalendarza z weekendami. Obecnie H5 liczy rekordy
poniedziałek–piątek. Holdout nadal nie jest uruchamiany: najnowszą historię już oceniano.


## Aktualny kalendarz ML (poniedziałek–piątek)

RAW zachowujemy bez zmian. `build_features` zawsze wywołuje `filter_weekdays`
przed wskaźnikami, intermarket, lagiem i targetem. Domyślny intermarket lag=1.
Diagnosticzne `label_end_date` i `future_close` nigdy nie trafiają do X.
Filtr poniedziałek–piątek pozostaje roboczą polityką eksperymentu: audyt źródła
potwierdził prezentowaną strefę i obecność weekendowych świec, ale nie rozstrzygnął
ich reprezentatywności dla celu tradingowego. Szczegóły i ograniczenia są w
[`CALENDAR_AUDIT.md`](CALENDAR_AUDIT.md).

```bash
.venv/bin/python -m src.features.build_features
.venv/bin/python -m src.models.benchmark
.venv/bin/python -m src.models.ablation
```

Pierwsze polecenie odbudowuje oba standardowe processed CSV (H1 i H5) z RAW.
Aktualne wyniki trafiają do `benchmark_weekdays` i `ablation_h5_weekdays`;
stare katalogi pozostają historyczne. [Wyniki i raport kalendarza](WEEKDAY_RESULTS.md).
Holdoutu nie wykonujemy. Daty i indeksy foldów zmieniają się z kalendarzem,
metodologia TimeSeriesSplit pozostaje identyczna.

## Controlled tuning H5, TECHNICAL + A+C

```bash
.venv/bin/python -m src.models.tune_h5
```

Runs fixed 5-fold expanding-window CV (gap=5), baselines plus a bounded parameter
set for LogisticRegression, RBF SVC, ExtraTrees, and HistGradientBoosting.
No holdout or ensemble. Detailed outputs are stored separately in
`data/processed/tuning_h5_ac/`.

## Fixed H5 A+C ensemble diagnostics

```bash
.venv/bin/python -m src.models.ensemble_h5
```

Reuses the saved tuning OOF predictions (no model refits) to evaluate only the
predeclared equal-weight ensemble variants. Outputs are written to
`data/processed/ensemble_h5_ac/`; no holdout or calibration is performed.

## H5 execution-aligned models, ensembles, AR, and MLP

```bash
.venv/bin/python -m src.models.compare_h5_execution_models
```

Development-only comparison for a signal after `close(t)`, entry at the next
weekday `open(t+1)`, and exit at `close(t+5)`. It evaluates fixed LogisticRegression,
ExtraTrees, SVC, a lagged-return AR classifier, a small MLP, and fixed hard/soft/rank
ensembles on the same five expanding-window folds (gap=5). The standard dataset
and its close-to-close target remain unchanged. Detailed methodology and results:
[`H5_EXECUTION_MODEL_COMPARISON.md`](H5_EXECUTION_MODEL_COMPARISON.md); machine outputs:
`data/processed/h5_execution_model_comparison/`. No costs or holdout are included.

## H5: tuning two extra majority voters

```bash
.venv/bin/python -m src.models.tune_h5_execution_ensemble
```

Tunes Random Forest and HistGradientBoosting only inside each outer training fold,
then compares majority voting with the existing three voters against five voters.
Methodology and results: [`H5_EXECUTION_ENSEMBLE_TUNING.md`](H5_EXECUTION_ENSEMBLE_TUNING.md);
outputs: `data/processed/h5_execution_ensemble_tuning/`. Development walk-forward only;
no transaction costs or independent holdout.

## H5: ARX-style return regression and six-voter check

```bash
.venv/bin/python -m src.models.evaluate_h5_arx
```

Fits a Ridge ARX-style return regression with lagged XAU returns and exogenous
technical/intermarket features, then checks a six-voter majority with abstention
on ties. Results and interpretation: [`H5_EXECUTION_ARX.md`](H5_EXECUTION_ARX.md);
outputs: `data/processed/h5_execution_arx/`.

## H5: cost-aware execution backtest

```bash
.venv/bin/python -m src.models.backtest_h5_execution_oof
```

Replays saved execution-aligned OOF signals as non-overlapping long-only and
long/short trades, against Always Long, over fixed round-trip cost scenarios.
Report: [`H5_EXECUTION_COST_BACKTEST.md`](H5_EXECUTION_COST_BACKTEST.md);
outputs: `data/processed/h5_execution_cost_backtest/`. The cost levels are
sensitivity scenarios, not broker quotes; this is not a holdout result.

## H5: volatility-regime analysis

```bash
.venv/bin/python -m src.models.analyze_h5_volatility_regimes
```

Classifies signals as low/medium/high realized volatility using fold-local
training terciles. Report: [`H5_VOLATILITY_REGIMES.md`](H5_VOLATILITY_REGIMES.md);
outputs: `data/processed/h5_execution_volatility_regimes/`. Exploratory analysis,
not an independent test of a volatility filter.

## H5: fixed trading-filter hypotheses

```bash
.venv/bin/python -m src.models.test_h5_trading_hypotheses
```

Compares low-volatility abstention, SMA50 trend, and positive 20-day momentum
filters against matched Always Long strategies. Report:
[`H5_TRADING_HYPOTHESES.md`](H5_TRADING_HYPOTHESES.md); outputs:
`data/processed/h5_trading_hypotheses/`. Exploratory OOF analysis, not a holdout.

## H5 signal strength and abstention diagnostics

```bash
.venv/bin/python -m src.models.signal_strength_h5
```

Uses the saved ensemble OOF scores only; coverage selection is fold-local and
does not use labels or future returns. Reports are written to
`data/processed/signal_strength_h5/`. This step is not a trading backtest.
