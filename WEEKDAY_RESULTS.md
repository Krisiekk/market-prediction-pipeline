# Kalendarz poniedziałek–piątek: benchmark i ablation

Stały filtr `src/features/calendar.py::filter_weekdays` jest wywoływany na wejściu
`build_features`, przed wszystkimi wskaźnikami, intermarket, lagiem i targetem.
RAW Twelve Data pozostaje bez zmian (zweryfikowane SHA256 wszystkich plików).
Nowe pobrania przechodzą przez tę samą funkcję przy każdym budowaniu datasetu ML.
Filtr kopiuje dane, konwertuje datetime, sortuje, usuwa weekendy, odrzuca duplikaty
dni i resetuje indeks. Nie dodaje weekday do X.

## Kalendarz i target

- RAW: 5000 wierszy, 2008-02-22–2026-09-27.
- Usunięto 101 sobót i 58 niedziel: razem 159.
- Po filtrze: 4841 wierszy, 2008-02-22–2026-09-25.
- H1: 4790 obserwacji, 2008-05-02–2026-09-24.
- H5: 4786 obserwacji, 2008-05-02–2026-09-18.
- H5 = piąty kolejny dostępny rekord poniedziałek–piątek. To nie kalendarz świąt.
- `label_end_date` i `future_close` służą diagnostyce i nie trafiają do X.
  `label_end` w walidacji jest aliasem tej samej daty, a nie przesunięciem po dropna.

Przykłady z odtworzonych danych (weekday tylko w raporcie):

| datetime | close | label_end_date | future_close | target | horizon | weekday | label_end_weekday |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-04-24 | 3354.900000 | 2025-04-25 | 3316.300000 | 0 | 1 | Thursday | Friday |
| 2026-09-17 | 4341.902050 | 2026-09-18 | 4377.245250 | 1 | 1 | Thursday | Friday |
| 2026-09-18 | 4377.245250 | 2026-09-21 | 4343.809640 | 0 | 1 | Friday | Monday |
| 2025-04-24 | 3354.900000 | 2025-05-01 | 3238.640000 | 0 | 5 | Thursday | Thursday |
| 2026-09-17 | 4341.902050 | 2026-09-24 | 4275.284100 | 0 | 5 | Thursday | Thursday |
| 2026-09-18 | 4377.245250 | 2026-09-25 | 4286.739730 | 0 | 5 | Friday | Friday |

## Benchmark

Najwyższe średnie AUC w tym przebiegu (bez dokonywania nowego wyboru konfiguracji):

- H1: LogisticRegression TECHNICAL — 0,511902, std 0,016163.
- H5: LogisticRegression SELECTED — 0,531437, std 0,035928.
- H5 LogisticRegression TECHNICAL — 0,508646.
- H5 LogisticRegression ALL — 0,488216.
- Always Up H1: accuracy 52,3308%, balanced accuracy 50%.
- Always Up H5: accuracy 54,0025%, balanced accuracy 50%.

Zachowano 6 rodzin, identyczne parametry, 3 feature sets, 5 expanding-window foldów,
gap=1/5, skalowanie w Pipeline i te same metryki. Numeryczne indeksy i granice dat
muszą zmienić się po usunięciu wierszy, ale w obrębie horyzontu są identyczne dla
wszystkich wariantów. Nie wykonywano tuningu ani wyboru progu.

## Ablation H5 LogisticRegression

A=dfii10_vs_sma20; B=dxy_vs_sma50; C=dxy_sma20_vs_sma50.
Każdy wariant obejmuje również komplet TECHNICAL.

| features | auc_mean | auc_std | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | delta_vs_technical_pp | positive_fold_delta_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TECHNICAL | 0.508646 | 0.038048 | 0.512451 | 0.514616 | 0.489774 | 0.564880 | 0.461509 | 0.000000 | 0 |
| TECHNICAL + A | 0.523940 | 0.041680 | 0.519096 | 0.518844 | 0.505333 | 0.593647 | 0.482778 | 1.529338 | 5 |
| TECHNICAL + B | 0.514370 | 0.042748 | 0.512495 | 0.514907 | 0.493159 | 0.583098 | 0.468192 | 0.572417 | 5 |
| TECHNICAL + C | 0.519562 | 0.036757 | 0.519866 | 0.522660 | 0.511883 | 0.573306 | 0.470093 | 1.091546 | 5 |
| TECHNICAL + A+B | 0.526411 | 0.044256 | 0.519128 | 0.519096 | 0.507410 | 0.601530 | 0.484891 | 1.776499 | 5 |
| TECHNICAL + A+C | 0.529313 | 0.041534 | 0.524050 | 0.526066 | 0.524516 | 0.594063 | 0.477870 | 2.066684 | 5 |
| TECHNICAL + B+C | 0.518834 | 0.029728 | 0.524536 | 0.528373 | 0.521157 | 0.550374 | 0.469729 | 1.018771 | 4 |
| TECHNICAL + A+B+C | 0.531437 | 0.035928 | 0.533889 | 0.535855 | 0.535240 | 0.576431 | 0.475770 | 2.279073 | 5 |

## Czy sygnał przetrwał?

**Tak, jakościowo jako względna poprawa nad TECHNICAL, ale nie jako stabilna
przewaga nad losowym rankingiem w każdym foldzie.** A+C ma AUC 0,529313,
+2,066684 pp wobec TECHNICAL; poprawia każdy z 5 foldów. Cztery foldy mają AUC>0,5,
ale ostatni ma 0,477870. Fold 4 jest wyraźnie mocniejszy (0,594063), std wynosi
0,041534. Większa niestabilność wymaga ostrożności.

Historyczne A+C 0,537450 jest wyłącznie punktem odniesienia: zmieniły się daty,
rolling windows, targety i granice foldów. To nie jest porównanie apples-to-apples.
Nie dokonano kolejnej selekcji: A+C pozostaje dotychczasowym kandydatem, nie
przełączono modelu ani zestawu na konfigurację z najwyższym nowym AUC.
Nie uruchomiono i nie utworzono holdoutu. Ta historia była już analizowana;
późniejszy forward holdout musi używać nowych, wcześniej nieocenianych obserwacji.

## Weryfikacja i artefakty

Przeszło 12 testów: filtr nowych danych, kopiowanie/idempotencja, daty i duplikaty,
H1/H5 przez weekend, rolling dopiero po filtrze, lag, brak targetu i diagnostyki w X,
prefix invariance rzeczywistych cech, backward-asof, gap oraz metryki.
Sprawdzono zgodność modeli i feature sets ze starymi metadanymi, wszystkie 220
fitów, wspólne daty/targety oraz metryki odtworzone z OOF predictions.
Ablation dokładnie odtwarza kontrolne predykcje nowego benchmarku TECHNICAL/SELECTED.
Intermarket.py pozostał bez zmian; backward asof i lag=1 nadal obowiązują.
Brak godzin publikacji/vintage FRED pozostaje ograniczeniem point-in-time.

Aktualne wyniki: `data/processed/benchmark_weekdays/` i
`data/processed/ablation_h5_weekdays/` (summary, folds, predictions, metadata, HTML).
Raport filtrowania i przykłady: `data/processed/weekday_calendar/`.
Historyczne benchmark/ablation pozostawiono w starych katalogach.
Standardowe processed H1/H5 odbudowano od RAW. Notebook filtruje dane przed
bezpośrednimi wskaźnikami; stare outputy wyczyszczono, stare opisy oznaczono.
