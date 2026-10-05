# Random Forest i HistGradientBoosting w ensemble H5

## Cel i walidacja

Dodano Random Forest i HistGradientBoosting jako dwóch kandydatów do istniejącego
ensemble Logistic Regression + Extra Trees + SVC. Target pozostał wykonalnym
wariantem badawczym: sygnał po `close(t)`, wejście na `open(t+1)`, wyjście na
`close(t+5)`, kierunek zwrotu. Standardowy target i pipeline nie zostały zmienione.

Parametry Random Forest (16 kombinacji) i HistGradientBoosting (32 kombinacje)
wybierano osobno wewnątrz części treningowej każdego z pięciu zewnętrznych foldów.
W każdym outer foldzie użyto 3 chronologicznych inner foldów z `gap=5`; asercja
sprawdzała, że etykiety treningowe kończą się przed początkiem walidacji. Parametry
wybierano według średniego ROC AUC inner CV. Wyniki w tabeli pochodzą wyłącznie z
zewnętrznych foldów, których nie użyto w strojeniu danego modelu.

Ensemble 3 głosów to Logistic Regression, Extra Trees i SVC, po co najmniej 2 głosy
UP. Ensemble 5 głosów dodaje dostrojony Random Forest i HistGradientBoosting, po co
najmniej 3 głosy UP. Głosy są binarnymi predykcjami `predict()`; nie optymalizowano
progów ani wag.

## Wyniki zewnętrznej walidacji kroczącej

| Model | Średnie AUC ± std | Foldy AUC > 0,5 | Accuracy | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Random Forest, strojony wewnętrznie | 0,5117 ± 0,0323 | 3/5 | 0,4949 | 0,5172 |
| HistGradientBoosting, strojony wewnętrznie | 0,5046 ± 0,0184 | 3/5 | 0,4841 | 0,5075 |
| Ensemble większościowe, 3 modele | 0,5130 ± 0,0346 | 4/5 | 0,4954 | 0,5017 |
| Ensemble większościowe, 5 modeli | **0,5166 ± 0,0299** | 3/5 | 0,4853 | 0,5065 |
| Zawsze UP (baseline) | — | — | **0,5370** | 0,5000 |

Ensemble z pięcioma modelami podnosi średnie AUC o około 0,0035 względem ensemble
3-modelowego, ale accuracy spada o około 0,010. Nadal jest dużo niższe niż Always UP.
Nie ma więc obecnie podstaw, by uznać dwa dodatkowe głosy za poprawę jakości decyzji.
Wyniki różnią się pomiędzy okresami; nie jest to niezależny holdout ani dowód
przewagi tradingowej.

Wybrane parametry różniły się między foldami, zgodnie z osobnym strojeniem:

| Fold | Random Forest | HistGradientBoosting |
| --- | --- | --- |
| 1 | depth 8, sqrt, leaf 10, 400 drzew | lr 0,03, 100 iter., 7 liści, leaf 20, L2 0,1 |
| 2 | depth 8, sqrt, leaf 5, 200 drzew | lr 0,03, 100 iter., 7 liści, leaf 10, L2 0,1 |
| 3 | depth 8, sqrt, leaf 5, 200 drzew | lr 0,05, 200 iter., 7 liści, leaf 10, L2 0,1 |
| 4 | depth 8, sqrt, leaf 5, 200 drzew | lr 0,03, 100 iter., 15 liści, leaf 20, L2 0,1 |
| 5 | bez limitu depth, sqrt, leaf 5, 200 drzew | lr 0,05, 100 iter., 15 liści, leaf 20, L2 0,1 |

## Odtworzenie i pliki

```bash
.venv/bin/python -m src.models.tune_h5_execution_ensemble
.venv/bin/python -m unittest tests.test_tune_h5_execution_ensemble -v
```

Kod: `src/models/tune_h5_execution_ensemble.py`. Wyniki maszynowe: `data/processed/h5_execution_ensemble_tuning/`.
`fold_tuning_results.csv` zawiera ustawienia wybrane wewnętrznie i ich outer-fold
metryki; `ensemble_oof_predictions.csv` zawiera trzy- i pięciogłosowe decyzje.
Historia była wcześniej analizowana. Nie wykonano kosztowego backtestu ani holdoutu.
