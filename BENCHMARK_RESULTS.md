# Benchmark XAU/USD — wyniki bieżącej próby

36 konfiguracji / 180 dopasowań, bez tuningu. Wyniki wygenerowane komendą
`.venv/bin/python -m src.models.benchmark`. Pełny raport: `data/processed/benchmark/report.html`.

H1: 4949 obserwacji; H5: 4945. W każdym horyzoncie wspólne daty i 5 foldów
(gap=1 / 5). Predykcja po zamknięciu; intermarket lag=1, przed dropna.
ALL obejmuje 26 kolumn intermarket (22 pochodne + 4 poziomy), TECHNICAL 14 cech.
Horyzonty liczone na kolejnych wierszach dziennych źródła; rekordy weekendowe
nie są usuwane. Selected pochodzi z wcześniejszej eksploracji tej samej historii.

## Wnioski

- **H5 wygląda ciekawiej**, głównie dla LogisticRegression + SELECTED: AUC 0,5379,
  std 0,0188, każdy fold powyżej 0,5. Poprawa względem TECHNICAL +1,87 pp,
  dodatnia w 4/5 foldów. To kandydat do kolejnego kontrolowanego eksperymentu.
- **H1:** największa średnia to HGB + SELECTED 0,5150 ± 0,0171 (3/5 foldów >0,5).
  SVC TECHNICAL jest stabilniejszy: 0,5123 ± 0,0080 i 5/5 foldów >0,5,
  ale efekt jest niewielki. SELECTED poprawia HGB w 5/5 foldów (+1,06 pp).
- Każda z 6 rodzin ma co najmniej jeden wariant AUC >0,5 dla H1 oraz H5;
  nie oznacza to, że wszystkie warianty danej rodziny mają przewagę.
- SELECTED poprawia średnie AUC w 4/6 rodzin H1 i 6/6 H5. W H5 poprawa
  RF i ExtraTrees jest praktycznie zerowa (+0,009 pp).
- **ALL pogarsza AUC wszystkich 6 rodzin w obu horyzontach.**
  W H1 żaden ALL nie przekracza 0,5; w H5 przekracza tylko ExtraTrees (0,5042).
- ExtraTrees H5 ma średnie około 0,520, ale ostatni fold spada do około 0,47.
  Nie wygląda tak stabilnie jak regresja SELECTED.
- Always Up: accuracy H1 52,11%, H5 53,71%; balanced accuracy 50%.
  Żaden model nie pokonuje Always Up średnią accuracy przy domyślnym progu.
  Regresja H5 SELECTED ma accuracy 49,34% i balanced accuracy 50,33%,
  więc poprawa rankingu AUC nie przekłada się jeszcze na dobry podział klas.
- Większość wyników pozostaje blisko losowości. Nie ma podstaw do ogłoszenia
  przewagi tradingowej ani uruchomienia ensemble. Najpierw sprawdzenie stabilności
  regresji H5 SELECTED i rzeczywistej dostępności danych; po wyborze konfiguracji
  konieczny całkowicie nietknięty holdout. Nie dobierano tutaj progu ani parametrów.

## Porównanie AUC i wpływu cech

| horizon | model | ALL | SELECTED | TECHNICAL | selected_delta_pp | all_delta_pp |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ExtraTreesClassifier | 0.4948 | 0.5028 | 0.5018 | 0.1019 | -0.6904 |
| 1 | HistGradientBoostingClassifier | 0.4996 | 0.5150 | 0.5044 | 1.0589 | -0.4757 |
| 1 | LogisticRegression | 0.4865 | 0.5086 | 0.5087 | -0.0178 | -2.2234 |
| 1 | RandomForestClassifier | 0.4998 | 0.5041 | 0.5001 | 0.3939 | -0.0305 |
| 1 | SVC | 0.4945 | 0.5022 | 0.5123 | -1.0082 | -1.7859 |
| 1 | XGBClassifier | 0.4995 | 0.5044 | 0.5030 | 0.1371 | -0.3512 |
| 5 | ExtraTreesClassifier | 0.5042 | 0.5198 | 0.5197 | 0.0089 | -1.5483 |
| 5 | HistGradientBoostingClassifier | 0.4917 | 0.5023 | 0.4972 | 0.5107 | -0.5582 |
| 5 | LogisticRegression | 0.4982 | 0.5379 | 0.5192 | 1.8709 | -2.1031 |
| 5 | RandomForestClassifier | 0.4909 | 0.5054 | 0.5053 | 0.0085 | -1.4377 |
| 5 | SVC | 0.4808 | 0.5039 | 0.4953 | 0.8581 | -1.4551 |
| 5 | XGBClassifier | 0.4813 | 0.5027 | 0.4983 | 0.4450 | -1.6914 |

## Top 3 per horyzont — AUC poszczególnych foldów

| horizon | model | feature_set | roc_auc_mean | roc_auc_std | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | HistGradientBoostingClassifier | SELECTED | 0.5150 | 0.0171 | 0.4981 | 0.4993 | 0.5154 | 0.5389 | 0.5233 |
| 1 | SVC | TECHNICAL | 0.5123 | 0.0080 | 0.5021 | 0.5067 | 0.5131 | 0.5180 | 0.5216 |
| 1 | LogisticRegression | TECHNICAL | 0.5087 | 0.0186 | 0.4847 | 0.5251 | 0.4934 | 0.5244 | 0.5161 |
| 5 | LogisticRegression | SELECTED | 0.5379 | 0.0188 | 0.5339 | 0.5666 | 0.5379 | 0.5373 | 0.5139 |
| 5 | ExtraTreesClassifier | SELECTED | 0.5198 | 0.0314 | 0.5200 | 0.5217 | 0.5277 | 0.5586 | 0.4712 |
| 5 | ExtraTreesClassifier | TECHNICAL | 0.5197 | 0.0337 | 0.5133 | 0.5202 | 0.5314 | 0.5636 | 0.4702 |

## Ograniczenia i weryfikacja

Brak godzin publikacji i historycznych wersji FRED; lag=1 nie jest gwarancją
point-in-time. Etykiety H5 nakładają się, a std foldów nie jest przedziałem ufności.
Wybór SELECTED oraz porównanie 36 konfiguracji na znanej historii mogą zawyżać ocenę.
Nie utworzono finalnego holdoutu i nie wykonywano selekcji ani tuningu na nim.

Sprawdzono 36 konfiguracji / 180 foldów, identyczne daty i targety pomiędzy
konfiguracjami oraz ponownie policzono wszystkie metryki z zapisanych predykcji OOF.
Do odczytu `predictions.csv` przy dokładnym odtwarzaniu AUC należy użyć
`pd.read_csv(..., float_precision="round_trip")`, aby zachować remisy score.
Testy automatyczne obejmują gap, kolejność lag/target/dropna, preprocessing w Pipeline,
różnice AUC w pp, wzory intermarket i brak korzystania z przyszłych obserwacji.
