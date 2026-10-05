# Controlled tuning H5 — TECHNICAL + A+C

Wykonano 198 konfiguracji × 5 expanding-window foldów = **990 fitów**.
Wszystkie daty X/y pochodzą z aktualnego weekdays pipeline, H5, lag=1, gap=5.
Feature set pozostał zamrożony na 14 cechach TECHNICAL + A=`dfii10_vs_sma20` +
C=`dxy_sma20_vs_sma50`. Wspólna próba ma 4786 wierszy; każdy fold testowy zawiera
797 OOF predykcji. Nie strojono na holdoucie; holdoutu/ensemble nie uruchamiano.

LogisticRegression i SVC skalują wewnątrz Pipeline; scaler `n_samples_seen_`
równał się liczbie treningowej w każdym foldzie. Score AUC pochodzi z prawdopodobieństwa
UP dla pozostałych klasyfikatorów i `decision_function` dla SVC. Parametry poniżej
są kompletnymi, efektywnymi ustawieniami estimatorów; `search_params_json` w CSV
pokazuje parametry strojone. Baselines zachowują dokładne ustawienia benchmarku.

## Punkt odniesienia Logistic A+C

AUC średnie **0,529313**, std **0,041534**, mediana **0,524516**, minimum **0,477870**.
Foldy: 0,524050 · 0,526066 · 0,524516 · 0,594063 · 0,477870.

## Najlepsza konfiguracja według średniej AUC dla każdej rodziny

Ranking konfiguracji był wyłącznie według mean AUC. Stabilność jest raportowana obok,
bez wagi/penalty. Wiersze konfiguracji baseline znajdują się w pełnej tabeli CSV/HTML.

| model | params_json | mean_roc_auc | median_auc | std_roc_auc | min_auc | fold_1_auc | fold_2_auc | fold_3_auc | fold_4_auc | fold_5_auc | folds_above_05 | mean_balanced_accuracy | mean_accuracy | delta_mean_auc_pp_vs_model_baseline | delta_mean_auc_pp_vs_previous_ac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ExtraTreesClassifier | {"bootstrap":false,"ccp_alpha":0.0,"class_weight":null,"criterion":"gini","max_depth":8,"max_features":0.5,"max_leaf_nodes":null,"max_samples":null,"min_impurity_decrease":0.0,"min_samples_leaf":5,"min_samples_split":2,"min_weight_fraction_leaf":0.0,"monotonic_cst":null,"n_estimators":300,"n_jobs":1,"oob_score":false,"random_state":42,"verbose":0,"warm_start":false} | 0.5310 | 0.5214 | 0.0376 | 0.5053 | 0.5258 | 0.5062 | 0.5214 | 0.5963 | 0.5053 | 5 | 0.5117 | 0.4891 | 1.2618 |  |
| HistGradientBoostingClassifier | {"categorical_features":"from_dtype","class_weight":null,"early_stopping":false,"interaction_cst":null,"l2_regularization":0.1,"learning_rate":0.01,"loss":"log_loss","max_bins":255,"max_depth":null,"max_features":1.0,"max_iter":200,"max_leaf_nodes":7,"min_samples_leaf":10,"monotonic_cst":null,"n_iter_no_change":10,"random_state":42,"scoring":"loss","tol":1e-07,"validation_fraction":0.1,"verbose":0,"warm_start":false} | 0.5105 | 0.4908 | 0.0510 | 0.4591 | 0.4740 | 0.5520 | 0.4908 | 0.5764 | 0.4591 | 2 | 0.5063 | 0.4878 | 0.9281 |  |
| LogisticRegression | {"logisticregression":"LogisticRegression(C=0.003, max_iter=3000, random_state=42)","logisticregression__C":0.003,"logisticregression__class_weight":null,"logisticregression__dual":false,"logisticregression__fit_intercept":true,"logisticregression__intercept_scaling":1,"logisticregression__l1_ratio":0.0,"logisticregression__max_iter":3000,"logisticregression__n_jobs":null,"logisticregression__penalty":"deprecated","logisticregression__random_state":42,"logisticregression__solver":"lbfgs","logisticregression__tol":0.0001,"logisticregression__verbose":0,"logisticregression__warm_start":false,"memory":null,"standardscaler":"StandardScaler()","standardscaler__copy":true,"standardscaler__with_mean":true,"standardscaler__with_std":true,"steps":[["standardscaler","StandardScaler()"],["logisticregression","LogisticRegression(C=0.003, max_iter=3000, random_state=42)"]],"transform_input":null,"verbose":false} | 0.5332 | 0.5159 | 0.0429 | 0.4930 | 0.5159 | 0.5516 | 0.5063 | 0.5994 | 0.4930 | 4 | 0.5028 | 0.4974 | 0.3934 | 0.3934 |
| SVC | {"memory":null,"standardscaler":"StandardScaler()","standardscaler__copy":true,"standardscaler__with_mean":true,"standardscaler__with_std":true,"steps":[["standardscaler","StandardScaler()"],["svc","SVC(C=0.01, gamma=0.001, probability=False)"]],"svc":"SVC(C=0.01, gamma=0.001, probability=False)","svc__C":0.01,"svc__break_ties":false,"svc__cache_size":200,"svc__class_weight":null,"svc__coef0":0.0,"svc__decision_function_shape":"ovr","svc__degree":3,"svc__gamma":0.001,"svc__kernel":"rbf","svc__max_iter":-1,"svc__probability":false,"svc__random_state":null,"svc__shrinking":true,"svc__tol":0.001,"svc__verbose":false,"transform_input":null,"verbose":false} | 0.5376 | 0.5289 | 0.0506 | 0.4779 | 0.5289 | 0.5235 | 0.4779 | 0.6176 | 0.5399 | 4 | 0.5000 | 0.5400 | 2.6635 |  |

### Najważniejsze porównania z baseline modeli

- LogisticRegression: `C=0.003`, L2, `class_weight=None`. Mean AUC 0,533247,
  **+0,393 pp** vs A+C baseline; zmiana foldów −0,82 / +2,55 / −1,82 / +0,54 / +1,52 pp.
  Poprawa w 3/5 foldów, std lekko rośnie 0,041534 → 0,042897. Fold 5 AUC rośnie
  0,477870 → 0,493032, nadal pozostaje poniżej 0,5. To mały przyrost, nie stabilna poprawa.
- SVC RBF: `C=0.01`, `gamma=0.001`, `class_weight=None`. Mean AUC 0,537564,
  +2,664 pp względem SVC baseline; std 0,050616, min 0,477911. AUC >0,5 w 4/5 foldów.
  Balanced accuracy 0,500 i klasyfikacja przewiduje **UP dla 100% obserwacji** w każdym foldzie.
  Score ma wąski zakres blisko 1; AUC odzwierciedla drobne rankingi score, nie użyteczną
  binarną klasyfikację przy domyślnym progu.
- ExtraTrees: `n_estimators=300`, `max_depth=8`, `min_samples_leaf=5`,
  `max_features=0.5`, `class_weight=None`. Mean AUC 0,530991 (+1,262 pp vs baseline),
  std 0,037629 i AUC >0,5 w 5/5 foldów; minimum 0,505279. Wciąż fold 4 ma 0,596319,
  więc średnia też korzysta z mocnego okresu, ale inne foldy pozostają nad 0,5.
- HistGradientBoosting: `learning_rate=0.01`, `max_iter=200`, `max_leaf_nodes=7`,
  `l2_regularization=0.1`, `min_samples_leaf=10`. Mean AUC 0,510459 (+0,928 pp vs baseline),
  std 0,051043, minimum 0,459111; tylko 2/5 foldów >0,5. Strojenie podnosi średnią,
  lecz zwiększa niestabilność wyraźnie.

## Wszystkie baseline'y i zwycięzcy

| model | params_json | mean_roc_auc | median_auc | std_roc_auc | min_auc | fold_1_auc | fold_2_auc | fold_3_auc | fold_4_auc | fold_5_auc | folds_above_05 | mean_balanced_accuracy | mean_accuracy | delta_mean_auc_pp_vs_model_baseline | delta_mean_auc_pp_vs_previous_ac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ExtraTreesClassifier | {"bootstrap":false,"ccp_alpha":0.0,"class_weight":null,"criterion":"gini","max_depth":8,"max_features":0.5,"max_leaf_nodes":null,"max_samples":null,"min_impurity_decrease":0.0,"min_samples_leaf":5,"min_samples_split":2,"min_weight_fraction_leaf":0.0,"monotonic_cst":null,"n_estimators":300,"n_jobs":1,"oob_score":false,"random_state":42,"verbose":0,"warm_start":false} | 0.5310 | 0.5214 | 0.0376 | 0.5053 | 0.5258 | 0.5062 | 0.5214 | 0.5963 | 0.5053 | 5 | 0.5117 | 0.4891 | 1.2618 |  |
| ExtraTreesClassifier | {"bootstrap":false,"ccp_alpha":0.0,"class_weight":null,"criterion":"gini","max_depth":null,"max_features":"sqrt","max_leaf_nodes":null,"max_samples":null,"min_impurity_decrease":0.0,"min_samples_leaf":5,"min_samples_split":2,"min_weight_fraction_leaf":0.0,"monotonic_cst":null,"n_estimators":200,"n_jobs":1,"oob_score":false,"random_state":42,"verbose":0,"warm_start":false} | 0.5184 | 0.5059 | 0.0345 | 0.4884 | 0.5059 | 0.4884 | 0.5160 | 0.5775 | 0.5040 | 4 | 0.5097 | 0.4861 | 0.0000 |  |
| HistGradientBoostingClassifier | {"categorical_features":"from_dtype","class_weight":null,"early_stopping":false,"interaction_cst":null,"l2_regularization":0.1,"learning_rate":0.01,"loss":"log_loss","max_bins":255,"max_depth":null,"max_features":1.0,"max_iter":200,"max_leaf_nodes":7,"min_samples_leaf":10,"monotonic_cst":null,"n_iter_no_change":10,"random_state":42,"scoring":"loss","tol":1e-07,"validation_fraction":0.1,"verbose":0,"warm_start":false} | 0.5105 | 0.4908 | 0.0510 | 0.4591 | 0.4740 | 0.5520 | 0.4908 | 0.5764 | 0.4591 | 2 | 0.5063 | 0.4878 | 0.9281 |  |
| HistGradientBoostingClassifier | {"categorical_features":"from_dtype","class_weight":null,"early_stopping":false,"interaction_cst":null,"l2_regularization":1.0,"learning_rate":0.05,"loss":"log_loss","max_bins":255,"max_depth":null,"max_features":1.0,"max_iter":200,"max_leaf_nodes":15,"min_samples_leaf":20,"monotonic_cst":null,"n_iter_no_change":10,"random_state":42,"scoring":"loss","tol":1e-07,"validation_fraction":0.1,"verbose":0,"warm_start":false} | 0.5012 | 0.4933 | 0.0217 | 0.4747 | 0.5205 | 0.4747 | 0.4933 | 0.5264 | 0.4909 | 2 | 0.5119 | 0.4913 | 0.0000 |  |
| LogisticRegression | {"logisticregression":"LogisticRegression(C=0.003, max_iter=3000, random_state=42)","logisticregression__C":0.003,"logisticregression__class_weight":null,"logisticregression__dual":false,"logisticregression__fit_intercept":true,"logisticregression__intercept_scaling":1,"logisticregression__l1_ratio":0.0,"logisticregression__max_iter":3000,"logisticregression__n_jobs":null,"logisticregression__penalty":"deprecated","logisticregression__random_state":42,"logisticregression__solver":"lbfgs","logisticregression__tol":0.0001,"logisticregression__verbose":0,"logisticregression__warm_start":false,"memory":null,"standardscaler":"StandardScaler()","standardscaler__copy":true,"standardscaler__with_mean":true,"standardscaler__with_std":true,"steps":[["standardscaler","StandardScaler()"],["logisticregression","LogisticRegression(C=0.003, max_iter=3000, random_state=42)"]],"transform_input":null,"verbose":false} | 0.5332 | 0.5159 | 0.0429 | 0.4930 | 0.5159 | 0.5516 | 0.5063 | 0.5994 | 0.4930 | 4 | 0.5028 | 0.4974 | 0.3934 | 0.3934 |
| LogisticRegression | {"logisticregression":"LogisticRegression(max_iter=3000, random_state=42)","logisticregression__C":1.0,"logisticregression__class_weight":null,"logisticregression__dual":false,"logisticregression__fit_intercept":true,"logisticregression__intercept_scaling":1,"logisticregression__l1_ratio":0.0,"logisticregression__max_iter":3000,"logisticregression__n_jobs":null,"logisticregression__penalty":"deprecated","logisticregression__random_state":42,"logisticregression__solver":"lbfgs","logisticregression__tol":0.0001,"logisticregression__verbose":0,"logisticregression__warm_start":false,"memory":null,"standardscaler":"StandardScaler()","standardscaler__copy":true,"standardscaler__with_mean":true,"standardscaler__with_std":true,"steps":[["standardscaler","StandardScaler()"],["logisticregression","LogisticRegression(max_iter=3000, random_state=42)"]],"transform_input":null,"verbose":false} | 0.5293 | 0.5245 | 0.0415 | 0.4779 | 0.5241 | 0.5261 | 0.5245 | 0.5941 | 0.4779 | 4 | 0.5110 | 0.4996 | 0.0000 | -0.0000 |
| SVC | {"memory":null,"standardscaler":"StandardScaler()","standardscaler__copy":true,"standardscaler__with_mean":true,"standardscaler__with_std":true,"steps":[["standardscaler","StandardScaler()"],["svc","SVC(C=0.01, gamma=0.001, probability=False)"]],"svc":"SVC(C=0.01, gamma=0.001, probability=False)","svc__C":0.01,"svc__break_ties":false,"svc__cache_size":200,"svc__class_weight":null,"svc__coef0":0.0,"svc__decision_function_shape":"ovr","svc__degree":3,"svc__gamma":0.001,"svc__kernel":"rbf","svc__max_iter":-1,"svc__probability":false,"svc__random_state":null,"svc__shrinking":true,"svc__tol":0.001,"svc__verbose":false,"transform_input":null,"verbose":false} | 0.5376 | 0.5289 | 0.0506 | 0.4779 | 0.5289 | 0.5235 | 0.4779 | 0.6176 | 0.5399 | 4 | 0.5000 | 0.5400 | 2.6635 |  |
| SVC | {"memory":null,"standardscaler":"StandardScaler()","standardscaler__copy":true,"standardscaler__with_mean":true,"standardscaler__with_std":true,"steps":[["standardscaler","StandardScaler()"],["svc","SVC(probability=False)"]],"svc":"SVC(probability=False)","svc__C":1.0,"svc__break_ties":false,"svc__cache_size":200,"svc__class_weight":null,"svc__coef0":0.0,"svc__decision_function_shape":"ovr","svc__degree":3,"svc__gamma":"scale","svc__kernel":"rbf","svc__max_iter":-1,"svc__probability":false,"svc__random_state":null,"svc__shrinking":true,"svc__tol":0.001,"svc__verbose":false,"transform_input":null,"verbose":false} | 0.5109 | 0.5076 | 0.0563 | 0.4501 | 0.5076 | 0.4648 | 0.5469 | 0.5852 | 0.4501 | 3 | 0.5164 | 0.4903 | 0.0000 |  |

## Korelacja score i zgodność predykcji najlepszych konfiguracji

Korelacje liczone na wspólnych 3985 OOF datach. Są opisowe: najlepsze konfiguracje
wybrano po tych samych foldach.

| model | ExtraTreesClassifier | HistGradientBoostingClassifier | LogisticRegression | SVC |
| --- | --- | --- | --- | --- |
| ExtraTreesClassifier | 1.0000 | 0.8590 | 0.6804 | 0.5466 |
| HistGradientBoostingClassifier | 0.8590 | 1.0000 | 0.5835 | 0.4428 |
| LogisticRegression | 0.6804 | 0.5835 | 1.0000 | 0.7893 |
| SVC | 0.5466 | 0.4428 | 0.7893 | 1.0000 |

Zgodność klas przy domyślnym progu:

| model | ExtraTreesClassifier | HistGradientBoostingClassifier | LogisticRegression | SVC |
| --- | --- | --- | --- | --- |
| ExtraTreesClassifier | 1.0000 | 0.8417 | 0.7036 | 0.5159 |
| HistGradientBoostingClassifier | 0.8417 | 1.0000 | 0.6903 | 0.4971 |
| LogisticRegression | 0.7036 | 0.6903 | 1.0000 | 0.7365 |
| SVC | 0.5159 | 0.4971 | 0.7365 | 1.0000 |

SVC ma niższą korelację score z ExtraTrees/HGB (0,547 / 0,443), ale ten konkretny
SVC zawsze przewiduje UP; nie wnosi użytecznej różnorodności klas przy domyślnym progu.
Wyniki pokazują pewną odmienność rankingów i błędów, więc osobny eksperyment ensemble
może być zasadny, ale dopiero po zaprojektowaniu osobnej walidacji. Nie zbudowano ensemble.

## Interpretacja i kontrola

Wszystkie cztery zwycięskie konfiguracje mają mean AUC >0,5; tylko ExtraTrees ma
AUC >0,5 w 5/5 foldów. SVC ma najwyższą średnią, ale największe praktyczne zastrzeżenie
stanowi stała predykcja UP; Logistic poprawił A+C tylko o 0,39 pp i nie w większości
zdecydowanej (3/5) foldów. Żadna z tych wartości nie potwierdza przewagi tradingowej.

Nie widać technicznego leakage: identyczne zapisane daty foldów/gap, H5 `label_end`
przed walidacją, weekday-only, stałe A+C, target/diagnostyka poza X, scaler fit na train,
OOF daty poza train i bez duplikatów. SHA256 wejściowych RAW zgadza się z benchmarkiem.
Wyniki top konfiguracji są optymistyczne selekcyjnie, bo wybrano je i oceniono na tych
samych 5 foldach. Fold 5 nie był celem strojenia i pozostaje częścią CV; jego zmiany są
jedynie diagnostyczne. Trzeba oceniać później na nowych forward danych.

## Następny etap

Nie zmieniono modelu produkcyjnego, cech ani threshold. A+C Logistic pozostaje dotychczasowym
kandydatem; tuning nie daje przekonującej poprawy Logistic. OOF score są zapisane do
przyszłego porównania błędów, ale nie wolno traktować ich korelacji jako niezależnej
podstawy wyboru ensemble. Nie wykonano holdoutu ani ensemble.

Pliki: `all_configs.csv` (198 konfiguracji), `top_configs.csv`, `fold_results.csv`
(990 wierszy), `oof_predictions.csv` (baseline + zwycięzca każdego modelu), korelacje,
zgodność, granice foldów, metadane i `report.html` w `data/processed/tuning_h5_ac/`.
Przeszło 13 testów; OOF unikalność/datowanie i fit scalerów sprawdzono asercjami podczas fitów.
