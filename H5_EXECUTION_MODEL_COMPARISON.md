# H5: porównanie modeli dla wykonalnego wejścia

## Pytanie i metodologia

Sprawdzono, czy po sygnale generowanym po zamknięciu dnia `t` modele przewidują
kierunek zwrotu od otwarcia następnego weekday do zamknięcia piątej kolejnej
obserwacji weekday:

```text
forward_return = close(t+5) / open(t+1) - 1
target = forward_return > 0
```

To osobny eksperyment — standardowy target `close(t)` → `close(t+5)` pozostał bez
zmian. Wspólna próba ma 4786 obserwacji (2008-05-02–2026-09-18), a dodatnia klasa
stanowi 54,60%. Zastosowano 5 expanding-window foldów, po 797 dat testowych na
fold, z `gap=5`. Maksymalna data końcowa etykiet treningowych poprzedza początek
testu w każdym foldzie.

Modele bazowe uczą się na stałym zestawie TECHNICAL + A+C. Porównano ustalone
wcześniej konfiguracje LogisticRegression, ExtraTrees i SVC; osobno LogisticRegression
na pięciu opóźnionych zwrotach (AR_Logistic) i mały MLP z 16 neuronami w warstwie
ukrytej. Ensemble mają stałe, równe głosy/wagi: majority vote, średnia trzech
prawdopodobieństw po sigmoid calibration oraz średnia rang. Kalibratory używają
tylko chronologicznych OOF score'ów z treningowej części zewnętrznego foldu.
Always Up jest kontrolą accuracy, nie modelem rankingowym.

## Wyniki OOF

| Model | AUC średnie ± std | Foldy AUC > 0,5 | Accuracy | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| LogisticRegression | 0,5320 ± 0,0443 | 4/5 | 0,4934 | 0,5055 |
| ExtraTrees | 0,5305 ± 0,0372 | 4/5 | 0,4856 | 0,5105 |
| SVC | 0,5325 ± 0,0546 | 4/5 | 0,5370 | 0,5000 |
| AR_Logistic | 0,4956 ± 0,0299 | 2/5 | 0,5360 | 0,4991 |
| MLP, 16 neuronów | 0,5158 ± 0,0249 | 4/5 | 0,4906 | 0,5157 |
| Ensemble majority vote | 0,5130 ± 0,0346 | 4/5 | 0,4954 | 0,5017 |
| Ensemble soft, 1/3 każdy | 0,4778 ± 0,0560 | 2/5 | 0,5164 | 0,5018 |
| Ensemble rank, 1/3 każdy | **0,5365 ± 0,0530** | 3/5 | 0,4688 | 0,5067 |
| Always Up | — | — | **0,5370** | 0,5000 |

### Interpretacja

- AR_Logistic nie poprawia rankingu względem losowego (`AUC=0,496`).
- Trzy pojedyncze modele mają AUC około 0,53, z dużą zmiennością foldów i bez
  przewagi accuracy nad Always Up.
- Rank ensemble ma najwyższe AUC, ale różnica jest niewielka, odchylenie foldów
  duże, a accuracy niższe niż Always Up. Nie jest kandydatem do użycia bez dalszej
  walidacji.
- Soft ensemble wypada słabo (`AUC=0,478`). Kalibracja sigmoidą na wcześniejszych
  wewnętrznych okresach nie zachowała kierunku score w części późniejszych foldów
  LogisticRegression/ExtraTrees; uśrednione prawdopodobieństwo przez to nie
  uogólniło się.
- MLP osiągnął limit 2000 iteracji bez zbieżności w **5/5 foldów**. Jego AUC jest
  wyłącznie diagnostyczne i nie pozwala ocenić poprawnie wytrenowanej sieci.

Wyniki to development walk-forward na historii już używanej w poprzednich
eksperymentach. Nie są niezależnym holdoutem. Nie wykonano optymalizacji wag,
doboru progu, backtestu po spreadzie/slippage/prowizji ani paper tradingu.

## Odtworzenie

```bash
.venv/bin/python -m src.models.compare_h5_execution_models
.venv/bin/python -m unittest tests.test_h5_execution_model_comparison -v
```

Kod i jawne parametry: `src/models/compare_h5_execution_models.py`. Wyniki
maszynowe (OOF predykcje, foldy, splity, summary, metadata):
`data/processed/h5_execution_model_comparison/`.
