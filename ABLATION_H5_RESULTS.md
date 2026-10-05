# Ablation H5 LogisticRegression

Uruchomienie: `.venv/bin/python -m src.models.ablation`.
Kod korzysta z `prepare_dataset`, `make_models`, `score_predictions`, `summarize`
i istniejącego `create_time_series_splits`. Pipeline danych i intermarket bez zmian.

A = `dfii10_vs_sma20`; B = `dxy_vs_sma50`; C = `dxy_sma20_vs_sma50`.
4945 wspólnych obserwacji benchmarku, 5 foldów, gap=5, intermarket lag=1.
StandardScaler pozostaje wewnątrz Pipeline. Bez tuningu, progu, holdoutu i ensemble.
Zweryfikowano SHA256 surowych źródeł, daty i liczebności foldów oraz dokładne
odtworzenie targetów, klas i score benchmarku dla TECHNICAL i A+B+C.

## Wyniki

| features | auc_mean | auc_std | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | delta_vs_technical_pp | positive_fold_delta_count | balanced_accuracy_mean | accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TECHNICAL | 0.5192 | 0.0205 | 0.5162 | 0.5414 | 0.4883 | 0.5342 | 0.5160 | 0.0000 | 0 | 0.5014 | 0.4915 |
| TECHNICAL + A | 0.5303 | 0.0223 | 0.5207 | 0.5462 | 0.5071 | 0.5606 | 0.5169 | 1.1093 | 5 | 0.5075 | 0.4988 |
| TECHNICAL + B | 0.5234 | 0.0248 | 0.5169 | 0.5437 | 0.4893 | 0.5513 | 0.5157 | 0.4166 | 4 | 0.5014 | 0.4913 |
| TECHNICAL + C | 0.5285 | 0.0176 | 0.5238 | 0.5541 | 0.5100 | 0.5380 | 0.5168 | 0.9322 | 5 | 0.5056 | 0.4947 |
| TECHNICAL + A+B | 0.5329 | 0.0264 | 0.5206 | 0.5477 | 0.5084 | 0.5719 | 0.5156 | 1.3641 | 4 | 0.5121 | 0.5027 |
| TECHNICAL + A+C | 0.5375 | 0.0207 | 0.5280 | 0.5590 | 0.5269 | 0.5596 | 0.5137 | 1.8239 | 4 | 0.5125 | 0.5027 |
| TECHNICAL + B+C | 0.5272 | 0.0187 | 0.5275 | 0.5594 | 0.5179 | 0.5140 | 0.5169 | 0.7945 | 4 | 0.4991 | 0.4886 |
| TECHNICAL + A+B+C | 0.5379 | 0.0188 | 0.5339 | 0.5666 | 0.5379 | 0.5373 | 0.5139 | 1.8709 | 4 | 0.5033 | 0.4934 |

## Interpretacja i zestaw do dalszej walidacji

1. Nie ma jednej cechy wyjaśniającej całość poprawy. A sama daje +1,109 pp
   (około 59% poprawy pełnego zestawu), C +0,932 pp (około 50%). Efekty nie są addytywne.
2. A daje wyraźniejszą poprawę eksploracyjną: AUC 0,5303 i dodatnie delty w 5/5 foldów.
   Nie jest to potwierdzenie istotności statystycznej ani praktycznej przewagi tradingowej.
3. DXY działa również samodzielnie: B +0,417 pp (4/5), C +0,932 pp (5/5).
   C jest lepszym i stabilniejszym pojedynczym kandydatem DXY niż B.
4. DXY nie pomaga dopiero z real yield. Połączenie A+C jednak podnosi AUC ponad
   oba pojedyncze warianty: 0,53745, czyli +0,715 pp względem A.
5. Pełny zestaw przewyższa najlepszą pojedynczą cechę A o 0,762 pp AUC.
6. Usunięcie żadnej cechy nie poprawia średniego AUC pełnego zestawu.
   Usunięcie B kosztuje tylko 0,047 pp; usunięcie C 0,507 pp; usunięcie A 1,076 pp.
7. A i C osobno poprawiają każdy fold, ale nierówno. A+C i A+B+C poprawiają
   4/5 foldów względem TECHNICAL. Ostatni fold lekko się pogarsza.
   Poprawa A+C pochodzi głównie z foldów 3 i 4, nie z jednego okresu.
8. **Rekomendowany zestaw do zamrożenia: TECHNICAL + A+C.** Zachowuje 97,5%
   przyrostu AUC pełnego zestawu, używając dwóch cech. To kompromis prostoty,
   a nie zwycięstwo stabilnością: std 0,0207 jest nieco gorsze od 0,0188 dla A+B+C.
   Najprostszy wariant z jedną cechą to A, ale zachowuje tylko 59% poprawy;
   C jest stabilniejsze (std 0,0176) i zachowuje około połowy poprawy.

A+C ma balanced accuracy 51,25% i accuracy 50,27% wobec 50,33% / 49,34%
dla A+B+C. To metryki pomocnicze przy niezmienionym progu; nie optymalizowano go.
Nie zmieniono konfiguracji benchmarku ani list SELECTED; rekomendacja dotyczy
wyłącznie następnej, oddzielnej walidacji. Żadna cecha nie została usunięta z datasetu.

Te same dane były już używane do eksploracji i wyboru cech. Wynik nie jest
niezależnym potwierdzeniem przewagi. H5 ma nakładające się etykiety, a lag=1 nie
rozwiązuje niepewności rzeczywistych godzin publikacji FRED. Kolejny krok wymaga
całkowicie nietkniętego chronologicznego okresu; końcówki tej już ocenionej historii
nie można nazwać nietkniętym holdoutem. W tym eksperymencie holdoutu nie tworzono.

Wyniki maszynowe: `data/processed/ablation_h5/` — summary, folds, predictions,
pełne indeksy train/test w splits.json, metadata i report.html. Przeszło 7 testów;
ponownie sprawdzono AUC z predykcji OOF, wspólne daty i wszystkie delty foldów.
