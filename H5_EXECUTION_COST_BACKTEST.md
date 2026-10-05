# Kosztowy backtest OOF dla targetu H5 z wykonalnym wejściem

## Zakres i reguły

Backtest używa zapisanych out-of-fold predykcji dla targetu: sygnał po `close(t)`,
wejście na `open(t+1)`, wyjście na `close(t+5)`. Pobrano ceny wejścia/wyjścia z
surowego weekday OHLC. Kolejne pozycje w każdej strategii nie nakładają się;
sygnały podczas otwartej pozycji są pomijane. Po wyjściu po cenie close nowy
sygnał z tego samego dnia może wejść następnego dnia.

Porównano stałe, już zapisane predykcje Logistic Regression, Extra Trees, SVC,
ensemble rank 3 modeli, majority vote 5 modeli i ARX Ridge z baseline'em
Always Long. Dla modeli policzono long-only (UP = long, DOWN = gotówka) i
long/short (UP = long, DOWN = short). Nie strojono progu ani reguły na wynikach
backtestu. Zwroty składają się po transakcjach z pełnym wykorzystaniem kapitału,
bez dźwigni.

Ponieważ nie wskazano brokera ani jego tabeli opłat, użyto analizy wrażliwości
kosztu round-trip **0 / 0,25 / 0,50 / 1,00 USD na uncję**. To scenariusze
hipotetyczne, a nie potwierdzone spready, poślizg i prowizje brokera. Koszt
odejmuje się od P&L każdej transakcji. Drawdown mierzony jest na zamknięciach
transakcji. Dzienna wycena otwartych pozycji nie została zaimplementowana,
choć dostępne są dzienne ceny zamknięcia; dokładnej ścieżki intraday nie znamy.
Miara ta może zaniżać obsunięcie względem wyceny pozycji w czasie jej trwania.

## Long-only: wyniki przy 0,50 USD/oz round-trip

| Reguła | Transakcje | Zwrot brutto składany | Zwrot po koszcie | Max drawdown na zamknięciach | Win rate | Profit factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always Long | 797 | 156,0% | **99,9%** | -50,1% | 53,1% | 1,142 |
| Logistic Regression | 594 | 32,2% | 7,7% | -48,2% | 52,5% | 1,045 |
| Extra Trees | 433 | 41,7% | 21,4% | -44,8% | 52,2% | 1,087 |
| SVC | 797 | 156,0% | 99,9% | -50,1% | 53,1% | 1,142 |
| Ensemble rank, 3 modele | 284 | -17,0% | -25,1% | -42,0% | 48,9% | 0,920 |
| Ensemble majority, 5 modeli | 488 | 39,9% | 18,1% | -44,9% | 53,1% | 1,075 |
| ARX Ridge | 432 | 11,8% | -4,1% | -42,9% | 52,8% | 1,019 |

SVC i Always Long są identyczne, ponieważ hard predictions SVC wskazują UP dla
każdego dnia. Żaden model nie pokonał Always Long w tym wariancie. Zwroty są
skumulowane przez około 15 lat bez dźwigni; dodatni wynik łączny nie oznacza
stabilnej przewagi. Przykładowo Logistic Regression traciła w pierwszych dwóch
foldach, a późny fold Always Long korzystał z mocnego wzrostu złota.

## Wrażliwość na koszt

| Reguła long-only | 0 USD/oz | 0,25 USD/oz | 0,50 USD/oz | 1,00 USD/oz |
| --- | ---: | ---: | ---: | ---: |
| Always Long | 156,0% | 126,2% | **99,9%** | 56,1% |
| Logistic Regression | 32,2% | 19,3% | 7,7% | -12,2% |
| Extra Trees | 41,7% | 31,2% | 21,4% | 4,1% |
| Ensemble majority, 5 modeli | 39,9% | 28,6% | 18,1% | -0,3% |
| ARX Ridge | 11,8% | 3,5% | -4,1% | -17,8% |

## Long/short przy koszcie 0,50 USD/oz

| Reguła | Zwrot po koszcie | Max drawdown na zamknięciach | Profit factor |
| --- | ---: | ---: | ---: |
| Logistic Regression | -64,7% | -77,5% | 0,886 |
| Extra Trees | -70,8% | -80,1% | 0,862 |
| Ensemble rank, 3 modele | -83,7% | -89,9% | 0,791 |
| Ensemble majority, 5 modeli | -78,8% | -82,9% | 0,823 |
| ARX Ridge | -66,2% | -77,5% | 0,881 |

To wyklucza obecną, naiwną regułę „DOWN = short” jako sensowny wariant dla
tych predykcji. Long-only również nie dorównuje prostemu benchmarkowi.

## Wniosek i ograniczenia

W tych wynikach nie ma podstaw, by wdrażać ensemble ani którykolwiek model jako
strategię. Najsilniejszym benchmarkiem jest ekspozycja Always Long, co wskazuje,
że wzrost złota w badanym okresie dominuje nad informacją z sygnałów. Wcześniej
wybierano cechy i modele na tej historii, więc jest to nadal diagnostyka
rozwojowa, a nie niezależny test. Nie należy traktować wartości 0,50 USD/oz jako
kosztu konkretnego brokera.

Następny krok to zamrożenie strategii badawczej albo uznanie, że nie ma jeszcze
przewagi, a po wskazaniu brokera zastąpienie scenariuszy jego faktycznymi
bid/ask, prowizją i poślizgiem. Dopiero po zamrożeniu zasad należy zbierać nowe
forward dane na nietknięty test. Warto też pamiętać, że obecny kalendarz
poniedziałek–piątek jest przybliżeniem sesji bez świąt giełdowych.

## Odtworzenie i artefakty

```bash
.venv/bin/python -m src.models.backtest_h5_execution_oof
.venv/bin/python -m unittest tests.test_backtest_h5_execution_oof -v
```

Kod: `src/models/backtest_h5_execution_oof.py`. Wyniki maszynowe:
`data/processed/h5_execution_cost_backtest/` — `summary.csv`, `fold_summary.csv`,
`trades.csv`, `oof_prediction_table.csv` i `metadata.json`.
