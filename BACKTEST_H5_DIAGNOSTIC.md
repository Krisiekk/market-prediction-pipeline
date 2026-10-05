# Wstępna diagnostyka backtestu H5

To demonstracyjne wyliczenie wykonano z zapisanych predykcji OOF i dziennych
surowych danych XAU/USD. Nie jest to niezależny test ani zatwierdzony wynik
strategii. Model i cechy wybierano, analizując tę samą historię.

## Założenia symulacji

- Predykcje: LogisticRegression, `TECHNICAL + A+C`, z
  `data/processed/ablation_h5_weekdays/predictions.csv`.
- Sygnał generowany po zamknięciu dnia `t`.
- Reguła long-only: `prediction = 1` otwiera pozycję długą; `prediction = 0`
  oznacza brak pozycji. Dodatkowo policzono wariant long/short oraz benchmark
  zawsze-long.
- Wejście po `open` kolejnej dostępnej obserwacji poniedziałek–piątek.
- Wyjście po `close` piątej kolejnej obserwacji poniedziałek–piątek liczonej od
  dnia sygnału.
- Pozycje nie nakładają się. Sygnały w czasie otwartej pozycji są pomijane.
- Jedna pozycja wykorzystuje cały kapitał, bez dźwigni; zwroty są składane po
  transakcjach. Obsunięcie liczone jest tylko na zamknięciach transakcji.
- Koszt demonstracyjny: 0,50 USD/oz za całą transakcję round-trip. To parametr
  przykładowy, nie kwotowanie brokera; nie rozdzielono spreadu, poślizgu i
  prowizji.

## Wzory

`side` wynosi `+1` dla long, `-1` dla short. Dla pozycji long-only sygnał DOWN
pomija transakcję.

```text
gross_usd_per_oz = side * (exit_price - entry_price)
gross_return = gross_usd_per_oz / entry_price
net_return = gross_return - round_trip_cost_usd_per_oz / entry_price
equity_next = equity_now * (1 + net_return)
```

Jeśli koszty będą podane osobno, przy cenach OHLC traktowanych jako mid-price
można przyjąć:

```text
round_trip_cost = full_spread + 2 * slippage_per_side + round_trip_commission
```

Przy rzeczywistych kwotowaniach bid/ask lepiej policzyć wejście i wyjście po
odpowiednich stronach rynku zamiast odejmować pełny spread jako osobny koszt.

## Przykład transakcji

Pierwszy sygnał UP dla `TECHNICAL + A+C`:

| Sygnał | Wejście | Cena wejścia | Wyjście | Cena wyjścia |
|---|---|---:|---|---:|
| 2011-05-30 | 2011-05-31 open | 1539,94995 | 2011-06-06 close | 1544,00 |

Zysk brutto to `1544,00 - 1539,94995 = 4,05005 USD/oz`, czyli około 0,2630%.
Po demonstracyjnym koszcie 0,50 USD/oz zostaje 3,55005 USD/oz, czyli około
0,2305%.

## Wyniki demonstracyjne

Predykcje obejmują 3985 dat OOF w pięciu foldach; symulacja long-only wykonała
541 transakcji. Zwrot składany zakłada reinwestowanie całego kapitału po każdej
transakcji i brak dźwigni.

| Reguła | Transakcje | Zwrot brutto składany | Zwrot po koszcie 0,50 USD/oz | Maks. obsunięcie po koszcie* |
|---|---:|---:|---:|---:|
| LogisticRegression TECHNICAL + A+C: long przy UP, gotówka przy DOWN | 541 | 58,7% | 32,1% | -43,7% |
| LogisticRegression TECHNICAL: long przy UP, gotówka przy DOWN | 577 | 11,0% | -8,7% | -47,0% |
| Zawsze long, bez sygnału modelu | 797 | 156,0% | 99,9% | -50,1% |
| LogisticRegression TECHNICAL + A+C: long przy UP, short przy DOWN | 797 | -49,0% | -60,2% | -77,5% |

\* Obsunięcie mierzone na zamknięciach transakcji, bez wyceny pozycji w trakcie
jej trwania. Benchmark zawsze-long ma więcej transakcji niż wariant long-only,
bo model pomija sygnały DOWN.

## Ograniczenia i następny krok

Model uczył się targetu `close(t)` do `close(t+5)`, a symulacja wchodzi po
`open(t+1)` i wychodzi po `close(t+5)`. To nie jest ten sam zwrot. Zatem wyniki
są wyłącznie diagnostycznym przykładem mechaniki kosztów i egzekucji. Ponadto
predykcje pochodzą z historii użytej wcześniej do wyboru cech i strojenia.

Przed interpretacją jako backtest strategii należy zdefiniować wykonalny target
zgodny z cenami wejścia i wyjścia, ponowić walk-forward, a następnie zastosować
rzeczywiste założenia kosztowe wybranego brokera. Do końcowego potwierdzenia
potrzebne będą nowe, wcześniej nieoceniane dane.
