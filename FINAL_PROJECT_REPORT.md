# Predykcja kierunku XAU/USD — raport końcowy projektu badawczego

**Stan: 5 października 2026.** Raport podsumowuje zapisane eksperymenty na
dostępnej historii. Nie oznacza wdrożenia agenta ani przeprowadzenia nowego holdoutu.

## 1. Cel i wynik projektu

Celem było zbudowanie pipeline’u do badania, czy cechy techniczne złota i dane
międzyrynkowe pozwalają przewidywać kierunek XAU/USD w horyzoncie H5 oraz czy
takie prognozy są użyteczne po uwzględnieniu sposobu zawierania transakcji i kosztów.

Powstał proces obejmujący pobieranie danych, audyt kalendarza, budowę cech,
walidację chronologiczną, porównanie modeli, strojenie wybranych modeli,
zespoły modeli oraz symulację transakcji. Zbadano również prostą sieć neuronową,
regresję z opóźnieniami zwrotów i filtry reżimów rynkowych.

**Główny wynik:** niektóre modele osiągają średnie AUC około 0,53–0,54,
ale badanie nie potwierdziło stabilnej przewagi inwestycyjnej. Rozbudowanie
ensemble nie przyniosło jednoznacznej poprawy. Pomijanie okresów niskiej
zmienności poprawiło wyniki badanych strategii ML, lecz podobny filtr pomaga
także prostej ekspozycji długiej. Wynik zależy od założonego kosztu transakcji.

Wartością projektu jest sprawdzenie całego łańcucha od jakości danych do
wyniku strategii oraz udokumentowanie, dlaczego nie wystarczy wybrać modelu
z najwyższym AUC. Negatywny wynik części hipotez jest wynikiem badawczym.

## 2. Dane i kalendarz

| Element | Stan wykorzystany w badaniu |
| --- | --- |
| Instrument i częstotliwość | XAU/USD, dzienne OHLC |
| Surowa historia | 5 000 rekordów, 2008-02-22–2026-09-27 |
| Rekordy weekendowe | 159: 101 sobót i 58 niedziel |
| Po filtrze poniedziałek–piątek | 4 841 rekordów, do 2026-09-25 |
| Kompletna próba H5 do porównania modeli | 4 786 dat sygnału, 2008-05-02–2026-09-18 |
| Predykcje poza treningiem (OOF) | 3 985 dat na model, 2011-05-30–2026-09-18 |
| Ostatnie zamknięcie transakcji H5 | 2026-09-25 |
| Dane międzyrynkowe | rekonstruowany DXY, nominalna i realna rentowność US 10Y, szeroki indeks USD |

Rekordy weekendowe są odfiltrowywane **przed** liczeniem wskaźników, opóźnień
i targetu. Surowe pliki zachowano. To przyjęta konwencja badania, a nie pełne
rozwiązanie kwestii sesji: nie odtworzono kalendarza świąt, granic świec ani
warunków wykonania zleceń konkretnego brokera.

**H5 oznacza piątą następną dostępną obserwację w kalendarzu poniedziałek–piątek.**
Nie oznacza zawsze tygodnia od poniedziałku do piątku ani pięciu dni kalendarzowych.
Dane pozostają dzienne; nie agregowano ich do jednej świecy tygodniowej.

Dane międzyrynkowe łączone są przez `merge_asof(direction="backward")`, a cechy
międzyrynkowe opóźniane o jedną obserwację. Chronologia dat jest kontrolowana,
ale brak godzin publikacji i historycznych wersji danych FRED uniemożliwia
stwierdzenie, że odtworzono pełny stan informacji dostępny w danym momencie.

Źródła szczegółów: [audyt kalendarza](CALENDAR_AUDIT.md),
[wyniki po filtrze](WEEKDAY_RESULTS.md), [łączenie danych](src/features/intermarket.py).

## 3. Cechy i definicja prognozy

Podstawowy zestaw obejmuje 14 cech: `SMA_20`, `RSI_14`, `EMA_20`, `MACD`,
`MACD_Signal`, `MACD_HIST`, `BB_UPPER`, `BB_LOWER`, `PRICE_TO_SMA20`,
`PRICE_TO_EMA20`, `BB_WIDTH`, `BB_POSITION`, `RETURN_1D`, `RETURN_5D`.

W badaniu wykonania transakcji dodano zestaw A+C:

- `dfii10_vs_sma20`: realna rentowność US 10Y minus jej średnia z 20 obserwacji,
  w punktach procentowych;
- `dxy_sma20_vs_sma50`: stosunek średnich DXY z 20 i 50 obserwacji minus jeden.

Okna międzyrynkowe liczone są na siatce obserwacji XAU po połączeniu danych.
AR Logistic wykorzystuje wyłącznie opóźnienia dziennych zwrotów 0, 1, 2, 4 i 9;
ARX Ridge dodaje je do zestawu technicznego i A+C.

W projekcie występują dwa różne targety, których wyników nie należy mieszać:

| Etap | Zwrot definiujący klasę UP | Interpretacja |
| --- | --- | --- |
| Wcześniejszy benchmark i ablation | `close(t+5) / close(t) - 1` | Kierunek zmiany między zamknięciami |
| Końcowe badanie wykonania | `close(t+5) / open(t+1) - 1` | Sygnał po close(t), wejście na kolejnym open |

Klasa UP oznacza zwrot większy od zera. ARX przewiduje zwrot ciągły,
a jego znak służy do wyznaczenia kierunku. Koszt nie jest częścią targetu;
odejmowany jest podczas symulacji transakcji.

## 4. Walidacja i zakres wiarygodności

Zastosowano pięć podziałów z rozszerzającym się oknem treningowym i `gap=5`.
Każdy zewnętrzny fold ma 797 obserwacji walidacyjnych. Skalowanie jest
dopasowywane w pipeline wyłącznie na treningu. Sprawdzane jest również,
czy koniec etykiety treningowej przypada przed początkiem walidacji.

| Fold | Liczba obserwacji treningowych | Daty sygnałów walidacyjnych |
| --- | ---: | --- |
| 1 | 796 | 2011-05-30–2014-06-18 |
| 2 | 1 593 | 2014-06-19–2017-07-10 |
| 3 | 2 390 | 2017-07-11–2020-08-03 |
| 4 | 3 187 | 2020-08-04–2023-08-24 |
| 5 | 3 984 | 2023-08-25–2026-09-18 |

Random Forest, HistGradientBoosting i ARX strojono w trzech dodatkowych,
chronologicznych podziałach wewnątrz każdego treningowego foldu. Parametry
wybierano bez wykorzystania jego zewnętrznej walidacji. Kalibracja
prawdopodobieństw dla soft voting także korzystała z wewnętrznych predykcji OOF.

**To nadal analiza rozwojowa, a nie niezależny holdout.** Dobór cech, rodzin
modeli i kolejnych hipotez następował po wcześniejszym oglądaniu tej historii.
Zagnieżdżone strojenie ogranicza wyciek przy doborze parametrów konkretnego
eksperymentu, lecz nie usuwa tego szerszego efektu selekcji.

Etykiety H5 w walidacji nakładają się. Odchylenie standardowe pięciu AUC nie
jest przedziałem ufności ani testem istotności. Nie wykazano statystycznie,
że niewielkie różnice między modelami utrzymają się na nowych danych.

## 5. Wyniki modeli i ensemble

Wcześniejszy benchmark porównał 6 rodzin modeli, 3 zestawy cech i 2 horyzonty
(36 konfiguracji, 180 dopasowań). Dla targetu close-to-close H5 regresja
logistyczna miała AUC 0,5086 na cechach technicznych, 0,5314 na zestawie
technicznym z A+B+C i 0,4882 na pełnym zestawie. Kontrolowane ablation dało
0,5293 dla A+C. Więcej cech nie oznaczało lepszego wyniku.

Poniższa tabela dotyczy już wyłącznie **targetu open(t+1) → close(t+5)**.
AUC i accuracy są średnimi z pięciu zewnętrznych foldów. AUC pojedynczych
klasyfikatorów dotyczy surowego score, a nie ich później kalibrowanych
prawdopodobieństw; dla majority voting score stanowi udział głosów UP.

| Model / zespół | Średnie AUC | Std AUC | Accuracy |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0,5320 | 0,0443 | 49,34% |
| Extra Trees | 0,5305 | 0,0372 | 48,56% |
| SVC | 0,5325 | 0,0546 | 53,70% |
| AR Logistic | 0,4956 | 0,0299 | 53,60% |
| MLP — wynik diagnostyczny | 0,5158 | 0,0249 | 49,06% |
| Random Forest po strojeniu | 0,5117 | 0,0323 | 49,49% |
| HistGradientBoosting po strojeniu | 0,5046 | 0,0184 | 48,41% |
| Majority voting, 3 modele | 0,5130 | 0,0346 | 49,54% |
| Soft voting, 3 modele | 0,4778 | 0,0560 | 51,64% |
| Średnia rang, 3 modele | 0,5365 | 0,0530 | 46,88% |
| Majority voting, 5 modeli | 0,5166 | 0,0299 | 48,53% |
| ARX Ridge | 0,5366 | 0,0293 | 49,31% |
| Always UP | — | — | 53,70% |

Ensemble trzech modeli składa się z **LR + Extra Trees + SVC**.
Wariant pięciu modeli dodaje **Random Forest + HistGradientBoosting**.
Przetestowano większość głosów, średnią kalibrowanych prawdopodobieństw
i średnią rang score. Te reguły nie są równoważne.

Najważniejsze obserwacje:

- SVC przy zastosowanej regule decyzyjnej głosował zawsze UP. Jego accuracy
  odpowiada baseline’owi, choć ranking score daje AUC powyżej 0,5.
- Soft voting pogorszył wynik. Kalibracja LR i Extra Trees osłabiła ranking
  w części foldów; uśrednianie prawdopodobieństw nie gwarantuje poprawy.
- Strojenie RF objęło 16 konfiguracji, HGB 32; łącznie 720 dopasowań
  wewnętrznych i 10 końcowych. Większy koszt obliczeń nie dał lepszego ensemble.
- MLP z jedną warstwą 16 neuronów nie osiągnął zbieżności w żadnym z pięciu
  foldów przy limicie 2 000 iteracji. To nie rozstrzyga o wartości sieci
  neuronowych jako całej klasy modeli.
- ARX to bezpośrednia regresja Ridge z opóźnieniami i cechami zewnętrznymi,
  nie ARIMA/SARIMAX. Miał AUC powyżej 0,5 we wszystkich foldach, ale średnie
  R² = −0,1174, MAE = 0,01779 i RMSE = 0,02373 dla zwrotów. Ujemne R² odnosi
  się do średniej rzeczywistych zwrotów danego testu w definicji tej metryki,
  a nie do osobno sprawdzonej prognozy średniej z treningu.
- Dodanie ARX jako szóstego równoprawnego głosu stworzyło remisy na 12,45%
  dat. Na pozostałych datach decyzje były identyczne jak przy pięciu głosach.
  Accuracy obu wariantów na tym samym podzbiorze wynosiło 49,44%, wobec
  54,54% dla Always UP. Nie uzyskano poprawy kierunku przez szóstego głosującego.

## 6. Symulacja transakcji i koszty

Symulacja wykorzystuje zapisane predykcje OOF. Wejście następuje na open(t+1),
wyjście na close(t+5). W danym momencie otwarta jest najwyżej jedna pozycja;
sygnały kolidujące z trwającą transakcją są pomijane. W long-only prognoza DOWN
oznacza gotówkę, a w long-short pozycję krótką. Kapitał jest reinwestowany,
bez dźwigni, z pełną ekspozycją na pojedynczą transakcję.

Analiza kosztów obejmuje 0 / 0,25 / 0,50 / 1,00 USD na uncję za pełną transakcję.
Są to scenariusze hipotetyczne. Nie odtwarzają kosztów finansowania, rzeczywistych
spreadów, poślizgu czy warunków brokera.

**Long-only, koszt 0,50 USD/oz round-trip:**

| Strategia | Transakcje | Składany zwrot netto | Max drawdown na zamknięciach transakcji |
| --- | ---: | ---: | ---: |
| Always Long | 797 | +99,90% | −50,14% |
| Logistic Regression | 594 | +7,75% | −48,17% |
| Extra Trees | 433 | +21,44% | −44,80% |
| SVC | 797 | +99,90% | −50,14% |
| Średnia rang, 3 modele | 284 | −25,10% | −41,97% |
| Majority voting, 5 modeli | 488 | +18,15% | −44,88% |
| ARX Ridge | 432 | −4,13% | −42,88% |

Zwroty dotyczą całego okresu OOF, około 2011–2026, **nie jednego roku**.
Always Long to powtarzane transakcje H5, nie klasyczne buy-and-hold. SVC
odtwarza tę samą regułę. Pozostałe strategie mają różny czas ekspozycji.

Long-short przy tym koszcie przyniósł straty około −65% do −84% dla LR,
Extra Trees, zespołów rangowego i pięciu głosów oraz ARX. Niewielka przewaga
rankingowa AUC nie przełożyła się na skuteczne decyzje kupna i krótkiej sprzedaży.

Drawdown jest liczony tylko na zamknięciach transakcji. Dzienna wycena otwartych
pozycji nie została zaimplementowana, choć są dostępne dzienne close. Ta miara
może zaniżać rzeczywiste obsunięcie. Przybliżony Sharpe zapisany w CSV opiera
się na zwrotach transakcji i nie uwzględnia dokładnie przerw w ekspozycji;
dlatego nie jest główną miarą porównawczą tego raportu.

## 7. Reżimy zmienności i dodatkowe hipotezy

Zmienność zdefiniowano jako odchylenie standardowe dziennych zwrotów z 20
obserwacji, bez annualizacji. Progi niskiej, średniej i wysokiej zmienności
wyznaczono przez tercyle **wyłącznie na treningu danego foldu**.

Pierwsza analiza przypisała istniejące transakcje do reżimów. Następnie
przeprowadzono osobną, pełną symulację filtrów, ponownie wybierając niekolidujące
wejścia ze wszystkich sygnałów. Sprawdzono cztery warianty: brak filtra,
pomijanie niskiej zmienności, close powyżej SMA50 i dodatnie momentum 20 obserwacji.
Pięć strategii i trzy scenariusze kosztów dały **60 przebiegów**.

Hipotezy ustalono przed tym przebiegiem, ale po analizie wcześniejszych wyników
na tej samej historii. Są eksploracyjne. Każdy model porównano z Always Long
działającym pod dokładnie tym samym filtrem.

**Pomijanie niskiej zmienności, long-only, koszt 0,50 USD/oz:**

| Strategia | Transakcje | Składany zwrot netto | Drawdown na zamknięciach |
| --- | ---: | ---: | ---: |
| Always Long + filtr | 453 | +47,56% | −25,52% |
| LR + filtr | 292 | +46,03% | −24,45% |
| Extra Trees + filtr | 207 | +42,87% | −19,94% |
| 5 głosów + filtr | 253 | +36,11% | −26,70% |
| ARX + filtr | 201 | +34,73% | −18,20% |

Filtr poprawił zwroty i obsunięcia badanych strategii ML względem ich wariantów
bez filtra. Przy koszcie 0,50 żaden z nich nie przekroczył zwrotu Always Long
z tym samym filtrem. Nie należy przypisywać całej poprawy modelowi.

**Wynik jest wrażliwy na koszt:** przy 1,00 USD/oz i tym samym filtrze LR
osiągnął +33,05%, Extra Trees +33,30%, a Always Long +30,08%. Mniejsza liczba
transakcji zmniejsza obciążenie kosztami. To wyjątek od rankingu przy 0,50,
ale nie dowód stabilnej przewagi na niezależnych danych.

Filtry SMA50 i momentum20 nie dały analogicznej poprawy modeli. Przy koszcie
0,50 wszystkie cztery modele miały niższy zwrot niż dopasowany Always Long
zarówno dla SMA50, jak i momentum20. Nie wybrano żadnego filtra jako
potwierdzonej strategii produkcyjnej.

## 8. Co zostało ukończone, a co pozostaje poza zakresem

| Obszar | Stan |
| --- | --- |
| Pobieranie i przygotowanie danych, audyt kalendarza | Zaimplementowane; kalendarz pozostaje przybliżeniem |
| Wskaźniki, cechy względne i międzyrynkowe | Zaimplementowane i wykorzystane w eksperymentach |
| Benchmark, ablation, chronologiczna walidacja i wybrane strojenie | Wykonane |
| Modele pojedyncze, ensemble, MLP i ARX | Sprawdzone; ograniczenia opisane wyżej |
| Symulacja wykonania, koszty i reżimy zmienności | Wykonane jako diagnostyka badawcza |
| Nietknięty holdout / test przyszłych sygnałów | Nie wykonano |
| Kalendarz i koszty konkretnego brokera, dzienny mark-to-market | Nie ukończono |
| Agent z egzekucją zleceń, monitoringiem i zarządzaniem ryzykiem | Nie wdrożono |
| LLM/news sentiment oraz handel demo | Nie wdrożono |

[Plan agenta](plan-agenta-xauusd.md) jest szerszy niż ukończone badanie.
Można zamknąć obecną wersję jako projekt portfolio bez deklarowania wykonania
wszystkich faz tamtego planu i bez oczekiwania na kolejne lata danych.

Naturalnym osobnym rozszerzeniem byłoby zamrożenie jednej specyfikacji,
uzupełnienie codziennej wyceny i założeń wykonania, a następnie ocena na
rzeczywiście niewykorzystanym okresie. Dalsze strojenie na obecnej historii
nie zastąpi tego testu.

## 9. Jak odtworzyć eksperymenty

Środowisko użyte do zapisanych wyników: Python 3.14.0; wersje bezpośrednich
zależności zapisano w [requirements.txt](requirements.txt). Jest to zapis
lokalnego środowiska, nie pełny lockfile wszystkich zależności. Instalacji od
zera nie zweryfikowano w ramach przygotowania raportu.

W `data/raw/` potrzebne są istniejące pliki: `xauusd_daily.csv`,
`dxy_reconstructed_daily.csv`, `dgs10_daily.csv`, `dfii10_daily.csv`,
`usd_broad_daily.csv`. XAU zawiera `datetime, open, high, low, close`, DXY
`datetime, dxy`, a pliki FRED `date, value`. Katalog `data/` jest ignorowany
przez Git: sam klon repozytorium nie zawiera tych danych. Ponowne pobranie
bieżących danych może zmienić próbę i wyniki. Porównaj hashe ze
[snapshotem raportu](reports/final/manifest.json).

W przygotowanym środowisku, z katalogu projektu:

```bash
# W istniejącym .venv; dla nowego środowiska najpierw utwórz venv.
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v

# Kolejność jest istotna: kolejne moduły korzystają z wcześniejszych wyników.
.venv/bin/python -m src.models.compare_h5_execution_models
.venv/bin/python -m src.models.tune_h5_execution_ensemble
.venv/bin/python -m src.models.evaluate_h5_arx
.venv/bin/python -m src.models.backtest_h5_execution_oof
.venv/bin/python -m src.models.analyze_h5_volatility_regimes
.venv/bin/python -m src.models.test_h5_trading_hypotheses
```

Moduły analityczne korzystają z lokalnych CSV i nie wymagają pobierania danych
ani kluczy API. Zapisują wyniki w odpowiednich katalogach `data/processed/`;
ponowne uruchomienie nadpisuje wyniki tych eksperymentów. Strojenie RF/HGB
jest najbardziej kosztowną częścią tej sekwencji. Raport powstał z już
zapisanych wyników, bez ponownego trenowania modeli.

Małe tabele źródłowe raportu i ich hashe znajdują się w
[reports/final/](reports/final/README.md). Pozwalają sprawdzić liczby bez
treningu, ale nie zastępują surowych danych ani kompletnych predykcji OOF.

Podczas przygotowania raportu uruchomiono cały zestaw: **25 testów przeszło**.
Sprawdzono także lokalne odnośniki raportu, zgodność siedmiu kopii CSV ze
źródłami i ich SHA256 oraz `git diff --check`. Nie wykonywano nowego treningu.

Testy obejmują m.in. daty targetu, odstęp między treningiem a walidacją,
brak wpływu przyszłych cen na wcześniejsze cechy/progi, koszty i brak
nakładania transakcji. Nie dowodzą poprawności wszystkich założeń ekonomicznych
ani dostępności danych makro w czasie rzeczywistym.

## 10. Jak przedstawić projekt na rozmowie

> Zbudowałem w Pythonie pipeline badawczy do prognozowania kierunku złota
> w horyzoncie pięciu obserwacji. Połączyłem dane cenowe z DXY i rentownościami,
> przeprowadziłem audyt kalendarza i porównanie modeli z chronologiczną
> walidacją. Sprawdziłem regresję logistyczną, modele drzewiaste, SVM,
> zespoły modeli, małą sieć neuronową i regresję z opóźnieniami zwrotów.
> Następnie zbadałem wykonanie sygnałów po kosztach i wpływ reżimów zmienności.
> Pokazałem, że niewielka poprawa AUC nie musi dawać lepszej strategii,
> a bardziej złożony ensemble nie musi przewyższać prostego baseline’u.

Na obronie lub rozmowie warto umieć wyjaśnić: różnicę między AUC a accuracy,
potrzebę chronologii i gap, odmienność targetu close-to-close i next-open,
różnicę między hard/soft voting oraz znaczenie dopasowanego baseline’u.
Nie należy deklarować dochodowego bota, niezależnego holdoutu ani przewagi
potwierdzonej statystycznie — tych rezultatów projekt nie dostarczył.

## 11. Dokumentacja eksperymentów

| Eksperyment | Opis i szczegółowe wyniki |
| --- | --- |
| Benchmark po zmianie kalendarza | [WEEKDAY_RESULTS.md](WEEKDAY_RESULTS.md) |
| Wybór cech | [ABLATION_H5_RESULTS.md](ABLATION_H5_RESULTS.md) |
| Modele i ensemble trzech modeli | [H5_EXECUTION_MODEL_COMPARISON.md](H5_EXECUTION_MODEL_COMPARISON.md) |
| Strojenie RF/HGB i pięć głosów | [H5_EXECUTION_ENSEMBLE_TUNING.md](H5_EXECUTION_ENSEMBLE_TUNING.md) |
| ARX i szósty głos | [H5_EXECUTION_ARX.md](H5_EXECUTION_ARX.md) |
| Symulacja po kosztach | [H5_EXECUTION_COST_BACKTEST.md](H5_EXECUTION_COST_BACKTEST.md) |
| Podział na reżimy | [H5_VOLATILITY_REGIMES.md](H5_VOLATILITY_REGIMES.md) |
| Dodatkowe filtry i 60 przebiegów | [H5_TRADING_HYPOTHESES.md](H5_TRADING_HYPOTHESES.md) |

Starszy [BACKTEST_H5_DIAGNOSTIC.md](BACKTEST_H5_DIAGNOSTIC.md) dotyczy wcześniejszej
diagnostyki targetu close-to-close przy wejściu na kolejnym open. Jego liczb
nie połączono z końcowym badaniem modeli trenowanych na target wykonania.
