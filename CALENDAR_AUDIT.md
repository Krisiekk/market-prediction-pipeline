# Audyt kalendarza XAU/USD — zatrzymanie przed holdoutem

**Holdout nie został utworzony ani uruchomiony.** Zgodnie z warunkiem zadania
zatrzymujemy się po audycie: H5 liczy rekordy, a nie zweryfikowane sesje.
Dodatkowo najnowsza historia została już wykorzystana w CV i ablation.
Nie zmieniono pipeline'u, targetu, cech, intermarket ani modelu.

Uruchomienie: `.venv/bin/python -m src.validation.calendar_audit`.
Wyniki: `data/processed/calendar_audit/` (HTML, JSON i CSV).

## Zakres i kalendarz

| Dane | Początek | Koniec | Wiersze |
|---|---|---|---:|
| Surowe | 2008-02-22 | 2026-09-27 | 5000 |
| Odtworzony dataset benchmarku H5, lag=1 | 2008-05-02 | 2026-09-22 | 4945 |

| Dzień | Surowe | Po feature engineering |
|---|---:|---:|
| Poniedziałek | 970 | 960 |
| Wtorek | 968 | 958 |
| Środa | 968 | 957 |
| Czwartek | 969 | 958 |
| Piątek | 966 | 955 |
| Sobota | 101 | 100 |
| Niedziela | 58 | 57 |

Brak duplikatów dat. Weekendowe rekordy: 159 surowych / 157 przetworzonych.
Pierwsza sobota: 2024-10-12. Pierwsza niedziela: 2025-04-27.
Liczba weekendowych rekordów rocznie: 2024: 11; 2025: 70; 2026: 78.
To zmiana struktury kalendarza w końcowej części historii.

| Odstęp | Surowe | Po feature engineering |
|---|---:|---:|
| 1 dzień | 4082 | 4037 |
| 2 dni | 46 | 46 |
| 3 dni | 866 | 856 |
| Inne: 4 dni | 5 | 5 |

Istniejący `xauusd_daily_features_h5.csv` jest starszym artefaktem: 4323 wiersze,
2008-03-20–2026-09-22; nie jest używany przez obecny benchmark. Audyt zapisuje
jego statystyki oddzielnie. Nie nadpisano go.

## Czy weekendowe ceny są sztuczne?

Wszystkie 159 rekordów mają skończone, dodatnie i algebraicznie spójne OHLC.
Jednak spójność OHLC nie dowodzi poprawności sesji ani autentyczności notowania.
158/159 zmienia close względem poprzedniego wiersza. Każda z 58 niedziel ma
poprzedzającą sobotę i inny close niż sobota. Jeden rekord, 2025-04-19,
ma O=H=L=C=3326,27 i jest identyczny z piątkiem 2025-04-18.
To jest zgodne z powieleniem/forward-fill, lecz sam CSV nie dowodzi mechanizmu.
Nie ma dowodu, że cała seria weekendowa powstała przez prosty forward-fill.

Mediana (high-low)/close: dni robocze 1,3103%, soboty 0,2801%, niedziele 0,0246%.
Weekendowe zakresy są wyraźnie mniejsze. Nie można na tej podstawie uznać ich
za pełne, porównywalne sesje. Kod fetch pobiera Twelve Data interval=1day,
bez jawnej strefy czasowej; zapisany CSV nie zachowuje metadanych strefy/sesji.

Przykład pełnego fragmentu:

| Data | Dzień | Open | High | Low | Close |
|---|---|---:|---:|---:|---:|
| 2025-04-25 | piątek | 3354,80 | 3376,80 | 3270,60 | 3316,30 |
| 2025-04-26 | sobota | 3316,60 | 3319,60 | 3309,90 | 3314,00 |
| 2025-04-27 | niedziela | 3315,10 | 3324,60 | 3315,10 | 3321,00 |
| 2025-04-28 | poniedziałek | 3319,80 | 3352,91 | 3274,30 | 3344,10 |

Trzy fragmenty piątek–poniedziałek znajdują się w `weekend_examples.csv` i raporcie HTML.
Brakująca niedziela w pierwszym fragmencie jest pokazana jako NaN, nie dodana do danych.

## Rzeczywista definicja H5

`build_features.py` sortuje daty, buduje cechy i przesuwa intermarket o lag.
Następnie liczy `future_close = df["close"].shift(-horizon)` i
`target = future_close > close`, a dopiero później usuwa NaN.
Target to dokładnie **piąty kolejny wiersz surowego kalendarza**, nie piąty
wiersz po dropna i nie potwierdzone pięć sesji tradingowych.
Odtworzenie targetów i label_end sprawdzono assertion dla wszystkich 4945 wierszy.

- 2024-10-07 → 2024-10-12: 5 dni kalendarzowych, 5 kolejnych rekordów,
  ale tylko 4 rekordy poniedziałek–piątek i 1 sobota.
- 2025-04-25 → 2025-04-30: 5 dni kalendarzowych, 3 rekordy poniedziałek–piątek
  i 2 weekendowe. Close zmienia się z 3316,30 do 3292,00.
- 2026-09-18 → 2026-09-23: 5 dni kalendarzowych, 3 rekordy poniedziałek–piątek
  i 2 weekendowe; 4377,24525 → 4286,97334.

Liczenie dni poniedziałek–piątek to tylko diagnostyczne przybliżenie sesji.
Dokładnej liczby rzeczywistych sesji nie można potwierdzić bez kalendarza i strefy
źródła. 559 etykiet H5 obejmuje rekord weekendowy (także 559 w próbie benchmarku),
pierwsza od 2024-10-07. Zatem horyzont nie ma jednolitej interpretacji jako pięć
sesji w całej historii. Nie należy automatycznie usuwać weekendów bez wyjaśnienia,
czy nie są fragmentami sesji przesuniętymi przez strefę czasową.

## Dlaczego nie utworzono holdoutu

Orientacyjne granice, bez wyboru i bez trenowania:

| Końcowa część przetworzonej próby | Początek | Koniec | N | Już ocenione w CV |
|---|---|---|---:|---:|
| 15% | 2024-06-21 | 2026-09-22 | 742 | 742 |
| 20% | 2023-07-11 | 2026-09-22 | 989 | 989 |

Nie ma ustalonego development ani holdoutu. N holdoutu i wszystkie jego metryki:
nie dotyczy — test nie został wykonany. CV A+C z zapisanego ablation: 0,537450.
Nie ma holdout AUC, delty, analizy okresów ani podstaw do oceny generalizacji.
Wcześniejsze CV/ablation korzystały z całej dostępnej, etykietowanej końcówki;
samo ponowne chronologiczne odcięcie jej nie przywróci niezależności testu.

## Konieczne następne kroki — bez wykonania zmian

1. Potwierdzić u dostawcy definicję świecy 1day, strefę i pochodzenie weekendowych
   wpisów oraz kompletność ostatniej świecy. Ustalić jawny kalendarz sesji.
2. Jeżeli to rekordy pozasesyjne/sztuczne, wykluczać je przed budowaniem cech
   i targetu; jeżeli fragmenty sesji, najpierw poprawnie agregować do sesji.
   Nie wystarczy usunąć weekendów dopiero po utworzeniu targetu.
3. Na poprawionym kalendarzu przeliczyć wszystkie rolling windows, lag=1 i H5.
   To zmienia eksperyment: potrzebne ponowne CV/benchmark i ablation na development,
   a dotychczasowych AUC nie można traktować jako porównywalnego punktu odniesienia.
4. A+C pozostaje kandydatem, ale zamrożenie do nowego targetu wymaga ponownej
   oceny. Nietknięty holdout musi pochodzić z okresu nieużytego w dotychczasowych
   analizach, np. nowych danych zebranych po zamrożeniu. Nie nazywać starej
   końcówki niezależnym holdoutem.

Kontrola backward-asof po lag=1 nie wykazała źródłowych dat intermarket późniejszych
niż obserwacja XAU. Daty nie dowodzą dostępności publikacji: nadal brak historycznych
godzin publikacji i vintage FRED. Szczegóły w `intermarket_date_checks.csv`.
Przeszło 9 testów. Testów treningu holdout nie uruchamiano, bo etap 2 jest zablokowany
wynikiem audytu; żaden scaler ani model nie został tutaj dopasowany.
