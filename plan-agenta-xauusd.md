# Plan budowy agenta tradingowego XAU/USD

Dokument roboczy — wracaj tu i aktualizuj status w miarę postępu. Każda faza ma bramkę (gate) — nie przechodzisz dalej, dopóki kryteria nie są spełnione.

---

## Faza 0 — Fundamenty projektu (masz to częściowo zrobione)

**Cel:** solidna baza danych i modelu, zanim cokolwiek zacznie "handlować".

- [x] Dane OHLCV z Twelve Data API
- [x] Modele klasyczne: logistic regression, Random Forest, XGBoost
- [x] Walk-forward validation
- [x] Cechy techniczne (RSI, SMA, EMA) implementowane ręcznie
- [x] Cechy intermarket (DXY, realne rentowności)
- [ ] **Do dodania:** cechy sentymentu z LLM na FOMC/newsach makro (Twój projekt do koła naukowego — to naturalnie staje się warstwą fundamentalną agenta, nie osobnym bytem)

**Bramka:** masz baseline (same cechy techniczne) i wariant z sentymentem, porównywalne na tym samym walk-forward split.

---

## Faza 1 — Silnik backtestingu z realnymi kosztami

**Cel:** żaden wynik nie liczy się, dopóki nie przejdzie przez backtester z kosztami.

- [ ] Dodaj do backtestu: spread (typowy dla XAU/USD ok. 0.2–0.5$ u brokerów retail), poślizg (slippage), prowizję
- [ ] Policz metryki po kosztach: Sharpe, max drawdown, win rate, profit factor
- [ ] Porównaj wynik "na sucho" (bez kosztów) vs "po kosztach" — jeśli przewaga znika, to sygnał, że była iluzoryczna

**Narzędzia (darmowe):** `backtrader`, `vectorbt` (open-source), albo własny silnik w pandas — dla dziennych danych własny silnik jest prostszy i bardziej przejrzysty do publikacji.

**Bramka:** masz raport backtestu z kosztami transakcyjnymi na out-of-sample danych.

---

## Faza 2 — Walidacja sygnału out-of-sample i po reżimach

**Cel:** sprawdzić, czy przewaga przeżywa różne warunki rynkowe, nie tylko jeden okres.

- [ ] Test na danych spoza okresu strojenia hiperparametrów (prawdziwy hold-out, nie tylko walk-forward w pętli)
- [ ] Podział na reżimy: covid 2020, podwyżki stóp 2022, okres niskiej zmienności — sprawdź stabilność metryk w każdym
- [ ] Test istotności statystycznej (np. permutation test — czy wynik odróżnia się od losowego tradingu)

**Bramka:** model ma sensowne metryki w co najmniej 2 różnych reżimach, nie tylko w jednym korzystnym okresie.

---

## Faza 3 — Warstwa fundamentalna jako niezależny filtr

**Cel:** fundamenty redukują ryzyko, nie próbują "poprawić trafność" modelu technicznego.

- [ ] Kalendarz makro (Forex Factory / Investing.com) — flaga "high-impact event w oknie X godzin"
- [ ] Reguła: wycisz/zmniejsz sygnał techniczny wokół newsów wysokiego ryzyka (NFP, CPI, FOMC)
- [ ] Cechy sentymentu z LLM (z Fazy 0) jako dodatkowy modyfikator, testowany osobno od filtru newsowego
- [ ] Zmierz efekt: czy dodanie filtra zmniejsza drawdown, nawet jeśli nie zwiększa zwrotu

**Bramka:** masz wynik A/B — model + filtr fundamentalny vs model bez — na tych samych danych testowych.

---

## Faza 4 — Warstwa decyzji i risk management

**Cel:** to jest miejsce, gdzie większość projektów pada — nie pomijaj.

- [ ] Position sizing (np. stały % kapitału na ryzyko na transakcję, nie stały wolumen)
- [ ] Twardy stop-loss na każdej pozycji
- [ ] Max dzienna/tygodniowa strata → automatyczne wyłączenie agenta (kill switch)
- [ ] Logika łącząca: sygnał ML + filtr fundamentalny + limity ryzyka → decyzja tak/nie

**Bramka:** masz moduł ryzyka, który działa niezależnie od modelu i może zablokować każdy sygnał.

---

## Faza 5 — Egzekucja na koncie demo

**Cel:** sprawdzić system na żywych danych, bez realnego kapitału.

- [ ] Wybór platformy: `MetaTrader5` (pakiet Python) na koncie demo brokera forex/CFD, albo OANDA v20 REST API (darmowe konto practice)
- [ ] Integracja: agent wysyła zlecenia programowo (market/limit order, SL/TP)
- [ ] Obsługa błędów: co się dzieje, gdy API nie odpowiada, dane są opóźnione, zlecenie się nie wykona

**Bramka:** agent wykonuje zlecenia na demo bez ręcznej interwencji przez min. kilka dni bez awarii.

---

## Faza 6 — Monitoring, logowanie, paper trading

**Cel:** długi test na żywo + pełna audytowalność każdej decyzji.

- [ ] Log każdej decyzji: sygnał, filtr fundamentalny, decyzja ryzyka, wynik — nawet gdy agent NIE handluje (dlaczego odrzucił sygnał)
- [ ] Alerty Telegram (bot, darmowy) na wykonane transakcje i błędy
- [ ] Prosty dashboard (Streamlit + darmowy hosting Community Cloud) do podglądu na żywo
- [ ] Paper trading min. kilka tygodni — dopiero to pokazuje realną skuteczność, backtest tego nie zastąpi

**Bramka:** masz min. kilkutygodniowy log paper tradingu, gotowy do analizy i do wykorzystania w publikacji.

---

## Faza 7 — Podsumowanie i (opcjonalnie) dalsze kroki

- [ ] Zestawienie wyników: backtest vs paper trading — czy są zgodne? Jeśli nie, dlaczego (przeuczenie, koszty, zmiana reżimu)
- [ ] Dokumentacja do publikacji: metodologia, wyniki, ograniczenia
- [ ] Decyzja: zostajesz na demo (bezpieczne, wystarczające do portfolio/publikacji) czy rozważasz realny kapitał (osobna rozmowa o regulacjach, brokerach, ryzyku)

---

## Zasada przewodnia na każdym etapie

Kolejność ważności: **risk management > filtr fundamentalny > model predykcyjny**. Jeśli musisz coś skrócić z braku czasu, skracaj złożoność modelu, nigdy warstwę ryzyka.
