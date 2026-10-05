# Tabele źródłowe raportu końcowego

Snapshot z 2026-10-05 do [raportu końcowego](../../FINAL_PROJECT_REPORT.md).
Pliki skopiowano z istniejących wyników `data/processed/` bez ponownego
trenowania modeli. Ten niewielki katalog jest przeznaczony do wersjonowania;
surowe dane i pełne wyniki nadal pozostają w ignorowanym katalogu `data/`.

| Plik | Zawartość |
| --- | --- |
| [model_comparison.csv](model_comparison.csv) | Pojedyncze modele i zespoły trzech modeli |
| [ensemble_tuning.csv](ensemble_tuning.csv) | RF/HGB po strojeniu i większość pięciu głosów |
| [arx.csv](arx.csv) | Regresja ARX i analiza szóstego głosu |
| [cost_backtest.csv](cost_backtest.csv) | Symulacja transakcji i scenariusze kosztów |
| [volatility_regimes.csv](volatility_regimes.csv) | Charakterystyki transakcji według zmienności |
| [trading_hypotheses.csv](trading_hypotheses.csv) | 60 przebiegów filtrów, modeli i kosztów |
| [splits.csv](splits.csv) | Daty i liczebności pięciu podziałów czasowych |
| [manifest.json](manifest.json) | Ścieżki źródeł, SHA256 tabel i wejść oraz wersje bibliotek |

W CSV zwroty, accuracy i drawdown są ułamkami: `0.10` oznacza 10%.
W raporcie procenty są zaokrąglone. AUC pozostaje liczbą z zakresu 0–1.
`std_auc` to odchylenie między foldami, nie błąd standardowy ani przedział ufności.
Drawdown dotyczy zamknięć transakcji, a zwroty portfela całego okresu OOF.

Hashe wejść opisują pliki lokalne w dniu przygotowania raportu. Hash XAU
zweryfikowano również względem metadanych porównania modeli. Sam manifest
nie dowodzi zgodności historycznych wejść każdego eksperymentu. Tabele
umożliwiają audyt liczb, ale pełne odtworzenie wymaga surowych danych,
kodu i ponownego uruchomienia modułów wskazanych w raporcie.
