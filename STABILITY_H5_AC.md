# Stabilność H5 LogisticRegression — TECHNICAL + A+C

Analiza używa wyłącznie zapisanych OOF predykcji z `ablation_h5_weekdays`.
Nie dopasowano modeli ponownie. Cechy diagnostyczne odtworzono z RAW przez
`prepare_dataset` i istniejący pipeline weekdays. Wszystkie daty/targety
OOF dopasowały się 1:1.

## Granice foldów i metryki

| fold | validation_start | validation_end | n_validation | positive_rate | auc | balanced_accuracy | accuracy | mean_future_return | median_future_return | std_future_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2011-05-30 | 2014-06-18 | 797 | 0.5232 | 0.5241 | 0.5158 | 0.5332 | -0.0006 | 0.0010 | 0.0263 |
| 2 | 2014-06-19 | 2017-07-10 | 797 | 0.4705 | 0.5261 | 0.5031 | 0.4806 | -0.0003 | -0.0014 | 0.0195 |
| 3 | 2017-07-11 | 2020-08-03 | 797 | 0.5822 | 0.5245 | 0.5057 | 0.5182 | 0.0034 | 0.0028 | 0.0185 |
| 4 | 2020-08-04 | 2023-08-24 | 797 | 0.5132 | 0.5941 | 0.5602 | 0.5596 | -0.0001 | 0.0007 | 0.0197 |
| 5 | 2023-08-25 | 2026-09-18 | 797 | 0.6110 | 0.4779 | 0.4704 | 0.4065 | 0.0054 | 0.0061 | 0.0277 |

Pięć foldów po 797 OOF obserwacji. Positive_rate to udział target UP.
Future return = future_close/close−1 dla H5; daily volatility = odchylenie
standardowe dziennych close-to-close returns w danym foldzie. To statystyki
opisowe, bez testów istotności.

## Porównanie TECHNICAL i A+C

| fold | date_start | date_end | technical_auc | ac_auc | delta_pp |
| --- | --- | --- | --- | --- | --- |
| 1 | 2011-05-30 | 2014-06-18 | 0.5125 | 0.5241 | 1.1599 |
| 2 | 2014-06-19 | 2017-07-10 | 0.5146 | 0.5261 | 1.1450 |
| 3 | 2017-07-11 | 2020-08-03 | 0.4898 | 0.5245 | 3.4742 |
| 4 | 2020-08-04 | 2023-08-24 | 0.5649 | 0.5941 | 2.9182 |
| 5 | 2023-08-25 | 2026-09-18 | 0.4615 | 0.4779 | 1.6361 |

A+C poprawia AUC względem TECHNICAL w **5/5 foldów**. Delta jest najmniejsza
w foldzie 1 (+1,16 pp) i największa w foldzie 3 (+3,47 pp), nie tylko w foldzie 4.

## Wpływ folda 4 na średnią

| metric | value |
| --- | --- |
| AUC mean all folds | 0.5293 |
| AUC median all folds | 0.5245 |
| AUC sample std all folds | 0.0415 |
| AUC mean folds 1,2,3,5 (diagnostic only) | 0.5131 |
| Fold 4 contribution to overall mean (pp) | 2.0234 |

Fold 4 podnosi średnią z pozostałych czterech foldów 0,513126 do pełnej średniej
0,529313: różnica wynosi **1,62 pp**. To jest wkład folda 4 do pięciofoldowej
średniej w porównaniu ze średnią pozostałych czterech foldów. Mediana 0,524516 jest niższa od średniej; std 0,041534 jest duże
względem samego efektu.

## Cechy i rozkłady w foldach

| fold | feature | n | mean | std | median | min | max | pearson_target | pearson_future_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | dfii10_vs_sma20 | 797 | -0.0049 | 0.1114 | -0.0145 | -0.5570 | 0.5190 | -0.0041 | -0.0742 |
| 1 | dxy_sma20_vs_sma50 | 797 | 0.0013 | 0.0081 | 0.0003 | -0.0185 | 0.0270 | 0.0665 | 0.0421 |
| 2 | dfii10_vs_sma20 | 797 | 0.0019 | 0.0878 | 0.0010 | -0.2575 | 0.2620 | -0.0388 | -0.0566 |
| 2 | dxy_sma20_vs_sma50 | 797 | 0.0037 | 0.0106 | 0.0024 | -0.0191 | 0.0270 | -0.0488 | -0.0427 |
| 3 | dfii10_vs_sma20 | 797 | -0.0172 | 0.0931 | -0.0170 | -0.3945 | 0.7590 | -0.0246 | 0.1159 |
| 3 | dxy_sma20_vs_sma50 | 797 | -0.0001 | 0.0077 | 0.0012 | -0.0206 | 0.0202 | -0.1166 | -0.1420 |
| 4 | dfii10_vs_sma20 | 797 | 0.0327 | 0.1404 | 0.0215 | -0.4215 | 0.6085 | -0.0562 | -0.0317 |
| 4 | dxy_sma20_vs_sma50 | 797 | 0.0010 | 0.0112 | 0.0028 | -0.0301 | 0.0242 | -0.0087 | 0.0230 |
| 5 | dfii10_vs_sma20 | 797 | 0.0080 | 0.0982 | 0.0090 | -0.3590 | 0.3685 | 0.0170 | 0.0090 |
| 5 | dxy_sma20_vs_sma50 | 797 | -0.0004 | 0.0096 | -0.0006 | -0.0260 | 0.0184 | 0.0056 | -0.0307 |

A↔C korelacja wynosi kolejno −0,029, −0,017, −0,080, **0,220**, **0,315**.
Ostatnie dwa foldy mają więc relację A/C inną niż pierwsze trzy, ale Pearson cecha–target
pozostaje mały (w foldzie 5: A 0,017, C 0,006). Proste korelacje z przyszłym H5 return
również nie wskazują mocnego liniowego związku. Nie służą do selekcji.

### Fold 4 względem pozostałych

Okres: **2020-08-04–2023-08-24**, 797 obserwacji; UP 51,32%, prawie równy podział.
Dzienne XAU volatility 0,933% versus 1,026% w innych foldach; std H5 return 1,968%
versus 2,301%. Nie jest to wyjątkowo wysoka zmienność złota.

W danych intermarket wyróżniają się większa zmienność rentowności w tym okresie:
std DGS10 1,142 pp vs 0,439 pp poza foldem; std DFII10 1,064 pp vs 0,350 pp.
Średnia realna rentowność to 0,015% vs 0,697%; A ma średnią +0,0327 pp
(std 0,1404 vs 0,0976). DXY średnio 98,39 vs 93,15; std 6,39 vs 3,45.
C ma podobną średnią (0,00098 vs 0,00113), ale większe std (0,0112 vs 0,0090).
A/C korelacja +0,220 vs +0,047. Fold 4 jest więc wyraźnie inny w rozkładzie
zmian rentowności i A/C, choć nie w samej zmienności XAU. To opis różnic w danych,
nie wyjaśnienie przyczyn AUC.

### Fold 5 względem wcześniejszych foldów

Okres: **2023-08-25–2026-09-18**, 797 obserwacji; UP **61,10%**, DOWN 38,90%.
AUC 0,477870, balanced accuracy 0,470408. Średni H5 return +0,543%, std 2,770%;
dzienne volatility 1,258% — wyższe niż pozostałe foldy (std H5 2,101%, daily 0,945%).

Średni poziom DXY 101,99 (pozostałe foldy 92,25). Trendowe cechy DXY A/C mają
średnie bliskie zeru: A +0,00796 pp (pozostałe +0,00315), C −0,00040
(pozostałe +0,00147). Średnia DGS10 4,310% (pozostałe 2,217%), DFII10 2,003%
(pozostałe 0,200%). A ma std 0,0982 (pozostałe 0,1082), C std 0,00961
(pozostałe 0,00941). A↔C korelacja rośnie do 0,315.

Najbardziej widoczna zmiana operacyjna modelu jest w score: UP rate w targetach to
61,1%, lecz model przewidział UP tylko w **20,6%** wierszy. Średnie prawdopodobieństwo
UP (sigmoid zapisanej funkcji decyzyjnej) to 0,430, mediana 0,424. W foldzie 4 model
przewidział UP w 47,9% obserwacji, średnie probability 0,500. To wskazuje na przesunięcie
rozkładu score/klasyfikacji wobec częstości klas, przy progu 0,5; próg pozostaje
bez zmian. Fold 5 ma też najniższy wynik z ostatnich lat, lecz roczne AUC nie jest
jednostajne — np. 2025 wraca do 0,545.

## Score modelu

Zapisany `score` jest `decision_function`; probability UP obliczono jako sigmoid(score),
co odpowiada binarnemu `predict_proba` LogisticRegression. AUC liczono z oryginalnego
score (ranking jest ten sam).

| fold | mean | std | min | p25 | median | p75 | max | decision_mean | decision_std | decision_min | decision_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.6325 | 0.1212 | 0.0634 | 0.5948 | 0.6615 | 0.7112 | 0.8506 | 0.5595 | 0.5526 | -2.6930 | 1.7389 |
| 2 | 0.5690 | 0.0588 | 0.3414 | 0.5296 | 0.5695 | 0.6100 | 0.7551 | 0.2818 | 0.2433 | -0.6569 | 1.1262 |
| 3 | 0.5111 | 0.0533 | 0.3465 | 0.4760 | 0.5124 | 0.5475 | 0.6497 | 0.0449 | 0.2157 | -0.6343 | 0.6179 |
| 4 | 0.4995 | 0.0572 | 0.3087 | 0.4613 | 0.4976 | 0.5378 | 0.6882 | -0.0018 | 0.2318 | -0.8060 | 0.7918 |
| 5 | 0.4304 | 0.1471 | 0.0707 | 0.3539 | 0.4236 | 0.4864 | 0.9980 | -0.2565 | 0.8620 | -2.5765 | 6.2337 |

Fold 1–2 mają wysokie średnie prawdopodobieństwa i przewidują UP w ok. 88% wierszy;
fold 3–4 są skupione bliżej 0,5; fold 5 przesuwa score w dół i mocno go rozszerza.
Model nie daje stale score≈0,5; rozkład zależy od foldu.

## AUC per year i rolling AUC

Rok oznaczono low_sample poniżej 100 obserwacji; żaden z raportowanych lat OOF nie
spadł poniżej progu (2026 ma 187). Pojedyncze lata są opisowe.

| year | n | auc | balanced_accuracy | accuracy | positive_rate | low_sample |
| --- | --- | --- | --- | --- | --- | --- |
| 2011 | 155 | 0.6109 | 0.5590 | 0.5806 | 0.5613 | False |
| 2012 | 261 | 0.4734 | 0.5034 | 0.5172 | 0.5172 | False |
| 2013 | 260 | 0.5750 | 0.5256 | 0.4923 | 0.4615 | False |
| 2014 | 260 | 0.5559 | 0.4944 | 0.5038 | 0.5115 | False |
| 2015 | 261 | 0.5309 | 0.5303 | 0.4406 | 0.3908 | False |
| 2016 | 261 | 0.4997 | 0.4919 | 0.5096 | 0.5249 | False |
| 2017 | 260 | 0.5976 | 0.5251 | 0.5731 | 0.5692 | False |
| 2018 | 259 | 0.5501 | 0.4986 | 0.4942 | 0.4826 | False |
| 2019 | 260 | 0.4766 | 0.4808 | 0.4962 | 0.6385 | False |
| 2020 | 262 | 0.5647 | 0.5718 | 0.5725 | 0.6107 | False |
| 2021 | 261 | 0.5276 | 0.5045 | 0.5134 | 0.5441 | False |
| 2022 | 260 | 0.6750 | 0.6219 | 0.6154 | 0.5192 | False |
| 2023 | 259 | 0.4779 | 0.4600 | 0.4595 | 0.5019 | False |
| 2024 | 260 | 0.4844 | 0.4991 | 0.4192 | 0.6038 | False |
| 2025 | 259 | 0.5446 | 0.5148 | 0.3436 | 0.6911 | False |
| 2026 | 187 | 0.5053 | 0.4932 | 0.4920 | 0.5134 | False |

Rolling AUC to trailing OOF score, bez retreningu. Okna nakładają się, więc punkty
są mocno zależne. Wartości na końcu są ważniejsze dla oceny ostatniego pogorszenia:

| window | min_auc | min_end | max_auc | max_end | last_auc | last_date |
| --- | --- | --- | --- | --- | --- | --- |
| 250 | 0.3440 | 2024-04-12 | 0.7162 | 2023-03-27 | 0.4948 | 2026-09-18 |
| 500 | 0.4117 | 2025-03-17 | 0.6421 | 2023-01-20 | 0.4873 | 2026-09-18 |

250-window AUC rośnie w 2020–2023, spada gwałtownie w 2024 (średnia roczna rolling
AUC 0,429), odbija w 2025 i kończy 0,495. Window 500 wygładza zmianę: średnia 2023
0,612, 2024 0,510, 2025 0,471, 2026 0,511; ostatni punkt 0,487. To nie wygląda na
równą, stopniową degradację. Jest wyraźne pogorszenie od okolic 2024, częściowe
odbicie i słaby wynik w foldzie 5. Rolling nie pozwala sprowadzić go ani do jednego
pojedynczego dnia, ani do trwałego, monotonicznego spadku.

## Leakage / jakość danych / dalszy krok

W zapisanych artefaktach nie znaleziono sygnału bezpośredniego leakage: OOF daty i
klasy zgadzają się 1:1 z builderem weekday H5; daty `label_end_date` odpowiadają
`label_end`; foldy zachowują gap=5; features powstają z rolling/asof wstecz i lag=1.
To nie dowodzi pełnej jakości historycznych danych. Pozostają brak godzin publikacji
i vintage FRED oraz ewentualne problemy z timestampem świec 1day. AUC z nakładającymi
się etykietami H5 i rolling/roczne podziały nie są niezależnymi testami.

**Wniosek:** Fold 4 ma nietypowe rozkłady zmian nominalnych/realnych rentowności i
silniejszą korelację A/C; Fold 5 ma wyraźnie inne poziomy DXY i rentowności, wyższy
udział UP i volatility XAU oraz mocno przesunięty w dół rozkład model score. Dane
wspierają opis zmiany rozkładów, ale nie pozwalają przypisać jej jednej przyczyny.
Nie widać technicznego błędu, który sam w sobie blokowałby kontrolowany tuning,
ale seria AUC i zmienność kalibracji oznaczają wysokie ryzyko niestabilności.
Przed jakimkolwiek tuningiem warto uzgodnić plan eksperymentu na development;
wyniki te nie są holdoutem i nie powinny sterować dalszą selekcją.

CSV rolling 250/500, roczne, score, cechy/foldy, contrast, OOF z diagnostykami,
metadane i pełna tabela foldów znajdują się obok tego raportu.
