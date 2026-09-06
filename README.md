# SkateFormer: transfer zero-shot NTU RGB+D 60 → PKU-MMD

Materiały uzupełniające do pracy magisterskiej Klasyfikacja aktywności 
osób w nagraniu wideo w oparciu o model typu Transformer i estymację 
pozy w ramkach wideo* (Urszula Szczepańska, Wydział Elektroniki i 
Technik Informacyjnych, Politechnika Warszawska, 
promotor: prof. Włodzimierz Kasprzak).

Praca ocenia model [SkateFormer](https://github.com/KAIST-VICLab/SkateFormer)
(KAIST-VICLab, ECCV 2024), pretrenowany na NTU RGB+D 60, w zadaniu
rozpoznawania akcji na PKU-MMD bez ponownego treningu (zero-shot), w trzech
etapach: klasyfikacja natywna na NTU RGB+D 60 (Etap I), klasyfikacja
segmentów PKU-MMD z fuzją joint+bone (Etap II), detekcja ciągła metodą
okna przesuwnego (Etap III).

## Struktura repozytorium

```
cpu/            12 skryptów uruchamianych lokalnie (CPU): preprocessing,
                diagnostyka, konwersja okien przesuwnych
                opis: cpu/skrypty_CPU.md
gpu/            10 notebooków Google Colab (GPU, A100 / L4): inferencja,
                ewaluacja, ablacje
                opis: gpu/skrypty_GPU_colab.md
ntu_official/   3 oryginalne skrypty preprocessingu NTU RGB+D
                (Microsoft Corporation, licencja MIT, niemodyfikowane)
                opis: ntu_official/skrypty_oficjalne_NTU.md
```

Każdy folder zawiera plik `.md` z opisem przeznaczenia każdego skryptu/
notebooka i jego powiązania z rozdziałami pracy.

## Ważna uwaga: korekta mapowania klas kieszeniowych

W trakcie prac skorygowano mapowanie PKU→NTU dla dwóch klas dotyczących
kieszeni (PKU 29 – *put in pocket*, PKU 38 – *take out from pocket*): obie
mapują się na NTU 24 (*reach into pocket*), a nie na NTU 56 (*touch
pocket*), jak w pierwotnej wersji. Korekta jest udokumentowana w pracy
(rozdz. 4.2) jako etap ablacji, nie jako poprawka błędu ukrywana w tle.
Skrypty i notebooki oznaczone jako "sprzed korekty" reprezentują ten
wcześniejszy, chronologicznie pierwszy etap projektu i są celowo
zachowane jako punkt odniesienia. Szczegóły w plikach `.md` w każdym
folderze.

## Dane

Repozytorium **nie zawiera** danych NTU RGB+D ani PKU-MMD: oba zbiory
wymagają osobnej rejestracji i akceptacji licencji użytkowania u
oryginalnych dystrybutorów. Skrypty operują na lokalnych ścieżkach
(`~/skateformer_pku/...`, `~/SkateFormer/data/ntu/...`) i wymagają
samodzielnego pobrania danych zgodnie z warunkami licencji.

## Wagi modelu

Wagi SkateFormer (`SkateFormer_j.pt`, `SkateFormer_b.pt`) pochodzą z
oficjalnego repozytorium [KAIST-VICLab/SkateFormer](https://github.com/KAIST-VICLab/SkateFormer)
i nie są dołączone tutaj.

## Licencja

Skrypty w `ntu_official/` pochodzą z oficjalnego pipeline'u NTU RGB+D
(Microsoft Corporation) i są objęte licencją MIT. Oryginalne nagłówki
licencyjne są zachowane bez zmian. Pozostałe skrypty i notebooki (`cpu/`,
`gpu/`) są mojego autorstwa i zostały udostępnione w ramach
materiałów do pracy magisterskiej.
