# Skrypty CPU

Skrypty pomocnicze uruchamiane lokalnie (macOS, CPU) w ramach pracy magisterskiej
*Klasyfikacja aktywności osób w nagraniu wideo w oparciu o model typu Transformer 
i estymację pozy w ramkach wideo* (NTU RGB+D 60, PKU-MMD).

Numeracja odpowiada kolejności powstawania / etapom eksperymentu, nie kolejności
w pracy.

## Mapowanie klas kieszeniowych (etap przed/po korekcie)

W trakcie prac skorygowano mapowanie PKU-NTU dla klas kieszeniowych: obie
klasy PKU (29 – *put in pocket*, 38 – *take out from pocket*) mapują się na
tę samą klasę NTU 24 (*reach into pocket*), a nie jak pierwotnie na
NTU 56 (*touch pocket*). Ta korekta jest udokumentowana w pracy (rozdz. 4.2)
jako etap ablacji: wynik bazowy sprzed korekty (XSub 67,95% / XView 67,08%)
vs. wynik kanoniczny po korekcie (XSub 64,65% / XView 64,68%).

**Skrypty 2, 3, 4, 5, 9, 11, 12 odpowiadają etapowi sprzed korekty
(29→56)**, **skrypt 8 etapowi po korekcie (29→24, 38→24)**. Rozróżnienie
to jest celowe i spójne z chronologią opisaną w pracy, a nie pomyłką
wymagającą poprawienia.

## Opis skryptów

### 1. `1_preprocessing_NTU_RGBD_warianty_przetwarzania.py`
Preprocessing danych NTU RGB+D 60 (Etap I). Generuje warianty przetwarzania
szkieletów: baseline, interpolacja brakujących klatek, normalizacja skali,
oraz wariant łączony ("both"). Dotyczy wyłącznie NTU.

### 2. `2_preprocessing_PKU_MMD_interpolation_scale.py`
Preprocessing PKU-MMD odpowiadający wariantom baseline/interpolacja/skala/both
ze skryptu 1, na potrzeby ewaluacji zero-shot na PKU-MMD. Zawiera mapowanie
PKU-NTU.

### 3. `3_preprocessing_PKU_MMD_3tryby_porownanie.py`
Rozszerzenie skryptu 2 o dodatkowy wariant rotacji (baseline/interpolacja/
skala/rotacja/both) w celu porównania wpływu poszczególnych kroków
przetwarzania na wynik.

### 4. `4_preprocessing_PKU_MMD_Gaussian_smoothing.py`
Dodaje warianty wygładzania Gaussa (smooth_15, smooth_30) oraz ich
kombinacje z wariantem "both" (both_smooth15, both_smooth30). Test wpływu
wygładzania trajektorii szkieletu na dokładność.

### 5. `5_preprocessing_PKU_MMD_padding.py`
Test wpływu rozszerzenia (paddingu) granic segmentu o 5/10/15 klatek z każdej
strony względem oryginalnych etykiet czasowych PKU-MMD, wynik negatywny
(brak poprawy). 

### 6. `6_diagnostyka_progow_i_statystyk_NTU_PKU_MMD.py`
Skrypt diagnostyczny: liczy pliki .skeleton w NTU i etykiety w PKU-MMD,
zawiera progi filtrowania szumu ciała (noise_len_thres, noise_spr_thres) oraz
liczy rozkład segmentów testowych per klasa (w tym osobno klasy interakcyjne
i klasy kieszeniowe 29/38) dla XSub i XView.

### 7. `7_korekta_dlugosci_segmentow_uzasadnienie_MAX_FRAMES_300_PKU_MMD.py`
Analiza rozkładu długości segmentów akcji w PKU-MMD (średnia, mediana,
percentyle) uzasadniająca dobór stałej MAX_FRAMES=300 oraz pokazująca
odsetek segmentów wymagających paddingu vs. przycięcia.

### 8. `8_analiza_wplywu_normalizacji_skali_na_interakcje.py`
Analiza wpływu normalizacji skali (stosunku odległości stawów między dwiema
osobami) na klasy interakcyjne, wspólnie dla NTU i PKU-MMD.

### 9. `9_sliding_window_konwersja_segmentow_PKU_MMD.py`
Konwersja ciągłych sekwencji PKU-MMD na okna przesuwne o rozmiarze 64 klatek
z krokiem 16 (zgodnie z konfiguracją feedera SkateFormer), przygotowanie do
Etapu III (detekcja ciągła).

### 10. `10_sliding_window_ewaluacja_agregacja_softmax.py`
Ewaluacja modelu na oknach przesuwnych metodą agregacji softmax, z
porównaniem do wyników bazowych (baseline overall/interakcje dla XSub i
XView) oraz per-class accuracy dla 8 klas interakcyjnych.

### 11. `11_sliding_window_sweep_W_S_sekwencje_ciagle_PKU_MMD.py`
Przeszukiwanie (sweep) kombinacji rozmiaru okna W i kroku S
(W∈{32,64,96,128}, dodatkowo S∈{8,32} przy W=64) na sekwencjach ciągłych
PKU-MMD, osobno dla XSub i XView.

### 12. `12_sliding_window_W64_S8_XView_PKU_MMD.py`
Finalna konfiguracja W=64, S=8 dla splitu Cross-View, wybrana na podstawie
wyników sweepu ze skryptu 11.

## Podsumowanie zależności między skryptami

```
1  preprocessing NTU (Etap I)
2,3,4,5 warianty preprocessingu PKU (Etap II), kolejne ablacje
6,7 diagnostyka i uzasadnienie parametrów (MAX_FRAMES, progi szumu)
8  analiza skali / poprawione mapowanie kieszeni
9  konwersja PKU na okna przesuwne (Etap III, baza)
11 sweep W/S na bazie 9
12 finalna konfiguracja W64/S8 XView na bazie 11
10 ewaluacja finalnego modelu na oknach z 9/11/12
```
