# Notebooki GPU (Google Colab)

Notebooki uruchamiane na GPU (Google Colab, A100 / L4), obejmujące ewaluację
i analizę modelu SkateFormer na PKU-MMD w trybie zero-shot (Etap II:
klasyfikacja segmentów, Etap III: detekcja ciągła) oraz weryfikację
preprocessingu na natywnym NTU RGB+D 60 (Etap I).

## 1. `SkateFormer NTU-60 Sanity Check Preprocessing.ipynb`
Etap I. Weryfikuje wpływ preprocessingu (baseline / interpolacja /
normalizacja skali / both) na natywnym NTU RGB+D 60, w obu protokołach
(CS/CV), na wagach jawnie nazwanych per protokół
(`SkateFormer_j_CSub.pt` / `SkateFormer_j_CView.pt`). Stanowi źródło aktualnych wartości
tabeli Etapu I w pracy (baseline CS=92,62%/CV=97,02% itd.).

## 2. `PKU Ewaluacja XSub XView both vs baseline.ipynb`
Etap II. Ewaluacja PKU-MMD (strumień joint) dla wariantów preprocessingu
baseline i "both", osobno dla XSub i XView. Punkt odniesienia pokazujący
przyrost dokładności dzięki preprocessingowi "both" względem baseline.

## 3. `PKU ewaluacja XSub XView bez korekty kieszeni.ipynb`
Wariant kontrolny do notebooka powyżej: te same dane, ten sam model i
inferencja, ale bez korekty mapowania klas 29/38 (klasa 29 pozostaje
zmapowana na NTU 56 zamiast 24), czyli stan sprzed korekty opisanej w pracy.
Izoluje wpływ samej korekty mapowania kieszeni na overall accuracy
(dokładność klas interakcyjnych 49–57 pozostaje identyczna, korekta
dotyczy wyłącznie klasy kieszeniowej).

## 4. `PKU ewaluacja XSub XView TTA bez korekty klasy kieszeniowej.ipynb`
Ewaluacja XSub z test-time augmentation, na wagach `ntu60_CSub` (`_j only`,
strumień bone celowo pominięty). Również w stanie sprzed korekty mapowania
kieszeni, dotyczy tego samego wariantu kontrolnego co notebook powyżej,
z dodatkiem TTA.

## 5. `Ewaluacja PKU-MMD na wszystkich zestawach wag SkateFormer (CSub vs CView, warianty inter, NTU60_120).ipynb`
Rozstrzyga, które wagi NTU RGB+D 60 są kanoniczne dla ewaluacji zero-shot na
PKU-MMD (rozdz. 4.2–4.3 pracy): ewaluuje `PKU_XSub_both.npz` /
`PKU_XView_both.npz` na wszystkich 8 zestawach wag, licząc ensemble j+b tam,
gdzie dostępny jest strumień bone. Potwierdza `ntu60_CSub` jako kanoniczny
(64,65%/64,67%) wobec odrzuconego `ntu60_CView` (60,65%/62,62%). Zawiera
też porównanie z wagami NTU120 i wariantami `inter` (`ntu60_inter`,
`ntu120_inter`) jako dodatkowy kontekst. Źródło wyniku
`ntu60_inter` (81,03%/89,32%) w pracy.

## 6. `Etap II analiza uzupełniająca_ rozbicie dokładności na klasy interakcyjne_pojedyncze, korekta mapowania kieszeniowego (56→24) i fuzja ważona joint_bone.ipynb`
Trzy analizy na gotowych logitach z cache (bez ponownego liczenia):
(1) rozbicie przewagi ~8 p.p. strumienia bone nad joint na klasy
interakcyjne vs. pojedyncze, co pokazuje, że przewaga bone pochodzi z klas
pojedynczych, natomiast bone psuje interakcje; (2) pełne zestawienie per-class dla
wszystkich 51 klas PKU; (3) ważona fuzja logitów α·joint + (1−α)·bone z
walidacją krzyżową między protokołami XSub/XView. Zawiera też potwierdzenie
wpływu korekty mapowania kieszeni (+2,67pp / +2,03pp).

## 7. `PKU etap3 bone fusion detekcja.ipynb`
Etap III. Sprawdza, czy fuzja strumieni joint+bone poprawia detekcję ciągłą
na PKU-MMD. Motywacja: niska precyzja detekcji klas "point at person" i
"pat on back" mimo dobrych wyników klasyfikacji w Etapie II. Dla strumienia
bone feeder ustawia `data_type='b'`, uruchamiając `joint2bone()`. Wyniki
obu strumieni cache'owane osobno (`scores` / `scores_bone`). Metryki
natywne SkateFormera (Accuracy/Top1/Top5 z `main.py`) nieużywane w pracy,
liczy się tylko sanity check i mAP/frame accuracy. Źródło kanonicznych
wyników detekcji joint-only i fuzji joint+bone (Etap III) w pracy.

## 8. `PKU window sweep xsub xview.ipynb`
Etap III. Ablacja szerokości okna W i kroku S dla detekcji ciągłej, w obu
protokołach XSub/XView. Parametry: `MIN_SEG_LEN=8`, NMS IoU=0,3 (tylko ta
sama klasa, per sekwencja), AP z interpolacją all-points (VOC2010+),
`CONF_TH=0` (sweep progu nie poprawiał mAP). Agregacja softmaxu per
klatka, scalanie sąsiednich klatek o tej samej predykcji w segmenty
(min. 8 klatek), NMS temporalny, AP per klasa, mAP.

## 9. `PKU window sweep xview W64 S8.ipynb`
Uzupełnienie do notebooka powyżej: w sweepie zabrakło punktu
Cross-View, W=64, S=8 w tabeli. Ten notebook dolicza brakującą kombinację
i pokazuje pełną tabelę S-sweep (oba protokoły, W=64).

## 10. `Dostrajanie Glowicy NTU Klasy Interakcyjne.ipynb`
Eksperyment negatywny: fine-tuning wyłącznie ostatniej warstwy (`fc`,
0,32% parametrów, ekstraktor cech zamrożony) z 3x wagą straty dla 8 klas
interakcyjnych (NTU idx 49–55, 57). Trenowane na `NTU60_CS.npz`,
ewaluowane po każdej epoce na `PKU_XSub_both.npz` / `PKU_XView_both.npz`
oraz kontrolnie na `NTU60_CS.npz` / `NTU60_CV.npz`. Wniosek: brak istotnej
poprawy na PKU-MMD (±0,4pp, w granicach szumu), nieznaczny spadek na
natywnym NTU (~0,1pp na CS). Ogranicznik transferu leży w ekstraktorze
cech, nie w warstwie klasyfikującej.

## Zależność między notebookami

```
1  Etap I: weryfikacja preprocessingu na NTU (wagi per-protokół)
5  wybór kanonicznych wag (ntu60_CSub) dla zero-shot na PKU
2  Etap II baseline vs both (joint, po korekcie kieszeni)
3, 4 warianty kontrolne bez korekty kieszeni (izolacja wpływu korekty)
6  analiza per-class + fuzja ważona joint/bone (na logitach z 2)
7  Etap III: fuzja joint+bone w detekcji ciągłej
8  Etap III: sweep W/S
9  uzupełnienie brakującego punktu sweepu z 8
10 eksperyment negatywny: fine-tuning głowicy na klasach interakcyjnych
```
