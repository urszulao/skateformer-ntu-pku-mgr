Oficjalne skrypty NTU RGB+D (kod źródłowy, niemodyfikowany)

Poniższe trzy skrypty pochodzą z oficjalnego pipeline'u preprocessingu NTU RGB+D (Microsoft Corporation, licencja MIT) i nie są mojego autorstwa. Stanowią punkt wyjścia dla moich własnych skryptów przetwarzania z folderu cpu, w szczególności 1_preprocessing_NTU_RGBD_warianty_przetwarzania.py, który jest moją przeróbką seq_transformation.py o warianty baseline, interpolacja, skala i both. Umieszczone tu wyłącznie dla kompletności i odtwarzalności pipeline'u.

get_raw_skes_data.py

Wczytuje surowe pliki .skeleton z NTU RGB+D i zapisuje dane wszystkich wykrytych ciał do pliku raw_skes_data.pkl. Dla każdego ciała zapisywane są cztery pola: joints (współrzędne 3D stawów), colors (odpowiadające im współrzędne 2D na obrazie RGB, przydatne np. do wizualizacji na wideo), interval (indeksy klatek, w których dane ciało występuje) i motion (miara ilości ruchu, liczona tylko gdy w sekwencji jest więcej niż jedno ciało). Główne funkcje to get_raw_bodies_data i get_raw_skes_data.

get_raw_denoised_data.py

Odszumianie danych szkieletowych. Usuwa błędne lub nadmiarowe ciała na podstawie długości sekwencji, rozrzutu przestrzennego i ilości ruchu, wybiera maksymalnie dwóch aktorów na sekwencję. Zawiera progi noise_len_thres, noise_spr_thres1, noise_spr_thres2, noise_mot_thres_lo i noise_mot_thres_hi, te same progi wykorzystane diagnostycznie we własnym skrypcie 6_diagnostyka_progow_i_statystyk_NTU_PKU_MMD.py. Główne funkcje to denoising_by_length, get_valid_frames_by_spread, denoising_by_spread, denoising_by_motion, denoising_bodies_data, get_one_actor_points, remove_missing_frames, get_bodies_info, get_two_actors_points i get_raw_denoised_data.

seq_transformation.py

Końcowy etap przetwarzania. Usuwa klatki z wartościami NaN (remove_nan_frames), przesuwa całą sekwencję tak, by pierwszy staw kręgosłupa pierwszego aktora był punktem odniesienia (seq_translation), następnie dla każdej klatki osobno przesuwa układ względem środka kręgosłupa i normalizuje skalę przez odległość między dwoma stawami kręgosłupa (frame_translation), wyrównuje długość wszystkich sekwencji do stałej liczby klatek dopełniając zerami (align_frames), generuje etykiety one-hot i dzieli dane na zbiór treningowy i testowy według protokołów Cross-Subject i Cross-View (split_dataset, get_indices). Funkcja split_train_val do wydzielenia zbioru walidacyjnego jest w kodzie zdefiniowana, ale w obecnej wersji split_dataset nieużywana (zakomentowana).

Kolejność przetwarzania

Dane przechodzą przez te trzy skrypty w kolejności, w jakiej są tu opisane: najpierw get_raw_skes_data.py wczytuje surowe pliki i zapisuje raw_skes_data.pkl, potem get_raw_denoised_data.py odszumia dane i zapisuje raw_denoised_joints.pkl, na końcu seq_transformation.py przygotowuje dane wejściowe dla SkateFormer na NTU RGB+D 60 w Etapie I. Moja przeróbka tego ostatniego kroku to skrypt 1_preprocessing_NTU_RGBD_warianty_przetwarzania.py w folderze cpu.
