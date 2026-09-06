import os
import os.path as osp
import numpy as np

NTU_SKES_PATH = os.path.expanduser('~/SkateFormer/data/ntu/nturgbd_raw/nturgb+d_skeletons')
PKU_LABEL_DIR = os.path.expanduser('~/skateformer_pku/Train_Label_PKU_final')

print(f'NTU_SKES_PATH: {NTU_SKES_PATH}  (istnieje: {os.path.exists(NTU_SKES_PATH)})')
print(f'PKU_LABEL_DIR: {PKU_LABEL_DIR}  (istnieje: {os.path.exists(PKU_LABEL_DIR)})')
print()

if os.path.exists(NTU_SKES_PATH):
    n = len([f for f in os.listdir(NTU_SKES_PATH) if f.endswith('.skeleton')])
    print(f'NTU .skeleton files: {n}')
if os.path.exists(PKU_LABEL_DIR):
    n = len([f for f in os.listdir(PKU_LABEL_DIR) if f.endswith('.txt')])
    print(f'PKU label files: {n}')
print()

#filtrowanie body
noise_len_thres = 11
noise_spr_thres1 = 0.8
noise_spr_thres2 = 0.69754


def read_skeleton_bodies(ske_file):
    with open(ske_file, 'r') as fr:
        str_data = fr.readlines()
    num_frames = int(str_data[0].strip())
    bodies = {}
    current_line = 1
    for f in range(num_frames):
        num_bodies = int(str_data[current_line].strip())
        current_line += 1
        for b in range(num_bodies):
            bodyID = str_data[current_line].strip().split()[0]
            current_line += 1
            num_joints = int(str_data[current_line].strip())
            current_line += 1
            joints = np.zeros((num_joints, 3), dtype=np.float32)
            for j in range(num_joints):
                vals = str_data[current_line].strip().split()
                joints[j, :] = np.array(vals[:3], dtype=np.float32)
                current_line += 1
            bodies.setdefault(bodyID, []).append(joints)
    return bodies


def spread_ratio(points_2d):
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    xr = x.max() - x.min()
    yr = y.max() - y.min()
    if yr == 0:
        return True
    return xr > noise_spr_thres1 * yr


def run_ntu_count():
    total_bodies = 0
    rejected_by_length = 0
    rejected_by_spread = 0
    kept_after_both = 0
    total_files = 0
    errors = 0

    if not os.path.exists(NTU_SKES_PATH):
        print('Brak NTU_SKES_PATH.')
        return None

    ske_files = [f for f in os.listdir(NTU_SKES_PATH) if f.endswith('.skeleton')]
    print(f'Przetwarzanie {len(ske_files)} plikow .skeleton...')

    for idx, fname in enumerate(ske_files):
        total_files += 1
        try:
            bodies = read_skeleton_bodies(osp.join(NTU_SKES_PATH, fname))
        except Exception:
            errors += 1
            continue
        total_bodies += len(bodies)
        for bodyID, frames in bodies.items():
            length = len(frames)
            if length <= noise_len_thres:
                rejected_by_length += 1
                continue
            arr = np.stack(frames)  # (T, 25, 3)
            invalid = sum(spread_ratio(arr[t, :, :2]) for t in range(arr.shape[0]))
            ratio = invalid / arr.shape[0]
            if ratio >= noise_spr_thres2:
                rejected_by_spread += 1
                continue
            kept_after_both += 1

        if (idx + 1) % 5000 == 0:
            print(f'  ... {idx + 1}/{len(ske_files)}')

    print()
    print('NTU')
    print(f'Plikow .skeleton przetworzonych: {total_files} (bledy odczytu: {errors})')
    print(f'Sylwetek (bodyID) lacznie: {total_bodies}')
    print(f'Odrzucone przez (dlugosc <= {noise_len_thres} klatek): '
          f'{rejected_by_length} ({rejected_by_length / total_bodies * 100:.2f}%)')
    print(f'Odrzucone przez (rozrzut >= {noise_spr_thres2}): '
          f'{rejected_by_spread} ({rejected_by_spread / total_bodies * 100:.2f}%)')
    print(f'Pozostalo po obu etapach: {kept_after_both} ({kept_after_both / total_bodies * 100:.2f}%)')
    return total_files


#diagnostyka dla PKU-MMD
MIN_CONF = 1
def parse_label(lab_path):
    instances = []
    with open(lab_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            action, start, end, conf = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            instances.append((action, start, end, conf))
    return instances


def run_pku_count():
    if not os.path.exists(PKU_LABEL_DIR):
        print('Brak PKU_LABEL_DIR.')
        return
    total_instances = 0
    rejected_by_conf = 0
    label_files = [f for f in os.listdir(PKU_LABEL_DIR) if f.endswith('.txt')]
    for fname in label_files:
        instances = parse_label(osp.join(PKU_LABEL_DIR, fname))
        for action, start, end, conf in instances:
            total_instances += 1
            if conf < MIN_CONF:
                rejected_by_conf += 1
    print()
    print('wynik PKU-MMD')
    print(f'Plikow etykiet przetworzonych: {len(label_files)}')
    print(f'Wystapien akcji lacznie: {total_instances}')
    print(f'Odrzuconych przez MIN_CONF = {MIN_CONF}: '
          f'{rejected_by_conf} ({rejected_by_conf / total_instances * 100:.2f}%)')
    print(f'Pozostalo do konwersji: {total_instances - rejected_by_conf} '
          f'({(total_instances - rejected_by_conf) / total_instances * 100:.2f}%)')

if __name__ == '__main__':
    run_ntu_count()
    run_pku_count()
    print()
    print('Rozklad wartosci conf w etykietach PKU')
    if os.path.exists(PKU_LABEL_DIR):
        label_files = [f for f in os.listdir(PKU_LABEL_DIR) if f.endswith('.txt')]
        all_conf = []
        for fname in label_files:
            for action, start, end, conf in parse_label(osp.join(PKU_LABEL_DIR, fname)):
                all_conf.append(conf)
        all_conf = np.array(all_conf)
        unique_vals, counts = np.unique(all_conf, return_counts=True)
        print(f'Unikalne wartosci conf: {len(unique_vals)}')
        for v, c in zip(unique_vals, counts):
            print(f'  conf={v}: {c} wystapien ({c/len(all_conf)*100:.2f}%)')
        if len(unique_vals) <= 2:
            print('Pole conf jest binrane (0/1) - MIN_CONF=1 to filtr, nie prog na skali ciaglej.')
        else:
            print('Pole conf jest ciagle - MIN_CONF=1 to faktyczny prog odcinajacy dolna czesc skali.')

   #rozkład długości segmentów PKU
    print()
    print('Rozklad dlugosci segmentow PKU')
    if os.path.exists(PKU_LABEL_DIR):
        lengths = []
        for fname in label_files:
            for action, start, end, conf in parse_label(osp.join(PKU_LABEL_DIR, fname)):
                if conf < MIN_CONF:
                    continue
                lengths.append(end - start)
        lengths = np.array(lengths)
        MAX_FRAMES = 300
        shorter = (lengths < MAX_FRAMES).sum()
        longer = (lengths > MAX_FRAMES).sum()
        equal = (lengths == MAX_FRAMES).sum()
        print(f'Segmentow lacznie: {len(lengths)}')
        print(f'Srednia dlugosc: {lengths.mean():.1f} klatek, mediana: {np.median(lengths):.1f}, '
              f'min: {lengths.min()}, max: {lengths.max()}')
        print(f'Krotszych niz {MAX_FRAMES} (wymagaja paddingu): {shorter} ({shorter/len(lengths)*100:.2f}%)')
        print(f'Dluzszych niz {MAX_FRAMES} (przycinane): {longer} ({longer/len(lengths)*100:.2f}%)')
        print(f'Rownych {MAX_FRAMES}: {equal} ({equal/len(lengths)*100:.2f}%)')
        # Percentyle, rozklad
        for p in [50, 90, 95, 99]:
            print(f'  percentyl {p}: {np.percentile(lengths, p):.1f} klatek')

    #rozkład klas w zbiorze testowym
    print()
    print('rozkład klas w zbiorze testowym PKU-MMD (XSub, XView) ')
    XSUB_TXT = os.path.expanduser('~/skateformer_pku/cross-subject.txt')
    XVIEW_TXT = os.path.expanduser('~/skateformer_pku/cross-view.txt')
    SKE_DIR = os.path.expanduser('~/skateformer_pku/PKU_Skeleton_Renew')

    def load_split_file(txt_path):
        train_set, test_set = set(), set()
        current = None
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('Training'):
                    current = train_set
                elif line.startswith('Validat'):
                    current = test_set
                else:
                    for token in line.split(','):
                        token = token.strip()
                        if token:
                            current.add(token)
        return train_set, test_set

    for split_name, txt_path in [('XSub', XSUB_TXT), ('XView', XVIEW_TXT)]:
        if not os.path.exists(txt_path):
            print(f' Brak pliku splitu: {txt_path} - pomijam {split_name}')
            continue
        if not os.path.exists(SKE_DIR):
            print(f'Brak katalogu SKE_DIR: {SKE_DIR} - pomijam {split_name}')
            continue
        train_set, test_set = load_split_file(txt_path)
        ske_files = sorted(os.listdir(SKE_DIR))
        per_class_test = {}
        skipped_no_label = 0
        for ske_fname in ske_files:
            stem = ske_fname.replace('.txt', '')
            if stem not in test_set:
                continue
            lab_path = osp.join(PKU_LABEL_DIR, stem + '.txt')
            if not os.path.exists(lab_path):
                skipped_no_label += 1
                continue
            for action, start, end, conf in parse_label(lab_path):
                if conf < MIN_CONF:
                    continue
                per_class_test[action] = per_class_test.get(action, 0) + 1
        total_test_segments = sum(per_class_test.values())
        print(f'\n{split_name}: {total_test_segments} segmentow testowych w {len(per_class_test)} klasach '
              f'(plikow bez etykiety: {skipped_no_label})')
        min_class = min(per_class_test.items(), key=lambda x: x[1]) if per_class_test else None
        max_class = max(per_class_test.items(), key=lambda x: x[1]) if per_class_test else None
        if min_class:
            print(f'  Najmniej liczna klasa PKU {min_class[0]}: {min_class[1]} segmentow')
            print(f'  Najliczniejsza klasa PKU {max_class[0]}: {max_class[1]} segmentow')
        # Klasy interakcji 
        INTERACTION_PKU = [12, 14, 16, 18, 21, 24, 26, 27]  # giving, handshake, hugging, kick, pat, point, punch, push
        int_counts = {k: v for k, v in per_class_test.items() if k in INTERACTION_PKU}
        print(f'  Segmenty testowe klas interakcji (8 klas) lacznie: {sum(int_counts.values())}')
        for k in sorted(int_counts):
            print(f'    PKU {k}: {int_counts[k]} segmentow')
        print(f'  Segmenty testowe dla klas 29 (put in pocket) i 38 (take out pocket): '
              f'{per_class_test.get(29, 0)}, {per_class_test.get(38, 0)}')

    print()

