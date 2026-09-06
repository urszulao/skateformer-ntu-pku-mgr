import os
import os.path as osp
import numpy as np

PKU_LABEL_DIR = os.path.expanduser('~/skateformer_pku/Train_Label_PKU_final')
SKE_DIR       = os.path.expanduser('~/skateformer_pku/PKU_Skeleton_Renew')
MIN_CONF      = 1
MAX_FRAMES    = 300


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


def count_skeleton_frames(ske_path):
    # jedna linia odpowiada jednej klatce
    with open(ske_path, 'r') as f:
        return sum(1 for _ in f)


if __name__ == '__main__':
    if not os.path.exists(PKU_LABEL_DIR) or not os.path.exists(SKE_DIR):
        print('Sprawdz sciezki PKU_LABEL_DIR / SKE_DIR.')
        raise SystemExit

    label_files = [f for f in os.listdir(PKU_LABEL_DIR) if f.endswith('.txt')]

    lengths = []
    dropped_by_conf = 0
    dropped_by_bad_range = 0
    total_instances = 0
    frame_cache = {}

    for fname in label_files:
        stem = fname[:-4]
        ske_path = osp.join(SKE_DIR, stem + '.txt')
        if not os.path.exists(ske_path):
            continue
        if stem not in frame_cache:
            frame_cache[stem] = count_skeleton_frames(ske_path)
        num_frames = frame_cache[stem]

        for action, start, end, conf in parse_label(osp.join(PKU_LABEL_DIR, fname)):
            total_instances += 1
            if conf < MIN_CONF:
                dropped_by_conf += 1
                continue
            s = max(0, start)
            e = min(num_frames, end)
            if e <= s:
                dropped_by_bad_range += 1
                continue
            lengths.append(e - s)

    lengths = np.array(lengths)
    print(f'Wystapien lacznie: {total_instances}')
    print(f'Odrzuconych przez MIN_CONF: {dropped_by_conf}')
    print(f'Odrzuconych przez zly zakres (end<=start po przycieciu do dl. pliku): {dropped_by_bad_range}')
    print(f'Segmentow poprawnych (uzytych do konwersji): {len(lengths)}')
    print()
    print(f'Srednia dlugosc: {lengths.mean():.1f} klatek, mediana: {np.median(lengths):.1f}, '
          f'min: {lengths.min()}, max: {lengths.max()}')
    shorter = (lengths < MAX_FRAMES).sum()
    longer = (lengths > MAX_FRAMES).sum()
    equal = (lengths == MAX_FRAMES).sum()
    print(f'Krotszych niz {MAX_FRAMES} (padding): {shorter} ({shorter/len(lengths)*100:.2f}%)')
    print(f'Dluzszych niz {MAX_FRAMES} (przycinanie): {longer} ({longer/len(lengths)*100:.2f}%)')
    print(f'Rownych {MAX_FRAMES}: {equal} ({equal/len(lengths)*100:.2f}%)')
    for p in [50, 90, 95, 99]:
        print(f'  percentyl {p}: {np.percentile(lengths, p):.1f} klatek')
    print()