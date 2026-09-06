#Cross-View W=64 S=8
import json
import os
import tempfile
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE       = Path('/Users/urszula/skateformer_pku')
SKE_DIR    = BASE / 'PKU_Skeleton_Renew'
LAB_DIR    = BASE / 'Train_Label_PKU_final'
XVIEW_TXT  = BASE / 'cross-view.txt'
MIN_CONF   = 1

SPLIT     = 'xview'
SPLIT_TXT = XVIEW_TXT
OUT_DIR   = BASE / 'output/sweep_xview'

W, S = 64, 8

SKIP_EXISTING = True

PKU_TO_NTU = {
    1: 34, 2: 3, 3: 2, 4: 32, 5: 21, 6: 9, 7: 39, 8: 0, 9: 4, 10: 1,
    11: 42, 12: 55, 13: 22, 14: 57, 15: 25, 16: 54, 17: 26, 18: 50,
    19: 23, 20: 27, 21: 52, 22: 5, 23: 28, 24: 53, 25: 30, 26: 49,
    27: 51, 28: 19, 29: 56, 30: 10, 31: 33, 32: 37, 33: 7, 34: 8,
    35: 20, 36: 18, 37: 14, 38: 24, 39: 31, 40: 12, 41: 6, 42: 45,
    43: 44, 44: 43, 45: 46, 46: 29, 47: 48, 48: 13, 49: 17, 50: 36,
    51: 11,
}

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


def parse_skeleton(filepath):
    frames = []
    with open(filepath, 'r') as f:
        for line in f:
            vals = list(map(float, line.split()))
            if len(vals) < 150:
                continue
            frames.append(vals[:150])
    if not frames:
        return np.zeros((1, 150), dtype=np.float32)
    return np.array(frames, dtype=np.float32)


def parse_label(filepath):
    instances = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            action = int(parts[0])
            start  = int(parts[1])
            end    = int(parts[2])
            conf   = int(parts[3]) if len(parts) > 3 else 2
            instances.append((action, start, end, conf))
    return instances


def normalize(skeleton):
    origin_frame = 0
    for t in range(skeleton.shape[0]):
        if np.any(skeleton[t, :75] != 0):
            origin_frame = t
            break
    origin = skeleton[origin_frame, 3:6].copy()
    skeleton = skeleton - np.tile(origin, 50)
    missing_1 = np.where(skeleton[:, :75].sum(axis=1) == 0)[0]
    missing_2 = np.where(skeleton[:, 75:].sum(axis=1) == 0)[0]
    if len(missing_1) > 0:
        skeleton[missing_1, :75] = 0
    if len(missing_2) > 0:
        skeleton[missing_2, 75:] = 0
    return skeleton

def one_hot(ntu_label_0indexed, num_classes=60):
    v = np.zeros(num_classes, dtype=np.float32)
    if 0 <= ntu_label_0indexed < num_classes:
        v[ntu_label_0indexed] = 1.0
    return v

def best_overlap_label(instances, win_start, win_end, min_overlap_frac=0.5):
    win_len = win_end - win_start
    best_label, best_frac = -1, 0.0
    for ntu_label, i_start, i_end in instances:
        overlap = max(0, min(win_end, i_end) - max(win_start, i_start))
        frac = overlap / win_len if win_len > 0 else 0
        if frac > best_frac:
            best_frac, best_label = frac, ntu_label
    return best_label if best_frac >= min_overlap_frac else -1
 
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'PKU_windows_test_W{W}_S{S}.npz'

    if SKIP_EXISTING and out_path.exists():
        print(f'{out_path.name} już istnieje')
        return

    _, test_set = load_split_file(SPLIT_TXT)
    ske_files = sorted(p for p in SKE_DIR.glob('*.txt') if p.stem in test_set)
    print(f'Protokół: {SPLIT.upper()}  ({SPLIT_TXT.name})')
    print(f'Katalog wyjściowy: {OUT_DIR}')
    print(f'Sekwencji testowych: {len(ske_files)}')
    print(f'Config: W={W}, S={S}\n')

    sequences = {}
    gt = {}
    seq_lengths = {}

    for ske_path in tqdm(ske_files, desc='Wczytywanie sekwencji'):
        stem = ske_path.stem
        lab_path = LAB_DIR / (stem + '.txt')
        if not lab_path.exists():
            print(f'brak etykiety: {stem}, pomijam')
            continue
        skeleton = parse_skeleton(str(ske_path))
        instances = parse_label(str(lab_path))
        sequences[stem] = skeleton
        seq_lengths[stem] = int(skeleton.shape[0])
        gt[stem] = [
            [PKU_TO_NTU[a], s, e]
            for a, s, e, c in instances
            if c >= MIN_CONF and a in PKU_TO_NTU
        ]

    with open(OUT_DIR / 'ground_truth.json', 'w') as f:
        json.dump(gt, f)
    with open(OUT_DIR / 'seq_lengths.json', 'w') as f:
        json.dump(seq_lengths, f)
    print(f'Zapisano ground_truth.json i seq_lengths.json ({len(gt)} sekwencji)\n')

    n_total = 0
    for seq_id, skeleton in sequences.items():
        T = skeleton.shape[0]
        if T < W:
            continue
        starts_c = list(range(0, T - W + 1, S))
        if starts_c and starts_c[-1] != T - W:
            starts_c.append(T - W)
        for st in starts_c:
            if np.any(skeleton[st:st + W]):
                n_total += 1

    gb = n_total * W * 150 * 4 / 1024**3
    print(f'W={W} S={S}: {n_total} okien, ~{gb:.2f} GB')

    tmp_dir = tempfile.mkdtemp(prefix='pku_window_gen_')
    memmap_path = os.path.join(tmp_dir, f'x_test_W{W}_S{S}.dat')
    print(f'Plik tymczasowy memmap: {memmap_path}')
    x_test = np.memmap(memmap_path, dtype=np.float32, mode='w+',
                        shape=(n_total, W, 150))

    y_test = np.zeros((n_total, 60), dtype=np.float32)
    y_ref  = np.empty(n_total, dtype=np.int32)
    seqid_list, start_list, end_list = [], [], []
    n_skipped_empty = 0
    k = 0

    for seq_id, skeleton in tqdm(sequences.items(), desc=f'W={W} S={S}'):
        T = skeleton.shape[0]
        if T < W:
            continue
        starts = list(range(0, T - W + 1, S))
        if starts and starts[-1] != T - W:
            starts.append(T - W)

        for start in starts:
            end = start + W
            window = skeleton[start:end].copy()

            if not np.any(window):
                n_skipped_empty += 1
                continue

            ref_label = best_overlap_label(gt[seq_id], start, end)

            x_test[k] = normalize(window)
            y_test[k] = one_hot(ref_label if ref_label >= 0 else 0)
            y_ref[k]  = ref_label
            seqid_list.append(seq_id)
            start_list.append(start)
            end_list.append(end)
            k += 1

    assert k == n_total, f'Niezgodnosc: wypelniono {k}, {n_total}'
    x_test.flush()

    np.savez(
        str(out_path),
        x_test=x_test, y_test=y_test, y_ref=y_ref,
        seq_id=np.array(seqid_list, dtype='<U16'),
        start=np.array(start_list, dtype=np.int32),
        end=np.array(end_list, dtype=np.int32),
    )

    overlap_pct = (y_ref >= 0).mean() * 100
    print(f'W={W} S={S}: {k} okien -> {out_path.name}  '
          f'(kształt: {x_test.shape}, okna z dominującą akcją: {overlap_pct:.1f}%, '
          f'pominięto pustych: {n_skipped_empty})')

    del x_test, y_test, y_ref
    try:
        os.remove(memmap_path)
        os.rmdir(tmp_dir)
    except OSError:
        print(f'(problem z plikiem tymczasowym {memmap_path}')


if __name__ == '__main__':
    main()
