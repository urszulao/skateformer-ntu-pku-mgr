#64 z krokiem 16 klatek
import numpy as np
from pathlib import Path
from tqdm import tqdm

SKE_DIR   = Path('/Users/urszula/skateformer_pku/PKU_Skeleton_Renew')
LAB_DIR   = Path('/Users/urszula/skateformer_pku/Train_Label_PKU_final')
OUT_DIR   = Path('/Users/urszula/skateformer_pku/output')
XSUB_TXT  = Path('/Users/urszula/skateformer_pku/cross-subject.txt')
XVIEW_TXT = Path('/Users/urszula/skateformer_pku/cross-view.txt')
MIN_CONF  = 1

WINDOW_SIZE = 64   #jak w SkateFormer feeder
STRIDE      = 16

PKU_TO_NTU = {
    1:34, 2:3, 3:2, 4:32, 5:21, 6:9, 7:39, 8:0, 9:4, 10:1,
    11:42, 12:55, 13:22, 14:57, 15:25, 16:54, 17:26, 18:50, 19:23, 20:27,
    21:52, 22:5, 23:28, 24:53, 25:30, 26:49, 27:51, 28:19, 29:56, 30:10,
    31:33, 32:37, 33:7, 34:8, 35:20, 36:18, 37:14, 38:24, 39:31, 40:12,
    41:6, 42:45, 43:44, 44:43, 45:46, 46:29, 47:48, 48:13, 49:17, 50:36, 51:11
}


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
            parts  = line.split(',')
            action = int(parts[0])
            start  = int(parts[1])
            end    = int(parts[2])
            conf   = int(parts[3]) if len(parts) > 3 else 2
            instances.append((action, start, end, conf))
    return instances


def interpolate_missing_frames(ske_joints):
    result = ske_joints.copy()
    for b in range(2):
        start = b * 75
        end   = start + 75
        actor = result[:, start:end]
        missing_mask = (actor.sum(axis=1) == 0)
        valid_mask   = ~missing_mask
        if valid_mask.sum() == 0 or missing_mask.sum() == 0:
            continue
        valid_indices = np.where(valid_mask)[0]
        for f in np.where(missing_mask)[0]:
            before = valid_indices[valid_indices < f]
            after  = valid_indices[valid_indices > f]
            if len(before) == 0:
                result[f, start:end] = actor[after[0]]
            elif len(after) == 0:
                result[f, start:end] = actor[before[-1]]
            else:
                f0, f1 = before[-1], after[0]
                alpha  = (f - f0) / (f1 - f0)
                result[f, start:end] = (1 - alpha) * actor[f0] + alpha * actor[f1]
    return result


def normalize(skeleton):
    origin_frame = 0
    for t in range(skeleton.shape[0]):
        if np.any(skeleton[t, :75] != 0):
            origin_frame = t
            break
    origin   = skeleton[origin_frame, 3:6].copy()
    skeleton = skeleton - np.tile(origin, 50)
    missing_1 = np.where(skeleton[:, :75].sum(axis=1) == 0)[0]
    missing_2 = np.where(skeleton[:, 75:].sum(axis=1) == 0)[0]
    if len(missing_1) > 0:
        skeleton[missing_1, :75] = 0
    if len(missing_2) > 0:
        skeleton[missing_2, 75:] = 0
    return skeleton


def normalize_scale(seg):
    j1  = seg[:, 0:3]
    j21 = seg[:, 60:63]
    spine_lengths = np.sqrt(((j1 - j21) ** 2).sum(axis=1))
    valid = spine_lengths > 1e-6
    if valid.sum() == 0:
        return seg
    median_spine = np.median(spine_lengths[valid])
    if median_spine < 1e-6:
        return seg
    return seg / median_spine


def extract_windows(seg, window_size, stride):
    T = seg.shape[0]
    windows = []

    if T < window_size:
        pad = np.zeros((window_size - T, seg.shape[1]), dtype=np.float32)
        windows.append(np.concatenate([seg, pad], axis=0))
        return windows

    starts = list(range(0, T - window_size + 1, stride))

    if starts[-1] + window_size < T:
        starts.append(T - window_size)

    for s in starts:
        windows.append(seg[s:s + window_size].copy())

    return windows


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

def one_hot(ntu_label_0indexed, num_classes=60):
    v = np.zeros(num_classes, dtype=np.float32)
    v[ntu_label_0indexed] = 1.0
    return v

def process_split(ske_files, train_set, test_set, label='XSub'):
    x          = {'train': [], 'test': []}
    y          = {'train': [], 'test': []}
    seg_ids    = {'train': [], 'test': []}
    seg_id_counter = 0
    skipped = 0

    for ske_path in tqdm(ske_files, desc=f'Konwersja {label}'):
        stem     = ske_path.stem
        lab_path = LAB_DIR / (stem + '.txt')

        if not lab_path.exists():
            skipped += 1
            continue

        if stem in train_set:
            split = 'train'
        elif stem in test_set:
            split = 'test'
        else:
            skipped += 1
            continue

        skeleton  = parse_skeleton(str(ske_path))
        instances = parse_label(str(lab_path))
        skeleton = interpolate_missing_frames(skeleton)

        for action, start, end, conf in instances:
            if conf < MIN_CONF:
                continue
            if action not in PKU_TO_NTU:
                continue
            start = max(0, start)
            end   = min(skeleton.shape[0], end)
            if end <= start:
                continue

            seg = skeleton[start:end].copy()
            seg = normalize(seg)
            seg = normalize_scale(seg)

            windows = extract_windows(seg, WINDOW_SIZE, STRIDE)

            ntu_label = PKU_TO_NTU[action]

            for win in windows:
                x[split].append(win)
                y[split].append(one_hot(ntu_label))
                seg_ids[split].append(seg_id_counter)

            seg_id_counter += 1

    print(f'  Pominięto: {skipped}  |  Segmentów: {seg_id_counter}')
    return x, y, seg_ids

if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ske_files = sorted(SKE_DIR.glob('*.txt'))
    print(f'Znaleziono {len(ske_files)} plików skeleton')
    print(f'Window size: {WINDOW_SIZE}  Stride: {STRIDE}')

    for benchmark, txt_path, out_stem in [
        ('XSub',  XSUB_TXT,  'PKU_XSub'),
        ('XView', XVIEW_TXT, 'PKU_XView'),
    ]:
        print(f'\n=== {benchmark} ===')
        train_set, test_set = load_split_file(txt_path)

        x, y, seg_ids = process_split(ske_files, train_set, test_set, label=benchmark)

        x_train    = np.array(x['train'],       dtype=np.float32)
        y_train    = np.array(y['train'],       dtype=np.float32)
        x_test     = np.array(x['test'],        dtype=np.float32)
        y_test     = np.array(y['test'],        dtype=np.float32)
        sid_train  = np.array(seg_ids['train'], dtype=np.int32)
        sid_test   = np.array(seg_ids['test'],  dtype=np.int32)

        print(f'  train: x={x_train.shape}  okien na segment: {len(x_train)/max(1,len(np.unique(sid_train))):.1f} avg')
        print(f'  test:  x={x_test.shape}   okien na segment: {len(x_test)/max(1,len(np.unique(sid_test))):.1f} avg')

        out_path = OUT_DIR / f'{out_stem}_sliding.npz'
        np.savez_compressed(
            str(out_path),
            x_train=x_train, y_train=y_train,
            x_test=x_test,   y_test=y_test,
            seg_id_train=sid_train, seg_id_test=sid_test
        )
        print(f'  Zapisano: {out_path}')
