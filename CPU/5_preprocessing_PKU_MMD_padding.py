#pad 5, pad 10, pad 15
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

SKE_DIR   = Path('/Users/urszula/skateformer_pku/PKU_Skeleton_Renew')
LAB_DIR   = Path('/Users/urszula/skateformer_pku/Train_Label_PKU_final')
OUT_DIR   = Path('/Users/urszula/skateformer_pku/output')
XSUB_TXT  = Path('/Users/urszula/skateformer_pku/cross-subject.txt')
XVIEW_TXT = Path('/Users/urszula/skateformer_pku/cross-view.txt')
MAX_FRAMES = 300
MIN_CONF   = 1

PKU_TO_NTU = {
    1:  34,  # bow  nod head/bow (NTU 35)
    2:  3,   # brushing hair  brush hair (NTU 4)
    3:  2,   # brushing teeth  brush teeth (NTU 3)
    4:  32,  # check time from watch  check time (NTU 33)
    5:  21,  # cheer up  cheer up (NTU 22)
    6:  9,   # clapping  clapping (NTU 10)
    7:  39,  # cross hands in front  cross hands (NTU 40)
    8:  0,   # drink water  drink water (NTU 1)
    9:  4,   # drop  drop (NTU 5)
    10: 1,   # eat meal  eat meal (NTU 2)
    11: 42,  # falling  falling down (NTU 43)
    12: 55,  # giving something  giving object (NTU 56)
    13: 22,  # hand waving  hand waving (NTU 23)
    14: 57,  # handshaking  shaking hands (NTU 58)
    15: 25,  # hopping  hopping (NTU 26)
    16: 54,  # hugging  hugging (NTU 55)
    17: 26,  # jump up  jump up (NTU 27)
    18: 50,  # kicking person  kicking person (NTU 51)
    19: 23,  # kicking something  kicking something (NTU 24)
    20: 27,  # phone call  phone call (NTU 28)
    21: 52,  # pat on back  pat on back (NTU 53)
    22: 5,   # pickup  pick up (NTU 6)
    23: 28,  # playing with phone  playing with phone (NTU 29)
    24: 53,  # point finger at person  point finger (NTU 54)
    25: 30,  # pointing to something  point to something (NTU 31)
    26: 49,  # punching/slapping  punch/slap (NTU 50)
    27: 51,  # pushing person  pushing person (NTU 52)
    28: 19,  # put on hat  put on hat (NTU 20)
    29: 56,  # put in pocket  touch pocket (NTU 57)
    30: 10,  # reading  reading (NTU 11)
    31: 33,  # rub two hands  rub two hands (NTU 34)
    32: 37,  # salute  salute (NTU 38)
    33: 7,   # sitting down  sit down (NTU 8)
    34: 8,   # standing up  stand up (NTU 9)
    35: 20,  # take off hat  take off hat (NTU 21)
    36: 18,  # take off glasses  take off glasses (NTU 19)
    37: 14,  # take off jacket  take off jacket (NTU 15)
    38: 24,  # take out from pocket  reach into pocket (NTU 25)
    39: 31,  # taking selfie  taking selfie (NTU 32)
    40: 12,  # tear up paper  tear up paper (NTU 13)
    41: 6,   # throw  throw (NTU 7)
    42: 45,  # touch back  back pain (NTU 46)
    43: 44,  # touch chest  stomachache (NTU 45)
    44: 43,  # touch head  headache (NTU 44)
    45: 46,  # touch neck  neck pain (NTU 47)
    46: 29,  # typing  typing (NTU 30)
    47: 48,  # use fan  fan self (NTU 49)
    48: 13,  # wear jacket  put on jacket (NTU 14)
    49: 17,  # wear glasses  put on glasses (NTU 18)
    50: 36,  # wipe face  wipe face (NTU 37)
    51: 11,  # writing  writing (NTU 12)
}

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

def normalize_rotation(seg):
    shoulder_l = seg[:, 12:15].copy()
    shoulder_r = seg[:, 24:27].copy()

    dx = shoulder_r[:, 0] - shoulder_l[:, 0]
    dz = shoulder_r[:, 2] - shoulder_l[:, 2]

    angles = np.arctan2(dz, dx)

    valid = seg[:, :75].sum(axis=1) != 0
    if valid.sum() == 0:
        return seg

    median_angle = np.median(angles[valid])
    cos_a = np.cos(-median_angle)
    sin_a = np.sin(-median_angle)

    result = seg.copy()
    for j in range(50):
        x_col = j * 3
        z_col = j * 3 + 2
        x = result[:, x_col].copy()
        z = result[:, z_col].copy()
        result[:, x_col] = cos_a * x - sin_a * z
        result[:, z_col] = sin_a * x + cos_a * z
    return result


def smooth_skeleton(seg, sigma=1.5):
    from scipy.ndimage import gaussian_filter1d
    result = seg.copy()
    for b in range(2):
        start = b * 75
        end   = start + 75
        present = seg[:, start:end].sum(axis=1) != 0
        if present.sum() < 3:
            continue
        for c in range(start, end):
            col = result[:, c].copy()
            col = gaussian_filter1d(col.astype(np.float64), sigma=sigma)
            col[~present] = 0
            result[:, c] = col.astype(np.float32)
    return result

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
            parts  = line.split(',')
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
    origin   = skeleton[origin_frame, 3:6].copy()
    skeleton = skeleton - np.tile(origin, 50)
    missing_1 = np.where(skeleton[:, :75].sum(axis=1) == 0)[0]
    missing_2 = np.where(skeleton[:, 75:].sum(axis=1) == 0)[0]
    if len(missing_1) > 0:
        skeleton[missing_1, :75] = 0
    if len(missing_2) > 0:
        skeleton[missing_2, 75:] = 0
    return skeleton


def pad_or_crop(skeleton, max_frames):
    T = skeleton.shape[0]
    if T >= max_frames:
        start = (T - max_frames) // 2
        return skeleton[start:start + max_frames]
    pad = np.zeros((max_frames - T, skeleton.shape[1]), dtype=np.float32)
    return np.concatenate([skeleton, pad], axis=0)

def one_hot(ntu_label_0indexed, num_classes=60):
    v = np.zeros(num_classes, dtype=np.float32)
    v[ntu_label_0indexed] = 1.0
    return v

def process_split(ske_files, train_set, test_set, mode, pad=0, label='XSub'):
    x = {'train': [], 'test': []}
    y = {'train': [], 'test': []}
    skipped = 0

    for ske_path in tqdm(ske_files, desc=f'Konwersja {label} [{mode}]'):
        stem     = ske_path.stem
        lab_path = LAB_DIR / (stem + '.txt')

        if not lab_path.exists():
            tqdm.write(f'  BRAK etykiety: {stem}')
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

        if mode in ('interpolation', 'both', 'both_smooth15', 'both_smooth30'):
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
            start_p = max(0, start - pad)
            end_p   = min(skeleton.shape[0], end + pad)
            seg = skeleton[start_p:end_p].copy()

            seg = normalize(seg)
            if mode in ('rotation',):
                seg = normalize_rotation(seg)
            if mode in ('scale', 'both', 'both_smooth15', 'both_smooth30'):
                seg = normalize_scale(seg)

            if mode == 'smooth_15':
                seg = smooth_skeleton(seg, sigma=1.5)
            elif mode == 'smooth_30':
                seg = smooth_skeleton(seg, sigma=3.0)
            elif mode == 'both_smooth15':
                seg = smooth_skeleton(seg, sigma=1.5)
            elif mode == 'both_smooth30':
                seg = smooth_skeleton(seg, sigma=3.0)

            seg = pad_or_crop(seg, MAX_FRAMES)

            ntu_label = PKU_TO_NTU[action]
            x[split].append(seg)
            y[split].append(one_hot(ntu_label))

    print(f'Pominięto plików: {skipped}')
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['baseline', 'interpolation', 'scale', 'rotation',
                                 'smooth_15', 'smooth_30',
                                 'both', 'both_smooth15', 'both_smooth30'],
                        help='Tryb preprocessingu')
    parser.add_argument('--pad', type=int, default=0,
                        help='Rozszerz każdy segment o N klatek z każdej strony (domyślnie 0)')
    args = parser.parse_args()

    print(f'Tryb: {args.mode}  |  pad: {args.pad} klatek')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ske_files = sorted(SKE_DIR.glob('*.txt'))
    print(f'Znaleziono {len(ske_files)} plików skeleton')

    for benchmark, txt_path, out_stem in [
        ('XSub',  XSUB_TXT,  'PKU_XSub'),
        ('XView', XVIEW_TXT, 'PKU_XView'),
    ]:
        print(f'\n=== {benchmark} ===')
        train_set, test_set = load_split_file(txt_path)
        print(f'  train: {len(train_set)} plików, test: {len(test_set)} plików')

        x, y = process_split(ske_files, train_set, test_set, mode=args.mode, pad=args.pad, label=benchmark)

        x_train = np.array(x['train'], dtype=np.float32)
        y_train = np.array(y['train'], dtype=np.float32)
        x_test  = np.array(x['test'],  dtype=np.float32)
        y_test  = np.array(y['test'],  dtype=np.float32)

        print(f'  train: x={x_train.shape}, y={y_train.shape}')
        print(f'  test:  x={x_test.shape},  y={y_test.shape}')

        mode_str = args.mode
        pad_str  = f'_pad{args.pad}' if args.pad > 0 else ''
        out_name = f'{out_stem}_{mode_str}{pad_str}.npz'
        out_path = OUT_DIR / out_name
        np.savez_compressed(str(out_path),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        print(f'  Zapisano: {out_path}')

    print('\nGotowe!')
