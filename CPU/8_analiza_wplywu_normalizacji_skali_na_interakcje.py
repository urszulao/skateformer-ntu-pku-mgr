#dla NTU i PKU
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

PKU_SKE_DIR = Path('/Users/urszula/skateformer_pku/PKU_Skeleton_Renew')
PKU_LAB_DIR = Path('/Users/urszula/skateformer_pku/Train_Label_PKU_final')
MIN_CONF = 1
NTU_ROOT = Path('/Users/urszula/SkateFormer/data/ntu')
NTU_DENOISED_PKL = NTU_ROOT / 'denoised_data' / 'raw_denoised_joints.pkl'
NTU_LABEL_FILE = NTU_ROOT / 'statistics' / 'label.txt'
NTU_SKES_NAME_FILE = NTU_ROOT / 'statistics' / 'skes_available_name.txt'

OUT_DIR = Path('./dist_ratio_output')

#PKU 29 i 38 to teraz NTU 25
PKU_TO_NTU = {
    1: 34, 2: 3, 3: 2, 4: 32, 5: 21, 6: 9, 7: 39, 8: 0, 9: 4, 10: 1,
    11: 42, 12: 55, 13: 22, 14: 57, 15: 25, 16: 54, 17: 26, 18: 50,
    19: 23, 20: 27, 21: 52, 22: 5, 23: 28, 24: 53, 25: 30, 26: 49,
    27: 51, 28: 19,
    29: 24,
    30: 10, 31: 33, 32: 37, 33: 7, 34: 8, 35: 20, 36: 18, 37: 14,
    38: 24,
    39: 31, 40: 12, 41: 6, 42: 45, 43: 44, 44: 43, 45: 46, 46: 29,
    47: 48, 48: 13, 49: 17, 50: 36, 51: 11,
}

NTU_INTERACTION_NAMES = {
    49: 'punch/slap', 50: 'kicking', 51: 'pushing', 52: 'pat on back',
    53: 'point finger', 54: 'hugging', 55: 'giving object',
    56: 'touch pocket', 57: 'handshaking', 58: 'walking towards',
    59: 'walking apart',
}
NTU_INTERACTION_CLASSES = set(NTU_INTERACTION_NAMES.keys())

def compute_dist_ratios(seg):
    j1_p1 = seg[:, 0:3]
    j21_p1 = seg[:, 60:63]
    j1_p2 = seg[:, 75:78]
    j21_p2 = seg[:, 135:138]

    dist1 = np.sqrt(((j1_p1 - j21_p1) ** 2).sum(axis=1))
    dist2 = np.sqrt(((j1_p2 - j21_p2) ** 2).sum(axis=1))

    person1_present = seg[:, 0:75].sum(axis=1) != 0
    person2_present = seg[:, 75:150].sum(axis=1) != 0
    valid = person1_present & person2_present & (dist1 > 1e-6)

    if valid.sum() == 0:
        return np.array([]), np.array([]), np.array([])
    return dist1[valid], dist2[valid], dist2[valid] / dist1[valid]

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
            start = int(parts[1])
            end = int(parts[2])
            conf = int(parts[3]) if len(parts) > 3 else 2
            instances.append((action, start, end, conf))
    return instances

def run_pku():
    ske_files = sorted(PKU_SKE_DIR.glob('*.txt'))
    if not ske_files:
        print(f'brak plików w {PKU_SKE_DIR}')
        return None

    rows = []
    for ske_path in tqdm(ske_files, desc='PKU-MMD'):
        stem = ske_path.stem
        lab_path = PKU_LAB_DIR / (stem + '.txt')
        if not lab_path.exists():
            continue

        skeleton = parse_skeleton(str(ske_path))
        instances = parse_label(str(lab_path))
        for action, start, end, conf in instances:
            if conf < MIN_CONF:
                continue
            if action not in PKU_TO_NTU:
                continue
            ntu_label = PKU_TO_NTU[action]
            if ntu_label not in NTU_INTERACTION_CLASSES:
                continue

            start_f = max(0, start)
            end_f = min(skeleton.shape[0], end)
            if end_f <= start_f:
                continue

            seg = skeleton[start_f:end_f]
            d1, d2, ratio = compute_dist_ratios(seg)
            for r, a, b in zip(ratio, d1, d2):
                rows.append({
                    'dataset': 'PKU',
                    'sequence': stem,
                    'pku_action': action,
                    'ntu_class_0idx': ntu_label,
                    'ntu_class_name': NTU_INTERACTION_NAMES[ntu_label],
                    'dist1': a,
                    'dist2': b,
                    'ratio': r,
                })

    if not rows:
        print('PKU brak danych z tymi kryteriami')
        return None
    return pd.DataFrame(rows)

def run_ntu():
    if not NTU_DENOISED_PKL.exists() or not NTU_LABEL_FILE.exists():
        print(f'BRAK plików ntu')
        return None

    with open(NTU_DENOISED_PKL, 'rb') as f:
        skes_joints = pickle.load(f)

    labels = np.loadtxt(NTU_LABEL_FILE, dtype=int) - 1  # 0in
    names = np.loadtxt(NTU_SKES_NAME_FILE, dtype=str) if NTU_SKES_NAME_FILE.exists() \
        else [f'seq_{i}' for i in range(len(labels))]

    rows = []
    for idx, seg in enumerate(tqdm(skes_joints, desc='NTU RGB+D 60')):
        label = labels[idx]
        if label not in NTU_INTERACTION_CLASSES:
            continue
        if seg.shape[1] != 150:
            continue

        d1, d2, ratio = compute_dist_ratios(seg)
        seq_name = names[idx] if idx < len(names) else f'seq_{idx}'
        for r, a, b in zip(ratio, d1, d2):
            rows.append({
                'dataset': 'NTU',
                'sequence': seq_name,
                'pku_action': None,
                'ntu_class_0idx': label,
                'ntu_class_name': NTU_INTERACTION_NAMES[label],
                'dist1': a,
                'dist2': b,
                'ratio': r,
            })

    if not rows:
        print('NTU brak danych spełniających kryteria')
        return None
    return pd.DataFrame(rows)

def summarize(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / 'ratio_per_frame.csv', index=False)

    summary = df.groupby(['dataset', 'ntu_class_name']).agg(
        n_frames=('ratio', 'size'),
        ratio_median=('ratio', 'median'),
        ratio_p05=('ratio', lambda x: np.percentile(x, 5)),
        ratio_p95=('ratio', lambda x: np.percentile(x, 95)),
        pct_over_10pct=('ratio', lambda x: (np.abs(x - 1) > 0.10).mean() * 100),
        pct_over_20pct=('ratio', lambda x: (np.abs(x - 1) > 0.20).mean() * 100),
    ).reset_index().sort_values(['dataset', 'ntu_class_name'])

    summary.to_csv(out_dir / 'ratio_summary_per_class.csv', index=False)

    overall = df.groupby('dataset').agg(
        n_frames=('ratio', 'size'),
        ratio_median=('ratio', 'median'),
        ratio_p05=('ratio', lambda x: np.percentile(x, 5)),
        ratio_p95=('ratio', lambda x: np.percentile(x, 95)),
        pct_over_10pct=('ratio', lambda x: (np.abs(x - 1) > 0.10).mean() * 100),
        pct_over_20pct=('ratio', lambda x: (np.abs(x - 1) > 0.20).mean() * 100),
    ).reset_index()
    overall.to_csv(out_dir / 'ratio_summary_overall.csv', index=False)

    print('\nPodsumowanie ogólne')
    print(overall.to_string(index=False))
    print('\nPodsumowanie per klasa (pierwsze 20 wierszy)')
    print(summary.head(20).to_string(index=False))
    print(f'\nZapisano: {out_dir}/ratio_per_frame.csv, '
          f'ratio_summary_per_class.csv, ratio_summary_overall.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ntu', 'pku', 'both'], default='both')
    args = parser.parse_args()

    frames = []
    if args.dataset in ('pku', 'both'):
        df_pku = run_pku()
        if df_pku is not None:
            frames.append(df_pku)
    if args.dataset in ('ntu', 'both'):
        df_ntu = run_ntu()
        if df_ntu is not None:
            frames.append(df_ntu)

    if not frames:
        print('Brak wyników')
    else:
        summarize(pd.concat(frames, ignore_index=True), OUT_DIR)
