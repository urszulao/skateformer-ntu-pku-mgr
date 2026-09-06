import sys, os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

SKATEFORMER_DIR = Path('/Users/urszula/skateformer_pku/SkateFormer')
WEIGHTS_PATH    = Path('/Users/urszula/skateformer_pku/SkateFormer') / 'SkateFormer_j.pt'
OUTPUT_DIR      = Path('/Users/urszula/skateformer_pku/output')

DEVICE = torch.device('cpu')

sys.path.insert(0, str(SKATEFORMER_DIR))
sys.path.insert(0, str(SKATEFORMER_DIR / 'torchlight'))

INTERACTION_NTU = [49, 50, 51, 52, 53, 54, 55, 57]
INT_NAMES = {
    49: 'punch/slap', 50: 'kick person', 51: 'push person', 52: 'pat on back',
    53: 'point at person', 54: 'hugging', 55: 'giving object', 57: 'handshaking'
}

def load_model(weights_path):
    from model.SkateFormer import SkateFormer_
    model = SkateFormer_(
        num_classes=60, num_people=2, num_points=24, kernel_size=7,
        num_heads=32, attn_drop=0.5, head_drop=0.0, rel=True,
        drop_path=0.2, type_1_size=[8,8], type_2_size=[8,12],
        type_3_size=[8,8], type_4_size=[8,12], mlp_ratio=4.0, index_t=True
    ).to(DEVICE)
    w = torch.load(weights_path, map_location=DEVICE)
    if 'model' in w:
        w = w['model']
    model.load_state_dict(w, strict=False)
    model.eval()
    return model

def get_index_t(window_size=64, batch_size=1):
    return torch.arange(window_size).unsqueeze(0).repeat(batch_size, 1).to(DEVICE)

def prepare_window(win):
    T = win.shape[0]
    data = win.reshape(T, 2, 25, 3)
    data = data[:, :, :24, :]
    data = data.transpose(3, 0, 2, 1)
    #(1, 3, T, 24, 2)
    data = data[np.newaxis]
    return torch.from_numpy(data).float().to(DEVICE)

def evaluate_with_sliding_window(model, npz_path, batch_size=64):
    data     = np.load(npz_path)
    x_test   = data['x_test']
    y_test   = data['y_test']
    seg_ids  = data['seg_id_test']

    y_true_windows = np.argmax(y_test, axis=1)

    print(f'Okien testowych: {len(x_test)}')
    print(f'Segmentów testowych: {len(np.unique(seg_ids))}')

    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            batch = x_test[i:i+batch_size]
            B = len(batch)
            tensors = []
            for win in batch:
                tensors.append(prepare_window(win))
            batch_tensor = torch.cat(tensors, dim=0)  #(B, 3, 64, 24, 2)
            index_t = get_index_t(window_size=64, batch_size=B)
            logits  = model(batch_tensor, index_t=index_t)  #(B, 60)
            probs   = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            if i % (batch_size * 10) == 0:
                print(f'  Inference: {i}/{len(x_test)}', end='\r')

    all_probs = np.concatenate(all_probs, axis=0)
    print(f'  Inference zakończony: {len(x_test)}/{len(x_test)}')

    #softmax
    unique_segs = np.unique(seg_ids)
    y_pred_seg  = []
    y_true_seg  = []

    for sid in unique_segs:
        mask       = (seg_ids == sid)
        seg_probs  = all_probs[mask].mean(axis=0)
        y_pred_seg.append(np.argmax(seg_probs))
        y_true_seg.append(y_true_windows[mask][0])

    y_pred = np.array(y_pred_seg)
    y_true = np.array(y_true_seg)

    overall = (y_pred == y_true).mean() * 100

    mask_int = np.isin(y_true, INTERACTION_NTU)
    int_acc  = (y_pred[mask_int] == y_true[mask_int]).mean() * 100 if mask_int.sum() > 0 else 0.0

    return y_pred, y_true, overall, int_acc


def print_per_class(y_pred, y_true, label=''):
    print(f'\n Per-class interakcje {label} ')
    print(f'{"Akcja":<22} {"Correct":>8} {"Total":>7} {"Acc":>7}')
    print('-' * 45)
    for ntu_idx in sorted(INTERACTION_NTU):
        mask    = (y_true == ntu_idx)
        total   = mask.sum()
        if total == 0: continue
        correct = (y_pred[mask] == ntu_idx).sum()
        print(f'{INT_NAMES[ntu_idx]:<22} {correct:>8} {total:>7} {correct/total*100:>6.1f}%')


if __name__ == '__main__':
    os.chdir(str(SKATEFORMER_DIR))
    model = load_model(str(WEIGHTS_PATH))

    #dla porównania
    BASELINE = {
        'XSub': (62.61, 84.10),
        'XView': (62.79, 89.84),
    }

    print(f'\n{"Split":<8} {"Overall (baseline)":>20} {"Overall (sliding)":>19} {"Δ":>6}  '
          f'{"Int (baseline)":>16} {"Int (sliding)":>15} {"Δ":>6}')
    print('-' * 95)

    for split, npz_name in [('XSub', 'PKU_XSub_sliding.npz'), ('XView', 'PKU_XView_sliding.npz')]:
        npz_path = OUTPUT_DIR / npz_name
        if not npz_path.exists():
            print(f'Brak pliku: {npz_path}')
            continue

        print(f'\n{split}:')
        y_pred, y_true, overall, int_acc = evaluate_with_sliding_window(model, str(npz_path))

        base_ov, base_int = BASELINE[split]
        print(f'{split:<8} {base_ov:>19.2f}% {overall:>18.2f}% {overall-base_ov:>+5.2f}pp  '
              f'{base_int:>15.2f}% {int_acc:>14.2f}% {int_acc-base_int:>+5.2f}pp')

        print_per_class(y_pred, y_true, label=split)
