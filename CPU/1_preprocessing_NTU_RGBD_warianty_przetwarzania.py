#baseline, interpolation, scale, both
import os
import os.path as osp
import numpy as np
import pickle
import logging
import argparse
from sklearn.model_selection import train_test_split

root_path = './'
stat_path = osp.join(root_path, 'statistics')
setup_file = osp.join(stat_path, 'setup.txt')
camera_file = osp.join(stat_path, 'camera.txt')
performer_file = osp.join(stat_path, 'performer.txt')
replication_file = osp.join(stat_path, 'replication.txt')
label_file = osp.join(stat_path, 'label.txt')
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')

denoised_path = osp.join(root_path, 'denoised_data')
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(denoised_path, 'frames_cnt.txt')

save_path = './'


def interpolate_missing_frames(ske_joints):
    num_frames = ske_joints.shape[0]
    result = ske_joints.copy()

    num_bodies = 1 if ske_joints.shape[1] == 75 else 2

    for b in range(num_bodies):
        start = b * 75
        end = start + 75
        actor = result[:, start:end]

        missing_mask = (actor.sum(axis=1) == 0)
        valid_mask = ~missing_mask

        if valid_mask.sum() == 0 or missing_mask.sum() == 0:
            continue

        valid_indices = np.where(valid_mask)[0]

        for f in np.where(missing_mask)[0]:
            before = valid_indices[valid_indices < f]
            after = valid_indices[valid_indices > f]

            if len(before) == 0:
                result[f, start:end] = actor[after[0]]
            elif len(after) == 0:
                result[f, start:end] = actor[before[-1]]
            else:
                f0, f1 = before[-1], after[0]
                alpha = (f - f0) / (f1 - f0)
                result[f, start:end] = (1 - alpha) * actor[f0] + alpha * actor[f1]

    return result


#Normalizacja rozmiaru szkieletu względem długości kręgosłupa
def normalize_scale(ske_joints):
    num_frames = ske_joints.shape[0]

    j1 = ske_joints[:, 0:3]
    j21 = ske_joints[:, 60:63]

    spine_lengths = np.sqrt(((j1 - j21) ** 2).sum(axis=1))

    valid = spine_lengths > 1e-6
    if valid.sum() == 0:
        return ske_joints
    median_spine = np.median(spine_lengths[valid])

    if median_spine < 1e-6:
        return ske_joints
    return ske_joints / median_spine

def normalize_scale_per_actor(ske_joints):
    result = ske_joints.copy()
    num_bodies = 1 if ske_joints.shape[1] == 75 else 2

    for b in range(num_bodies):
        start = b * 75
        end = start + 75
        actor = result[:, start:end]
        j1  = actor[:, 0:3]
        j21 = actor[:, 60:63]
        spine_lengths = np.sqrt(((j1 - j21) ** 2).sum(axis=1))
        valid = spine_lengths > 1e-6
        if valid.sum() == 0:
            continue
        median_spine = np.median(spine_lengths[valid])
        if median_spine < 1e-6:
            continue
        result[:, start:end] = actor / median_spine
    return result

def seq_translation(skes_joints):
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2

        if num_bodies == 2:
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 3:6])

        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:
                ske_joints[f] -= np.tile(origin, 50)

        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)
        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)
        skes_joints[idx] = ske_joints
    return skes_joints


def align_frames(skes_joints, frames_cnt):
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()
    aligned = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            aligned[idx, :num_frames] = np.hstack((ske_joints, np.zeros_like(ske_joints)))
        else:
            aligned[idx, :num_frames] = ske_joints

    return aligned

def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 60))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1
    return labels_vector

def get_indices(performer, camera, evaluation='CS'):
    test_indices = np.empty(0)
    train_indices = np.empty(0)

    if evaluation == 'CS':
        train_ids = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
                     17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        test_ids  = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
                     24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
        for idx in test_ids:
            test_indices = np.hstack((test_indices,
                                      np.where(performer == idx)[0])).astype(int)
        for idx in train_ids:
            train_indices = np.hstack((train_indices,
                                       np.where(performer == idx)[0])).astype(int)
    else:
        train_ids = [2, 3]
        test_ids = 1
        test_indices = np.hstack((test_indices,
                                   np.where(camera == test_ids)[0])).astype(int)
        for idx in train_ids:
            train_indices = np.hstack((train_indices,
                                       np.where(camera == idx)[0])).astype(int)
    return train_indices, test_indices


def split_dataset(skes_joints, label, performer, camera, evaluation, mode):
    train_indices, test_indices = get_indices(performer, camera, evaluation)

    train_x = skes_joints[train_indices]
    train_y = one_hot_vector(label[train_indices])
    test_x  = skes_joints[test_indices]
    test_y  = one_hot_vector(label[test_indices])

    save_name = 'NTU60_%s_%s.npz' % (evaluation, mode)
    np.savez_compressed(save_name,
                        x_train=train_x.astype(np.float32),
                        y_train=train_y.astype(np.float32),
                        x_test=test_x.astype(np.float32),
                        y_test=test_y.astype(np.float32))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['baseline', 'interpolation', 'scale', 'both',
                                 'scale_per_actor', 'both_per_actor'],
                        help='Preprocessing mode')
    args = parser.parse_args()

    camera    = np.loadtxt(camera_file, dtype=int)
    performer = np.loadtxt(performer_file, dtype=int)
    label     = np.loadtxt(label_file, dtype=int) - 1
    frames_cnt = np.loadtxt(frames_file, dtype=int)

    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)

    if args.mode in ('interpolation', 'both', 'both_per_actor'):
        total_missing = 0
        for idx in range(len(skes_joints)):
            before = (skes_joints[idx].sum(axis=1) == 0).sum()
            skes_joints[idx] = interpolate_missing_frames(skes_joints[idx])
            after  = (skes_joints[idx].sum(axis=1) == 0).sum()
            total_missing += int(before - after)
        print('Interpolated %d missing frames across all sequences.' % total_missing)
    else:
        print('Skipping interpolation (baseline/scale mode).')

    skes_joints = seq_translation(skes_joints)

    if args.mode in ('scale', 'both'):
        print('Applying scale normalization (shared, actor-1 based)...')
        for idx in range(len(skes_joints)):
            skes_joints[idx] = normalize_scale(skes_joints[idx])
        print('Scale normalization done.')
    elif args.mode in ('scale_per_actor', 'both_per_actor'):
        print('Applying scale normalization (per-actor)...')
        for idx in range(len(skes_joints)):
            skes_joints[idx] = normalize_scale_per_actor(skes_joints[idx])
        print('Per-actor scale normalization done.')
    else:
        print('Skipping scale normalization (baseline/interpolation mode).')

    skes_joints = align_frames(skes_joints, frames_cnt)

    print('Saving NPZ files...')
    for evaluation in ['CS', 'CV']:
        split_dataset(skes_joints, label, performer, camera, evaluation, args.mode)
