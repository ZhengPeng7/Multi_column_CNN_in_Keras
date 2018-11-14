import os
import cv2
import numpy as np
from keras import backend as K
from sklearn.utils import shuffle


def gen_imgPaths_and_labelPaths(dataset="B"):
    train_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train/'
    train_den_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train_den/'
    val_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val/'
    val_den_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val_den/'
    img_path = './data/original/ShanghaiTech/part_' + dataset + '/test_data/images/'
    den_path = './data/original/ShanghaiTech/part_' + dataset + '/test_data/ground-truth_csv/'
    print(img_path)
    train_paths = sorted([train_path + p for p in os.listdir(train_path)], key=lambda x: float(x[:-len('.jpg')].split('/')[-1].replace('_', '.')))
    train_labels = sorted([train_den_path + p for p in os.listdir(train_den_path)], key=lambda x: float(x[:-len('.jpg')].split('/')[-1].replace('_', '.')))
    validation_paths = sorted([val_path + p for p in os.listdir(val_path)], key=lambda x: float(x[:-len('.jpg')].split('/')[-1].replace('_', '.')))
    validation_labels = sorted([val_den_path + p for p in os.listdir(val_den_path)], key=lambda x: float(x[:-len('.jpg')].split('/')[-1].replace('_', '.')))
    test_paths = sorted([img_path + p for p in os.listdir(img_path)], key=lambda x: int(x[:-len('.jpg')].split('/')[-1].split('_')[1]))
    test_labels = sorted([den_path + p for p in os.listdir(den_path)], key=lambda x: int(x[:-len('.jpg')].split('/')[-1].split('_')[1]))

    return train_paths, train_labels, validation_paths, validation_labels, test_paths, test_labels


def generate_generator(img_paths, label_paths, batch_size=32, is_shuffle=False, img_flip=0):
    flag_continue = 0
    idx_total = 0
    img_paths = np.squeeze(img_paths).tolist() if isinstance(img_paths, np.ndarray) else img_paths
    label_paths = np.squeeze(label_paths).tolist() if isinstance(label_paths, np.ndarray) else label_paths
    if is_shuffle:
        paths_shuffled = shuffle(np.hstack([np.asarray(img_paths).reshape(-1, 1), np.asarray(label_paths).reshape(-1, 1)]))
        img_paths, label_paths = np.squeeze(paths_shuffled[:, 0]).tolist(), np.squeeze(paths_shuffled[:, 1]).tolist()
    data_len = len(label_paths)
    while True:
        if not flag_continue:
            x = []
            y = []
            inner_iter_num = batch_size
        else:
            idx_total = 0
            inner_iter_num = batch_size - data_len % batch_size
        for _ in range(inner_iter_num):
            if idx_total >= data_len:
                flag_continue = 1
                break
            else:
                flag_continue = 0
            img = (cv2.imread(img_paths[idx_total], 0) - 127.5) / 128
            density_map = np.loadtxt(label_paths[idx_total], delimiter=',')
            stride = 4
            density_map_quarter = np.zeros((np.asarray(density_map.shape).astype(int)//stride).tolist())
            for r in range(density_map_quarter.shape[0]):
                for c in range(density_map_quarter.shape[1]):
                    density_map_quarter[r, c] = np.sum(density_map[r*stride:(r+1)*stride, c*stride:(c+1)*stride])
            x.append(img.reshape(*img.shape, 1))
            y.append(density_map_quarter.reshape(*density_map_quarter.shape, 1))
            if img_flip:
                pass
            idx_total += 1
        if not flag_continue:
            x, y = np.asarray(x), np.asarray(y)
            yield x, y


def monitor_mae(labels, preds):
    return K.sum(K.abs(labels - preds)) / 1


def monitor_mse(labels, preds):
    return K.sum(K.square(labels - preds)) / 1
