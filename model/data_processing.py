# coding: utf-8
import os
from scipy import io
from sklearn.decomposition import PCA
import numpy as np
import torch

file_dict = {'BO': ['Botswana.mat', 'Botswana_gt.mat'], 'KS': ['KSC.mat', 'KSC_gt.mat'],
             'PC': ['Pavia.mat', 'Pavia_gt.mat'], 'IP': ['Indian_pines_corrected.mat', 'Indian_pines_gt.mat'],
             'PU': ['PaviaU.mat', 'PaviaU_gt.mat'], 'SA': ['Salinas_corrected.mat', 'Salinas_gt.mat'],
             'HS': ['houston.mat', 'houston_gt.mat']}

key_dict = {'BO': ['Botswana', 'Botswana_gt'], 'KS': ['KSC', 'KSC_gt'],
            'PC': ['pavia', 'pavia_gt'], 'IP': ['indian_pines_corrected', 'indian_pines_gt'],
            'PU': ['paviaU', 'paviaU_gt'], 'SA': ['salinas_corrected', 'salinas_gt'],
            'HS': ['houston', 'houston_gt']}

K_dict = {'BO': 15, 'KS': 15, 'PC': 20, 'IP': 30, 'PU': 15, 'SA': 15, 'HS': 15}
out_units = {'BO': 14, 'KS': 13, 'PC': 9, 'IP': 16, 'PU': 9, 'SA': 16, 'HS': 15}


def load_data(name='IP', normalize=True):
    data_path = os.path.join(os.getcwd(), 'data')
    data = io.loadmat(os.path.join(data_path, file_dict[name][0]))[key_dict[name][0]]
    labels = io.loadmat(os.path.join(data_path, file_dict[name][1]))[key_dict[name][1]]
    # 标准化
    if normalize:
        a = np.max(data)
        b = np.min(data)
        data = (data - b) / (a - b)
    return data, labels


'''
def train_test_set_split(data, label, test_ratio=0.3, random_seed=345):
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=test_ratio,
                                                                      random_state=random_seed)
    return data_train, data_test, label_train, label_test
'''


def band_select(data, bands):
    band_var = {}
    for i in range(data.shape[-1]):
        band_i = data[:, :, i]
        band_var[np.var(band_i)] = i
    sorted_band_var = sorted(band_var.items())
    delete_band = [sorted_band_var[k][1] for k in range(data.shape[-1] - bands)]
    delete_band = sorted(delete_band)
    new_data = np.zeros(shape=(data.shape[0], data.shape[1], bands), dtype=np.float64)
    count = 0
    for i in range(data.shape[-1]):
        if i in delete_band:
            continue
        new_data[:, :, count] = data[:, :, i]
        count += 1
    return new_data


def get_meta_data(data, label, k_shot, q_query, batch=100):
    class_num = np.max(label) - np.min(label) + 1
    train_set = []
    test_set = []
    val_set = []
    for b in range(batch):
        train_tmp = []
        test_tmp = []
        val_tmp = []
        for i in range(int(class_num)):
            t1 = []
            t2 = []
            t3 = []
            index_list = np.argwhere(label == i)
            np.random.shuffle(index_list)
            total_num = len(index_list)
            if total_num < 3 * (k_shot + q_query):
                raise Exception('总样本数小于k_shot与q_query之和')
            for x in index_list[: k_shot + q_query]:
                t1.append(data[x[0]])
            for x in index_list[k_shot + q_query: 2 * (k_shot + q_query)]:
                t2.append(data[x[0]])
            for x in index_list[2 * (k_shot + q_query): 3 * (k_shot + q_query)]:
                t3.append(data[x[0]])
            train_tmp.append(t1)
            test_tmp.append(t2)
            val_tmp.append(t3)

        train_set.append(train_tmp)
        test_set.append(test_tmp)
        val_set.append(val_tmp)

    train_set = np.array(train_set, dtype=np.float32)
    test_set = np.array(test_set, dtype=np.float32)
    val_set = np.array(val_set, dtype=np.float32)
    return train_set, test_set, val_set, int(class_num)


def create_label(n_way, k_shot):
    return torch.arange(n_way).repeat_interleave(k_shot).long()


def train_test_set_split(data, label, test_ratio=0.3, limit=200, augment=False):
    class_num = np.max(label) - np.min(label) + 1
    data_train = []
    data_train_1d = []
    data_test = []
    data_test_1d = []
    label_train = []
    label_test = []
    center = int(data.shape[1]/2)
    for i in range(int(class_num)):
        index_list = np.argwhere(label == i)
        np.random.shuffle(index_list)
        total_num = len(index_list)
        train_num = int(total_num * (1 - test_ratio))
        if train_num < 5:
            train_num = 5
        test_num = total_num - train_num
        if train_num > limit:
            train_num = limit
        if test_num > 2000:
            test_num = 2000
        for x in index_list[:train_num]:
            data_train.append(data[x[0]])
            # data_train_1d.append(data[x[0]][center][center])
            label_train.append(i)
        for x in index_list[-test_num:]:
            data_test.append(data[x[0]])
            # data_test_1d.append(data[x[0]][center][center])
            label_test.append(i)
    #     print('class{} total num: {} train num: {} test num: {}'.format(i, total_num, train_num, test_num))
    #
    # print('data train: {} data train1d: {} label: {}\n'
    #       'data test: {} data test1d: {} label: {}'.format(
    #         len(data_train), len(data_train_1d), len(label_train),
    #         len(data_test), len(data_test_1d), len(label_test)))
    data_train = np.array(data_train, dtype=np.float32)
    data_train_1d = np.array(data_train_1d, dtype=np.float32)
    data_test = np.array(data_test, dtype=np.float32)
    data_test_1d = np.array(data_test_1d, dtype=np.float32)
    label_train = np.array(label_train, dtype=np.float32)
    label_test = np.array(label_test, dtype=np.float32)
    data_train_row = data_train
    data_train_1d_row = data_train_1d
    label_train_row = label_train
    if augment:
        # 正太分布增强样本
        # 光谱维度每4列其中2列增加高斯噪声
        noise_1 = np.random.normal(0, 0.1, data_train.shape)
        noise_2 = np.random.normal(0, 0.1, data_train_1d.shape)
        for i in range(data_train.shape[0]):
            for j in range(data_train.shape[-1])[3::4]:
                noise_1[i, :, :, j - 1] = 0
                noise_1[i, :, :, j - 2] = 0
                # noise_1[i, :, :, j - 3] = 0
        for i in range(data_train_1d.shape[0]):
            for j in range(data_train_1d.shape[-1])[3::4]:
                # noise_2[i, :, :, j - 1] = 0
                noise_2[i, :, :, j - 2] = 0
                noise_2[i, :, :, j - 3] = 0
        augment_data_train_1 = data_train + noise_1
        augment_data_train_1d_1 = data_train_1d + noise_2
        # 光谱维度每8列其中4列增加高斯噪声
        noise_3 = np.random.normal(0, 0.1, data_train.shape)
        noise_4 = np.random.normal(0, 0.1, data_train_1d.shape)
        for i in range(data_train.shape[0]):
            for j in range(data_train.shape[-1])[-8::-8]:
                noise_3[i, :, :, j] = 0
                noise_3[i, :, :, j + 1] = 0
                noise_3[i, :, :, j + 2] = 0
                noise_3[i, :, :, j + 3] = 0
        rest_num = data_train.shape[-1] % 8
        rest_num = rest_num // 2
        for i in range(data_train.shape[0]):
            for j in range(rest_num):
                noise_3[i, :, :, j] = 0
        for i in range(data_train_1d.shape[0]):
            for j in range(data_train_1d.shape[-1])[-8::-8]:
                noise_4[i, :, :, j] = 0
                noise_4[i, :, :, j + 1] = 0
                noise_4[i, :, :, j + 2] = 0
                noise_4[i, :, :, j + 3] = 0
        augment_data_train_2 = data_train + noise_3
        augment_data_train_1d_2 = data_train_1d + noise_4
        new_data_train = np.concatenate(
            (data_train, augment_data_train_1, augment_data_train_2),
            axis=0)
        new_data_train_1d = np.concatenate(
            (data_train_1d, augment_data_train_1d_1, augment_data_train_1d_2),
            axis=0)
        # print(new_data_train.shape, new_data_train_1d.shape)
        new_label_train = np.concatenate((label_train_row, label_train_row, label_train_row), axis=0)
        data_train = new_data_train
        data_train_1d = new_data_train_1d
        label_train = new_label_train
        # 在光谱维度上随机置固定数量的0增强样本
        # new_data_train = data_train_row
        # new_data_train_1d = data_train_1d_row
        # zero_num = 300
        # row_list = np.array([i for i in range(new_data_train.shape[1])])
        # col_list = np.array([i for i in range(new_data_train.shape[2])])
        #
        # for i in range(new_data_train.shape[0]):
        #     for j in range(new_data_train.shape[-1]):
        #         np.random.shuffle(row_list)
        #         np.random.shuffle(col_list)
        #         for m in range(15):
        #             for n in range(zero_num // 15):
        #                 new_data_train[i, row_list[m], col_list[n], j] = 0
        # zero_num = 6
        # row_list = np.array([i for i in range(new_data_train_1d.shape[1])])
        # col_list = np.array([i for i in range(new_data_train_1d.shape[2])])
        # for i in range(new_data_train_1d.shape[0]):
        #     for j in range(new_data_train_1d.shape[-1]):
        #         np.random.shuffle(row_list)
        #         np.random.shuffle(col_list)
        #         for m in range(2):
        #             for n in range(zero_num // 2):
        #                 new_data_train_1d[i, row_list[m], col_list[n], j] = 0
        #
        # data_train = np.concatenate((data_train, new_data_train), axis=0)
        # data_train_1d = np.concatenate((data_train_1d, new_data_train_1d), axis=0)
        # label_train = np.concatenate((label_train, label_train_row), axis=0)
    seed = np.random.randint(20, 100)
    np.random.seed(seed * 10)
    np.random.shuffle(data_train)
    np.random.seed(seed * 10)
    np.random.shuffle(data_train_1d)
    np.random.seed(seed * 10)
    np.random.shuffle(label_train)
    seed = np.random.randint(40, 60)
    np.random.seed(seed * 10)
    np.random.shuffle(data_test)
    np.random.seed(seed * 10)
    np.random.shuffle(data_test_1d)
    np.random.seed(seed * 10)
    np.random.shuffle(label_test)
    # print(data_train.shape, data_train_1d.shape, noise_1.shape, noise_2.shape)
    return data_train, data_test, label_train, label_test


def apply_PCA(data, num_components=75):
    """

    :param data: input data
    :param num_components: the num reversed
    :return:
    """
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], num_components))
    return new_data, pca


def pad_with_zeros(data, margin=2):
    """

    :param data: Input data
    :param margin: the with of padding
    :return: new data after padding
    """
    new_data = np.zeros((data.shape[0] + 2 * margin, data.shape[1] + 2 * margin, data.shape[2]))
    x_offset = margin
    y_offset = margin
    new_data[x_offset:data.shape[0] + x_offset, y_offset:data.shape[1] + y_offset, :] = data
    return new_data


def pad_with_mirror(data, margin=2):
    new_data = np.zeros((data.shape[0] + 2 * margin, data.shape[1] + 2 * margin, data.shape[2]))
    x_offset = margin
    y_offset = margin
    new_data[x_offset:data.shape[0] + x_offset, y_offset:data.shape[1] + y_offset, :] = data
    new_data[0:x_offset, y_offset:data.shape[1] + y_offset, :] = data[margin:0:-1, :, :]
    new_data[data.shape[0] + x_offset:, y_offset:data.shape[1] + y_offset, :] = \
        data[-2:-(margin+2):-1, :, :]
    new_data[:, 0:y_offset, :] = new_data[:, margin*2:margin:-1, :]
    new_data[:, data.shape[1] + y_offset:, :] = new_data[:, -(margin+2):-(margin*2+2):-1, :]
    return new_data


def create_image_cubes(data, labels, window_size=5, remove_zero_labels=True):
    """

    :param data: row data
    :param labels: class
    :param window_size: num of band
    :param remove_zero_labels: if True: delete class 0, and all class sub 1, because one-hot
    :return:
    """
    margin = int((window_size - 1) / 2)
    zero_padded_data = pad_with_mirror(data, margin)
    # zero_padded_data = pad_with_zeros(data, margin=margin)
    # split patches
    patches_data = np.zeros((data.shape[0] * data.shape[1], window_size, window_size, data.shape[2]), dtype=np.float32)
    patches_labels = np.zeros((data.shape[0] * data.shape[1]))
    patch_index = 0
    for r in range(margin, zero_padded_data.shape[0] - margin):
        for c in range(margin, zero_padded_data.shape[1] - margin):
            patch = zero_padded_data[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patches_data[patch_index, :, :, :] = patch
            patches_labels[patch_index] = labels[r-margin, c-margin]
            patch_index = patch_index + 1

    if remove_zero_labels:
        patches_data = patches_data[patches_labels > 0, :, :, :]
        patches_labels = patches_labels[patches_labels > 0]
        patches_labels -= 1
    return patches_data, patches_labels


def apply_PCA(data, num_components=75):
    """

    :param data: input data
    :param num_components: the num reversed
    :return:
    """
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], num_components))
    return new_data, pca

