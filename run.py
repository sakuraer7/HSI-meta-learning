import torch
import numpy as np
from model import data_processing as dp
from model.network import Classifier
from model.function import MAML, predict
from tqdm import tqdm
import os
from collections import OrderedDict

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


def train(dataset_='IP', k_shot_=1, q_query_=1, batch_=100, epoch_=50):
    dataset = dataset_
    batch = batch_
    window_size = 25
    k_shot = k_shot_
    q_query = q_query_
    inner_train_step = 1
    inner_lr = 0.2
    meta_lr = 0.001
    meta_batch_size = 2
    max_epoch = epoch_
    eval_batches = test_batches = 2

    data, label = dp.load_data(dataset, normalize=False)
    data, _ = dp.apply_PCA(data, 30 if dataset is 'IP' else 15)
    data, label = dp.create_image_cubes(data, label, window_size)
    # 分割数据，供全量测试用
    train_data, test_data, train_label, test_label = dp.train_test_set_split(data, label, 0.7, limit=100)

    # 分割数据供meta-learning用
    train_set, test_set, val_set, n_way = dp.get_meta_data(data, label, k_shot, q_query, batch)

    train_set = np.expand_dims(train_set, 6)
    test_set = np.expand_dims(test_set, 6)
    val_set = np.expand_dims(val_set, 6)
    # print(train_set.shape, val_set.shape, test_set.shape, n_way)
    # exit()

    train_set, test_set, val_set = torch.Tensor(train_set), torch.Tensor(test_set), torch.Tensor(val_set)

    # torch 是channel在前，tensorflow是channel在后
    train_set = train_set.permute(0, 1, 2, 6, 5, 3, 4)
    test_set = test_set.permute(0, 1, 2, 6, 5, 3, 4)
    val_set = val_set.permute(0, 1, 2, 6, 5, 3, 4)
    model = Classifier(train_set.shape[-4])
    # fast_weight = OrderedDict(model.named_parameters())
    # print(fast_weight)
    # exit()
    example = torch.randn((1, train_set.shape[-4], train_set.shape[-3], train_set.shape[-2], train_set.shape[-1]))
    model.build_2d(example)
    model.get_in_ch(example)
    model.build_fc(n_way)

    model.to(device)

    train_iter = iter(train_set.numpy())
    val_iter = iter(val_set.numpy())
    test_iter = iter(test_set.numpy())
    optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    loss = torch.nn.CrossEntropyLoss()

    # 得到一个batch的数据
    def get_meta_batch(batch_size, k_shot, q_query, data_loader, data_iterator):
        data = []
        for _ in range(batch_size):
            try:
                task_data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader.numpy())
                task_data = next(data_iterator)

            train_data = []
            val_data = []
            for x in task_data:
                train_data.extend(x[:k_shot])
                val_data.extend(x[k_shot:])
            train_data = torch.Tensor(train_data)
            val_data = torch.Tensor(val_data)
            task_data = torch.cat((train_data, val_data), 0)
            data.append(task_data)

        return torch.stack(data), data_iterator
    # train
    for epoch in range(max_epoch):
        print('epoch: {}/{}'.format(epoch, max_epoch))
        train_meta_loss = []
        train_acc = []
        for step in tqdm(range(len(train_set) // meta_batch_size)):
            x, train_iter = get_meta_batch(meta_batch_size, k_shot, q_query, train_set, train_iter)

            meta_loss, acc = MAML(model, optimizer, x, n_way, k_shot, q_query, loss, inner_train_step=inner_train_step,
                                  inner_lr=inner_lr)
            train_meta_loss.append(meta_loss.item())
            train_acc.append(acc)

        print("loss:", np.mean(train_meta_loss))
        print("acc :", np.mean(train_acc))
    # val
    val_acc = []
    for eval_step in tqdm(range(len(val_set) // eval_batches)):
        x, val_iter = get_meta_batch(eval_batches, k_shot, q_query, val_set, val_iter)
        _, acc = MAML(model, optimizer, x, n_way, k_shot, q_query, loss, inner_train_step=3, inner_lr=inner_lr, train=False)
        val_acc.append(acc)

    print('val acc: {}'.format(np.mean(val_acc)))
    # test
    test_acc = []
    for test_step in tqdm(range(len(test_set) // test_batches)):
        x, test_iter = get_meta_batch(test_batches, k_shot, q_query, test_set, test_iter)
        _, acc = MAML(model, optimizer, x, n_way, k_shot, q_query, loss, inner_train_step=3, inner_lr=inner_lr, train=False)
        test_acc.append(acc)
    print('test acc: {}'.format(np.mean(test_acc)))

    # 全量数据test
    train_data = np.expand_dims(train_data, 1)
    train_data = np.transpose(train_data, (0, 1, 4, 2, 3))
    pred_path = './predicted/%s_%sway_%shot.npy' % (dataset, str(n_way), str(k_shot))
    measure_path = './measure/%s_%sway_%shot.txt' % (dataset, str(n_way), str(k_shot))
    predict(model, train_data, train_label, n_way, 16, pred_path, measure_path)
    path = './save_model/%s_meta_model_%sway_%sshot.pkl' % (dataset, str(n_way), str(k_shot))
    torch.save(model, path)


if __name__ == '__main__':
    for name in ['IP', 'PU']:
        train(dataset_=name, k_shot_=3, q_query_=1, batch_=128, epoch_=10)
