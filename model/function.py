import numpy as np
from collections import OrderedDict
import torch
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score
from .data_processing import create_label
from operator import truediv

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


# 接下來這裡是MAML的核心。演算法就跟原文完全一樣，這個函數做的事情就是用 "一個 meta-batch的 data" 更新參數。
# 這裡助教實作的是二階MAML(inner_train_step = 1)，對應老師投影片 meta learning p.13~p.18。如果要找一階的數學推導，在老師投影片 p.25。
# (http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Meta1%20(v6).pdf)
# 以下詳細解釋：
def MAML(model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step=1, inner_lr=0.4, train=True):
    """
    Args:
    x is the input omniglot images for a meta_step, shape = [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
    n_way: 每個分類的 task 要有幾個 class
    k_shot: 每個類別在 training 的時候會有多少張照片
    q_query: 在 testing 時，每個類別會用多少張照片 update
    """
    criterion = loss_fn
    task_loss = []  # 這裡面之後會放入每個 task 的 loss
    task_acc = []   # 這裡面之後會放入每個 task 的 acc
    for meta_batch in x:
        # 前一半数据进行第一次更新，后一半数据进行第二次更新即更新meta_model
        train_set = meta_batch[:n_way * k_shot]  # train_set 是我們拿來 update inner loop 參數的 data
        val_set = meta_batch[n_way * k_shot:]    # val_set 是我們拿來 update outer loop 參數的 data

        fast_weights = OrderedDict(
            model.named_parameters())  # 在inner loop update參數時，我們不能動到實際參數，因此用fast_weights來儲存新的參數θ'
        # 由于使loss用第二次的梯度更新meta，所以要对参数求两次导，fast_weight存储的是更新过第一次之后的参数
        for inner_step in range(inner_train_step):  # 這個 for loop 是 Algorithm2 的 line 7~8
            # 實際上我們 inner loop 只有 update 一次 gradients，不過某些 task 可能會需要多次 update inner loop 的 θ'，
            # 所以我們還是用 for loop 來寫
            logits = model.functional_forward(train_set, fast_weights)
            train_label = create_label(n_way, k_shot).cuda()
            logits = logits.to(device)
            train_label = train_label.to(device)
            loss = criterion(logits, train_label)
            grads = torch.autograd.grad(loss, fast_weights.values(),
                                        create_graph=True)  # 這裡是要計算出 loss 對 θ 的微分 (∇loss)
            fast_weights = OrderedDict((name, param - inner_lr * grad)
                                       for ((name, param), grad) in
                                       zip(fast_weights.items(), grads))  # 這裡是用剛剛算出的 ∇loss 來 update θ 變成 θ'

        logits = model.functional_forward(val_set, fast_weights)  # 這裡用 val_set 和 θ' 算 logit
        logits = logits.to(device)
        val_label = val_label.to(device)
        val_label = create_label(n_way, q_query).cuda()
        loss = criterion(logits, val_label)  # 這裡用 val_set 和 θ' 算 loss
        task_loss.append(loss)  # 把這個 task 的 loss 丟進 task_loss 裡面
        acc = np.asarray([torch.argmax(logits, -1).cpu().numpy() == val_label.cpu().numpy()]).mean()  # 算 accuracy
        task_acc.append(acc)

    model.train()
    optimizer.zero_grad()
    # 使用第二次的loss来更新meta model
    meta_batch_loss = torch.stack(task_loss).mean()  # 我們要用一整個 batch 的 loss 來 update θ (不是 θ')
    if train:
        meta_batch_loss.backward()
        optimizer.step()
    task_acc = np.mean(task_acc)
    return meta_batch_loss, task_acc


def predict(model, data, label, n_way, batch_size=16, pred_path=None, measure_path=None):
    model.eval()
    # one_hot = torch.zeros(data.shape[0], n_way).scatter_(1, label, 1)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    batch_num = data.shape[0] // batch_size
    predicted = np.array([])
    for i in range(batch_num):
        sub_data = data[batch_size*i: batch_size*(i+1)]
        tensor_data = torch.Tensor(sub_data)
        tensor_data = tensor_data.to(device)
        pred = model(tensor_data)
        pred = pred.detach().cpu().numpy()
        if len(predicted) == 0:
            predicted = pred
        else:
            predicted = np.vstack((predicted, pred))
    if pred_path:
        np.save(pred_path, np.argmax(predicted, axis=1))
    label = label[:predicted.shape[0]]
    classification = classification_report(np.argmax(predicted, axis=1), label)
    print(classification)
    oa, aa, kappa = get_measure(predicted, label)
    if measure_path:
        with open(measure_path, 'w') as f:
            f.write('OA: {}\nAA: {}\nKappa: {}'.format(oa, aa, kappa))
    print('oa: {} aa: {} kappa: {}'.format(oa, aa, kappa))


def get_measure(y_pred, y):
    y_pred = np.argmax(y_pred, axis=1)
    confusion = confusion_matrix(y, y_pred)
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    avg_acc = np.mean(each_acc)
    oa = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    return oa * 100, avg_acc * 100, kappa

