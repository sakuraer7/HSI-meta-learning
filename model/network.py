import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from .data_processing import create_label

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


# 卷积块类实现
def conv_block(in_ch, out_ch):
    # 每一个卷积块有四层分别为卷积、BN、relu、pooling
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2, stride=2))


# 2D卷积块函数实现
def conv_block_function(x, w, b, w_bn, b_bn):
    x = x.to(device)
    x = F.conv2d(x, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


# 3D卷积块函数实现
def conv3d_block_function(x, w, b):
    x = x.to(device)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    x = F.conv3d(x, w, b)
    x = F.relu(x)
    return x


class Classifier(nn.Module):
    def __init__(self, in_ch):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, 8, kernel_size=(7, 3, 3))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3))
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3))
        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(0.3)
        self.dropout1 = nn.Dropout(0.1)

    def build_2d(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        self.conv_2d_in = x.shape[-3]
        self.conv4 = nn.Conv2d(self.conv_2d_in, 64, kernel_size=(3, 3))

    def build_fc(self, n_way):
        self.fc1 = nn.Linear(self.fc_in_ch, 256)
        self.fc2 = nn.Linear(256, 128)
        self.logits = nn.Linear(128, n_way)

    def get_in_ch(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        x = self.conv4(x)
        # flatten
        x = x.view(x.shape[0], -1)
        self.fc_in_ch = x.shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 3dConv->2dConv
        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        x = self.conv4(x)
        # flatten
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.logits(x)
        x = F.softmax(x, 0)
        return x

    def functional_forward(self, x, params):
        """
        Arguments:
        x: input images [batch, 1, 28, 28]
        params: 模型的參數，也就是 convolution 的 weight 跟 bias，以及 batchnormalization 的  weight 跟 bias
                這是一個 OrderedDict
        """
        # 3dConv
        for block in [1, 2, 3]:
            x = conv3d_block_function(x, params[f'conv{block}.weight'], params[f'conv{block}.bias'])
        # 2dConv
        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        x = F.conv2d(x, params['conv4.weight'], params['conv4.bias'])
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        for i in [1, 2]:
            x = F.linear(x, params[f'fc{i}.weight'], params[f'fc{i}.bias'])
            x = F.relu(x)
        x = F.linear(x, params['logits.weight'], params['logits.bias'])
        x = F.softmax(x, dim=0)
        return x


if __name__ == '__main__':
    data = torch.randn((10, 3, 21, 21))
    model = Classifier(3)
    model.get_in_ch(data)
    model.build_fc(16)
    out = model(data)
    print(out.shape)
    fast_weights = OrderedDict(model.named_parameters())
    print(fast_weights)

