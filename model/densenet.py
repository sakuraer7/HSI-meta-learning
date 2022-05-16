import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        # 每一个bottleneck内包含两层卷积，输入是nChaanel，输出是growRate与输入的并联
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        # 这里第一层卷积的作用是对输入降维，减少通道数，即bottleneck
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


# transition是将输出减半，通过pooling的方式
class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        # block数量，一个block中包含conv，relu，pool等操作
        # block = (总层数-最后的全连接层) // 每个block中包含的层数
        nDenseBlocks = (depth - 4) // 3
        # bottleneck工作在block内部，是为了校正输入的channel不过于庞大，而transition工作在block之间，作用同bottleneck
        # 由于每层之前都有一个bottleneck，所以总层数不变，block数量再减半
        if bottleneck:
            nDenseBlocks //= 2

        # growRate是每个block输入channel增长速度，一般输入算第一层，所以第一个block为2 * growRate，依次类推：n * growRate
        nChannels = 2 * growthRate
        # 构建第一个卷积层
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        # 第一个dense block
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        # 每经过一个dense block，输入增长nDenseBlocks * growthRate，因为一个dense block正好有nDenseBlocks个层，每过一层涨1growth
        nChannels += nDenseBlocks * growthRate
        # transition层，将输入降低到原来的reduction，取值一般是0.1~1，一般取0.5
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        # 上一个block的输出为当前block的输入，然后构建过程同第一个block
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        # 最后的全连接层，即分类层
        self.fc = nn.Linear(nChannels, nClasses)

        # 手动参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    # 根据block数，构建网络，每构建一个block，输入通道nChannel增长growthRate
    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        # 第一层卷积
        out = self.conv1(x)
        # 第一个dense block，并且将输出transition
        out = self.trans1(self.dense1(out))
        # 第二个dense block，并且将输出transition
        out = self.trans2(self.dense2(out))
        # 第三个dense block，无transition
        out = self.dense3(out)
        # pooling操作，然后分类
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out


class Bottleneck3D(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size):
        super(Bottleneck3D, self).__init__()
        # 每一个bottleneck内包含两层卷积，输入是nChaanel，输出是growRate与输入的并联
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm3d(nChannels)
        # 这里第一层卷积的作用是对输入降维，减少通道数，即bottleneck
        self.conv1 = nn.Conv3d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(interChannels)
        self.conv2 = nn.Conv3d(interChannels, growthRate, kernel_size=kernel_size,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer3D(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size):
        super(SingleLayer3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(nChannels)
        self.conv1 = nn.Conv3d(nChannels, growthRate, kernel_size=kernel_size,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


# transition是将输出减半，通过pooling的方式
class Transition3D(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(nChannels)
        self.conv1 = nn.Conv3d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool3d(out, 2)
        return out


class SpacChannel(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(SpacChannel, self).__init__()
        nDenseBlocks = (depth - 4) // 3
        # bottleneck工作在block内部，是为了校正输入的channel不过于庞大，而transition工作在block之间，作用同bottleneck
        # 由于每层之前都有一个bottleneck，所以总层数不变，block数量再减半
        if bottleneck:
            nDenseBlocks //= 2

        # growRate是每个block输入channel增长速度，一般输入算第一层，所以第一个block为2 * growRate，依次类推：n * growRate
        nChannels = 2 * growthRate
        # 构建第一个卷积层
        self.conv1 = nn.Conv3d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)
        # 第一个dense block
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, (3, 3, 3))
        # 每经过一个dense block，输入增长nDenseBlocks * growthRate，因为一个dense block正好有nDenseBlocks个层，每过一层涨1growth
        nChannels += nDenseBlocks * growthRate
        # transition层，将输入降低到原来的reduction，取值一般是0.1~1，一般取0.5
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition3D(nChannels, nOutChannels)

        # 上一个block的输出为当前block的输入，然后构建过程同第一个block
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, (3, 3, 3))
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition3D(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, (3, 3, 3))
        nChannels += nDenseBlocks * growthRate

        self.bn1 = nn.BatchNorm3d(nChannels)

        # 手动参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        # 根据block数，构建网络，每构建一个block，输入通道nChannel增长growthRate

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, kernel_size):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck3D(nChannels, growthRate, kernel_size))
            else:
                layers.append(SingleLayer3D(nChannels, growthRate, kernel_size))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        # 第一层卷积
        out = self.conv1(x)
        # 第一个dense block，并且将输出transition
        out = self.trans1(self.dense1(out))
        # 第二个dense block，并且将输出transition
        out = self.trans2(self.dense2(out))
        # 第三个dense block，无transition
        out = self.dense3(out)
        a, b, c = out.shape[-3] // 2 + 1, out.shape[-2] // 2 + 1, out.shape[-1] // 2 + 1
        # pooling操作，然后分类
        out = torch.squeeze(F.avg_pool3d(F.relu(self.bn1(out)), kernel_size=(a, b, c), stride=(a, b, c)))
        return out
