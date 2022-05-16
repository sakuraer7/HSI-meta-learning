import numpy as np
import os
from model.densenet import SpacChannel
from torchsummary import summary
import torch

model = SpacChannel(growthRate=12, depth=50, reduction=0.5, nClasses=16, bottleneck=True)

data = torch.rand((10, 1, 25, 25, 30))

conv = torch.nn.Conv3d(1, 16, kernel_size=(3, 3, 7), padding=1, bias=False)
out = conv(data)
print(out.shape)

summary(model, input_size=(1, 25, 25, 30))

out = model(data)
print(out.shape)


