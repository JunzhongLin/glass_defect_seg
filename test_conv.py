import torch
import numpy
from torch import nn
from model import UNetWrapper

m = nn.ConvTranspose2d(16, 33, 3, stride=2)

downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

# input = torch.randn(20, 16, 50, 100)
input = torch.randn(1, 16, 12, 12)
h = downsample(input)


