import torch
from torch import nn


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
