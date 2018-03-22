import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from data_sampler import SequentialSampler, BatchSampler
from utl import aduc


k3same = dict(kernel_size=3, stride=1, padding=1)
k5same = dict(kernel_size=5, stride=1, padding=2)
k7same = dict(kernel_size=7, stride=1, padding=3)


class InceptionSham(nn.Module):

    def __init__(self,
                 num_classes=10,
                 input_nc=1,
                 dropout=0.5):
        super().__init__()

        foot = [
            BasicConv2d(input_nc, 4, **k3same),
            BasicConv2d(4, 8, **k3same)
        ]

        body = [
            InceptionX(8, pool_features=16),
            InceptionX(64, pool_features=16),

            InceptionZ(64),
            InceptionZ(96)
        ]

        neck = [
            nn.AvgPool2d(7),
            nn.Dropout(p=dropout)
        ]

        head = nn.Linear(96, num_classes)

        self.seq = nn.Sequential(*(foot + body + neck))
        self.head = head

    def forward(self, x):
        x = self.seq(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class InceptionX(nn.Module):

    def __init__(self, in_channels, pool_features, **kwargs):
        super().__init__()
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, 16, **k3same),
            BasicConv2d(16, 32, **k3same)
        )

        self.branch5x5 = BasicConv2d(in_channels, 16, **k5same)

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(**k3same),
            BasicConv2d(in_channels, pool_features, **k3same)
        )

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class InceptionZ(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, 64,
                        kernel_size=3, stride=2, padding=1),
            BasicConv2d(64, 64, **k3same)
        )

        self.branch7x7 = BasicConv2d(in_channels, 32,
                                     kernel_size=7, stride=2, padding=3)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch7x7 = self.branch7x7(x)

        outputs = [branch3x3, branch7x7]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def inception_score(gener_a, gener_b,
                    inception,
                    data_a, data_b,
                    batch_size=10, splits=10):
    gener_a.eval()
    gener_b.eval()

    n_images = min(data_a.shape[0], data_b.shape[0])

    sampler = SequentialSampler(n_images)
    batch_sampler = BatchSampler(sampler, batch_size)
    split_sampler = BatchSampler(sampler, batch_size=(n_images // splits))

    pred_a = np.zeros((n_images, 10))
    pred_b = np.zeros((n_images, 10))

    for i in batch_sampler:
        batch_a = aduc(torch.FloatTensor(data_a[i]).view(-1, 1, 28, 28))
        batch_b = aduc(torch.FloatTensor(data_b[i]).view(-1, 1, 28, 28))

        pred_a[i] = F.softmax(inception(gener_a(batch_b)),
                              dim=1).data.cpu().numpy()
        pred_b[i] = F.softmax(inception(gener_b(batch_a)),
                              dim=1).data.cpu().numpy()

    obs_kl_a = []
    obs_kl_b = []

    for k in split_sampler:
        part_a = pred_a[k]
        part_b = pred_b[k]

        py_a = np.mean(pred_a, axis=0)
        py_b = np.mean(pred_b, axis=0)

        scores_a = []
        scores_b = []
        for row in range(part_a.shape[0]):
            pyx_a = part_a[row]
            pyx_b = part_b[row]

            scores_a.append(entropy(pyx_a, py_a))
            scores_b.append(entropy(pyx_b, py_b))

        obs_kl_a.append(np.exp(np.mean(scores_a)))
        obs_kl_b.append(np.exp(np.mean(scores_b)))

    return np.mean(obs_kl_a), np.mean(obs_kl_b)
