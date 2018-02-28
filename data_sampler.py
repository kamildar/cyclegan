# data sampler for fashion mnist
# work with data in memory

# author: Kamaldinov Ildar
import numpy as np
import torch
from torch.autograd import Variable
from utl import aduc


def data_sampler(batchsize, sample_one, sample_two, use_gpu=None):
    ind_one = np.random.randint(
        low=0, high=sample_one.shape[0] - 1, size=batchsize)
    ind_two = np.random.randint(
        low=0, high=sample_one.shape[0] - 1, size=batchsize)

    out_1 = aduc(Variable(torch.from_numpy(sample_one[ind_one]).float()),
                 use_gpu)
    out_2 = aduc(Variable(torch.from_numpy(sample_two[ind_two]).float()),
                 use_gpu)
    return out_1, out_2