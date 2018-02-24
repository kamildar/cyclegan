# data sampler for fashion mnist
# work with data in memory

# author: Kamaldinov Ildar
import numpy as np
from torch.autograd import Variable
import torch


def data_sampler(batchsize, sample_one, sample_two):
    ind_one = np.random.randint(
        low=0, high=sample_one.shape[0] - 1, size=batchsize)
    ind_two = np.random.randint(
        low=0, high=sample_one.shape[0] - 1, size=batchsize)
    out_1 = Variable(torch.from_numpy(sample_one[ind_one]))
    out_2 = Variable(torch.from_numpy(sample_two[ind_two]))
    return out_1, out_2