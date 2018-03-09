# data sampler for fashion mnist
# work with data in memory

# author: Kamaldinov Ildar
import numpy as np
import torch
from torch.autograd import Variable

def data_sampler(batchsize, sample_one, sample_two, requires_grad=False, use_gpu=None):
    ind_one = np.random.randint(
        low=0, high=sample_one.shape[0] - 1, size=batchsize)
    ind_two = np.random.randint(
        low=0, high=sample_one.shape[0] - 1, size=batchsize)

    if use_gpu:
        out_1 = Variable(torch.from_numpy(sample_one[ind_one]).float(),
                         requires_grad=requires_grad).cuda()
        out_2 = Variable(torch.from_numpy(sample_two[ind_two]).float(),
                         requires_grad=requires_grad).cuda()
    else:
        out_1 = Variable(torch.from_numpy(sample_one[ind_one]).float())
        out_2 = Variable(torch.from_numpy(sample_two[ind_two]).float())
    return out_1, out_2
