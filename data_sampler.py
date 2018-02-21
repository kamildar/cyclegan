# data sampler for fashion mnist
# work with data in memory

# author: Kamaldinov Ildar
import numpy as np

def data_sampler(batchsize, sample_one, sample_two):
    ind_one = np.random.randint(
        low=0, high=sample_one.shape[0] - 1, size=batchsize)
    ind_two = np.random.randint(
        low=0, high=sample_one.shape[0] - 1, size=batchsize)
    return sample_one[ind_one], sample_two[ind_two]