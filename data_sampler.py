# data sampler for fashion mnist
# work with data in memory

# author: Kamaldinov Ildar
import numpy as np
import random
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


class Sampler(object):
    """Base class for samplers.

    All samplers should subclass `Sampler` and define `__iter__` and `__len__`
    methods.
    """
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements from [0, length) sequentially.

    Parameters
    ----------
    length : int
        Length of the sequence.
    """
    def __init__(self, length):
        self._length = length

    def __iter__(self):
        return iter(range(self._length))

    def __len__(self):
        return self._length


class RandomSampler(Sampler):
    """Samples elements from [0, length) randomly without replacement.

    Parameters
    ----------
    length : int
        Length of the sequence.
    """
    def __init__(self, length):
        self._length = length

    def __iter__(self):
        indices = list(range(self._length))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self._length


class BatchSampler(Sampler):
    """Wraps over another `Sampler` and return mini-batches of samples.

    Parameters
    ----------
    sampler : Sampler
        The source Sampler.
    batch_size : int
        Size of mini-batch.
    last_batch : {'keep', 'discard', 'rollover'}
        Specifies how the last batch is handled if batch_size does not evenly
        divide sequence length.

        If 'keep', the last batch will be returned directly, but will contain
        less element than `batch_size` requires.

        If 'discard', the last batch will be discarded.

        If 'rollover', the remaining elements will be rolled over to the next
        iteration.

    Examples
    --------
    >>> sampler = gluon.data.SequentialSampler(10)
    >>> batch_sampler = gluon.data.BatchSampler(sampler, 3, 'keep')
    >>> list(batch_sampler)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    def __init__(self, sampler, batch_size, last_batch='keep'):
        self._sampler = sampler
        self._batch_size = batch_size
        self._last_batch = last_batch
        self._prev = []

    def __iter__(self):
        batch, self._prev = self._prev, []
        for i in self._sampler:
            batch.append(i)
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        if batch:
            if self._last_batch == 'keep':
                yield batch
            elif self._last_batch == 'discard':
                return
            elif self._last_batch == 'rollover':
                self._prev = batch
            else:
                raise ValueError(
                    "last_batch must be one of 'keep', 'discard', or 'rollover', " \
                    "but got %s"%self._last_batch)

    def __len__(self):
        if self._last_batch == 'keep':
            return (len(self._sampler) + self._batch_size - 1) // self._batch_size
        if self._last_batch == 'discard':
            return len(self._sampler) // self._batch_size
        if self._last_batch == 'rollover':
            return (len(self._prev) + len(self._sampler)) // self._batch_size
        raise ValueError(
            "last_batch must be one of 'keep', 'discard', or 'rollover', " \
            "but got %s"%self._last_batch)
