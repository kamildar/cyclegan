import numpy as np
import torch
from collections import OrderedDict

def linear_size(output):
    output_size = np.array(output.size())
    h, w = output_size[2], output_size[3]
    size = int(h * w)
    return size


conv_normal_mean = 0.0
conv_normal_sd = 0.02
bnorm_mean = 1.0
bnorm_sd = 0.02
bnorm_fill = 0

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(conv_normal_mean,
                              conv_normal_sd)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(bnorm_mean,
                              bnorm_sd)
        m.bias.data.fill_(bnorm_fill)


def aduc(x, use_gpu=None):
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    if use_gpu:
        return x.cuda()
    else:
        return x

def set_names(discr_a, discr_b,
              gener_a, gener_b,
              opt_gener_a, opt_gener_b,
              opt_discr_a, opt_discr_b):
    discr_a.__doc__ = 'discr_a'
    discr_b.__doc__ = 'discr_b'
    gener_a.__doc__ = 'gener_a'
    gener_b.__doc__ = 'gener_b'
    opt_gener_a.__doc__ = 'opt_gener_a'
    opt_gener_b.__doc__ = 'opt_gener_b'
    opt_discr_a.__doc__ = 'opt_discr_a'
    opt_discr_b.__doc__ = 'opt_discr_b'
    pass

def create_checkpoint(*args):
    checkpoint = OrderedDict()
    for net in args:
        name = net.__doc__
        checkpoint[name] = net.state_dict()
    return checkpoint
