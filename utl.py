import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict


def linear_size(output):
    output_size = np.array(output.size())
    h, w = output_size[2], output_size[3]
    size = int(h * w)
    return size


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def train_stage(*args):
    for arg in args:
        arg.train()


def create_checkpoint(*args):
    checkpoint = OrderedDict()
    for net in args:
        name = net.__doc__
        checkpoint[name] = net.state_dict()
    return checkpoint


def exp_moving_mean(data, window=250):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev**(n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def visualize_loss(da_loss_log, db_loss_log,
                   ga_loss_log, gb_loss_log,
                   exp_window=None):
    if exp_window is not None:
        da_loss_log = exp_moving_mean(da_loss_log, exp_window)
        db_loss_log = exp_moving_mean(db_loss_log, exp_window)
        ga_loss_log = exp_moving_mean(ga_loss_log, exp_window)
        gb_loss_log = exp_moving_mean(gb_loss_log, exp_window)

    plt.figure(figsize=(10, 4))
    plt.tight_layout()

    plt.subplot(1, 2, 1)
    plt.plot(ga_loss_log, label="gener_a")
    plt.plot(gb_loss_log, label="gener_b")
    plt.xlabel("train step")
    plt.ylabel("MSE")
    plt.title("generators loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(da_loss_log, label="discr_a")
    plt.plot(db_loss_log, label="discr_b")
    plt.tight_layout()
    plt.xlabel("train step")
    plt.ylabel("MSE")
    plt.title("discriminators loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_geners(sample_a, sample_b,
                gener_a, gener_b):
    gener_a.eval()
    gener_b.eval()

    plt.figure(figsize=(8, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(sample_b.cpu().view(1, 1, 28, 28).data[0]
               .numpy().reshape((28, 28)),
               cmap='binary')
    plt.title("b")

    plt.subplot(2, 3, 2)
    plt.imshow(gener_a(sample_b.view(1, 1, 28, 28)).cpu().data[0]
               .numpy().reshape((28, 28)),
               cmap='binary')
    plt.title("gener_a(b)")

    plt.subplot(2, 3, 3)
    plt.imshow(gener_b(gener_a(sample_b.view(1, 1, 28, 28))).cpu().data[0]
               .numpy().reshape((28, 28)),
               cmap='binary')
    plt.title("gener_b(gener_a(b))")

    plt.subplot(2, 3, 4)
    plt.imshow(sample_a.cpu().view(1, 1, 28, 28).data[0]
               .numpy().reshape((28, 28)),
               cmap='binary')
    plt.title("a")

    plt.subplot(2, 3, 5)
    plt.imshow(gener_b(sample_a.view(1, 1, 28, 28)).cpu().data[0]
               .numpy().reshape((28, 28)),
               cmap='binary')
    plt.title("gener_b(a)")

    plt.subplot(2, 3, 6)
    plt.imshow(gener_a(gener_b(sample_a.view(1, 1, 28, 28))).cpu().data[0]
               .numpy().reshape((28, 28)),
               cmap='binary')
    plt.title("gener_a(gener_b(a))")

    plt.tight_layout()
    plt.show()


def grad_norm(model, norm_type=2):
    total_norm = 0
    for param in model.parameters():
        param_norm = param.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def plot_grad_norms(da_grad_log, db_grad_log,
                    ga_grad_log, gb_grad_log):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(ga_grad_log, label="gener_a")
    plt.plot(gb_grad_log, label="gener_b")
    plt.xlabel("step")
    plt.ylabel("grad norm")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(da_grad_log, label="disrc_a")
    plt.plot(db_grad_log, label="discr_b")
    plt.xlabel("step")
    plt.ylabel("grad norm")
    plt.legend()

    plt.tight_layout()
    plt.show()
