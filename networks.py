import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


#dim = None
#dropout = None
#norm_lay = None
activation = nn.LeakyReLU(0.1)


# base resnet block
# conv > ?dropout > conv
# possible params
# kernel_size, padding, bias, reflection type
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_lay, dropout):
        super().__init__()
        self.conv_seq = self.conv_block(dim, norm_lay, dropout)

    def conv_block(self, dim, norm_lay, dropout):
        conv_block = []

        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            norm_lay(dim),
            activation
        ]
        
        
        if dropout is not None:
            conv_block += [nn.Dropout(dropout)]

        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            norm_lay(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, input):
        output = input + self.conv_seq(input)
        return output

# generative block via resnet
# conv > downconv* > resnet block* > transp conv* > conv
# possible params
# first/last conv: ksize, pad, stride
# down/trans conv: number, conv params
class ResnetGenerator(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 gen_filters,
                 norm_lay,
                 dropout,
                 n_blocks):
        super().__init__()
        self._input_nc = input_nc
        self._output_nc = output_nc
        self._gen_filters = gen_filters

        resnet_seq = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, gen_filters, kernel_size=7, padding=0),
            norm_lay(gen_filters),
            activation
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            power = 2 ** i
            resnet_seq += [
                nn.Conv2d(
                    gen_filters * power,
                    gen_filters * power * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1
                    ),
                norm_lay(gen_filters * power * 2),
                activation
            ]

        power = 2 ** n_downsampling
        for i in range(n_blocks):
            resnet_seq += [
                ResnetBlock(
                    gen_filters * power,
                    norm_lay,
                    dropout)
                ]

        for i in range(n_downsampling):
            power = 2 ** (n_downsampling - i)
            resnet_seq += [
                nn.ConvTranspose2d(
                    gen_filters * power,
                    int(gen_filters * power / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                    ),
                norm_lay(int(gen_filters * power / 2)),
                activation
                ]

        resnet_seq += [nn.ReflectionPad2d(3)]
        resnet_seq += [nn.Conv2d(gen_filters, output_nc,
                                 kernel_size=7,
                                 padding=0)]
        resnet_seq += [nn.Tanh()]

        self._resnet_seq = nn.Sequential(*resnet_seq)

    def forward(self, input):
        output = self._resnet_seq(input)
        return output

# almost magic
class GANLoss(nn.Module):
    def __init__(self,
                 MSE=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if MSE:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# conv > conv* > conv
# possible params
# kernel_sizes, paddings, strides, actiavations, norm_lay, min_filt
# improvements: dropout
class Discrimanator(nn.Module):
    def __init__(self,
                 input_nc,
                 discr_filters,
                 max_power,
                 n_layers,
                 norm_lay):
        super().__init__()

        ksize = 4
        strd = 1
        pad = 1
        discr_seq = [
            nn.Conv2d(input_nc, discr_filters,
                      kernel_size=ksize,
                      stride=strd,
                      padding=pad),
            norm_lay(discr_filters),
            activation    
        ]

        filters = np.linspace(discr_filters,
                              discr_filters * max_power,
                              num=n_layers,
                              dtype=int).tolist()

        prev_filter = discr_filters
        for interm_filter in filters[1:]:
            discr_seq += [
                nn.Conv2d(
                    prev_filter,
                    interm_filter,
                    kernel_size=ksize,
                    stride=strd,
                    padding=pad
                    ),
                norm_lay(interm_filter),
                activation
            ]
            prev_filter = interm_filter

        discr_seq += [
            nn.Conv2d(
                prev_filter,
                1,
                kernel_size=ksize,
                stride=strd,
                padding=pad
                )
            ]

        self.model = nn.Sequential(*discr_seq)

    def forward(self, input):
        output = self.model(input)
        return output