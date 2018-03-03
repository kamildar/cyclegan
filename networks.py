import torch.nn as nn
import numpy as np
import torch.nn.functional as F



#dim = None
#dropout = None
#norm_lay = None
activation = nn.LeakyReLU(0.1)
bias = False

# for square images
def conv_size(input_size, kernel_size, stride, padding):
    out = int((input_size + 2 * padding - kernel_size) / stride) + 1
    return out

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
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=bias),
            norm_lay(dim),
            activation
        ]
        
        
        if dropout is not None:
            conv_block += [nn.Dropout(dropout)]

        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=bias),
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
            nn.Conv2d(input_nc, gen_filters, kernel_size=7, padding=0, bias=bias),
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
                    padding=1,
                    bias=bias
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
                                 padding=0,
                                 bias=bias)]
        resnet_seq += [nn.Tanh()]

        self._resnet_seq = nn.Sequential(*resnet_seq)

    def forward(self, input):
        output = self._resnet_seq(input)
        return output


class Discriminator(nn.Module):
    def __init__(self,
                 input_nc,
                 discr_filters,
                 max_power,
                 n_layers,
                 norm_lay,
                 start_size,
                 out_linear=10):
        super().__init__()
        self._input_nc = input_nc

        size = start_size
        ksize = 4
        strd = 1
        pad = 1
        discr_body = [
            nn.Conv2d(input_nc, discr_filters,
                      kernel_size=ksize,
                      stride=strd,
                      padding=pad),
            norm_lay(discr_filters),
            activation
        ]

        filters = np.linspace(discr_filters,
                              discr_filters * max_power,
                              num=n_layers+1,
                              dtype=int).tolist()

        prev_filter = discr_filters
        for interm_filter in filters[1:]:
            discr_body += [
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

        discr_body += [
            nn.Conv2d(
                prev_filter,
                1,
                kernel_size=ksize,
                stride=strd,
                padding=pad
                )
            ]

        for i in range(n_layers + 2):
            size = conv_size(size, ksize, strd, pad)
        self._linear_size = size * size

        discr_head = [nn.Linear(size * size, out_linear)]
        if out_linear != 1:
            discr_head += [nn.Linear(out_linear, 1)]

        self.body = nn.Sequential(*discr_body)
        self.head = nn.Sequential(*discr_head)

    def forward(self, input):
        body_out = self.body(input)
        output = self.head(body_out.view(-1, self._linear_size))
        return F.sigmoid(output)