# Author: Kamaldinov IR, kamildraf@gmail.com

# gener -- is GENERATE, i.e. gener_a is
# mapping form A to B


import torch
import torch.nn.functional as F
from utl import aduc


def compute_loss(gener_a, gener_b, 
                 discr_a, discr_b,
                 batch_a, batch_b,
                 alpha,
                 discr_loss = 'mse',
                 # cyclic_opposite = 0,
                 use_gpu=None,):
    if discr_loss.lower() == 'mse':
        discr_loss = F.mse_loss
    elif discr_loss.lower() == 'bce':
        discr_loss = F.binary_cross_entropy

    # outputs
    discr_a_onrly = discr_a(batch_a)
    discr_b_onrly = discr_b(batch_b)

    fake_a = aduc(gener_a(batch_b), use_gpu)
    fake_b = aduc(gener_b(batch_a), use_gpu)

    discr_a_onfke = discr_a(fake_a)
    discr_b_onfke = discr_b(fake_b)
    
    cyclic_a = aduc(gener_a(fake_b), use_gpu)
    cyclic_b = aduc(gener_a(fake_a), use_gpu)

    # parts of discrimentators loss
    discr_a_rly_loss = discr_loss(
        discr_a_onrly,
        torch.ones_like(discr_a_onrly))
    discr_b_rly_loss = discr_loss(
        discr_b_onrly,
        torch.ones_like(discr_b_onrly))

    discr_a_fke_loss = discr_loss(
        discr_a_onfke,
        torch.zeros_like(discr_a_onfke))
    discr_b_fke_loss = discr_loss(
        discr_b_onfke,
        torch.zeros_like(discr_a_onfke))

    # discrementators loss
    discr_a_loss = discr_a_rly_loss + discr_a_fke_loss
    discr_b_loss = discr_b_rly_loss + discr_b_fke_loss

    # generators fool loss
    gener_a_fool = F.mse_loss(
        discr_a_onfke,
        torch.ones_like(discr_a_onfke))
    gener_b_fool = F.mse_loss(discr_b_onfke,
        torch.ones_like(discr_b_onfke))

    # cyclic loss
    gener_a_cyc_loss = F.l1_loss(cyclic_a, batch_a)
    gener_b_cyc_loss = F.l1_loss(cyclic_b, batch_b)

    # generators loss
    gener_a_loss = gener_a_fool + alpha * gener_a_cyc_loss #+ cyclic_opposite * gener_b_cyc_loss
    gener_b_loss = gener_b_fool + alpha * gener_b_cyc_loss #+ cyclic_opposite * gener_a_cyc_loss

    return discr_a_loss, discr_b_loss, gener_a_loss, gener_b_loss
