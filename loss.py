# Author: Kamaldinov IR, kamildraf@gmail.com

# gener -- is GENERATE, i.e. gener_a is
# mapping form A to B


import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utl import aduc


def compute_loss(gener_a, gener_b, 
                 discr_a, discr_b,
                 batch_a, batch_b,
                 alpha,
                 discr_loss = 'mse',
                 use_gpu=None,):
    if discr_loss.lower() == 'mse':
        discr_loss = F.mse_loss
    elif discr_loss.lower() == 'bce':
        discr_loss = F.binary_cross_entropy

    # outputs
    discr_a_onrly = discr_a(batch_a)
    discr_b_onrly = discr_b(batch_b)

    fake_a = gener_a(batch_b)
    fake_b = gener_b(batch_a)

    discr_a_onfke = discr_a(fake_a)
    discr_b_onfke = discr_b(fake_b)
    
    cyclic_a = gener_a(fake_b)
    cyclic_b = gener_b(fake_a)

    # parts of discrimentators loss
    discr_a_rly_loss = discr_loss(
        discr_a_onrly,
        aduc(torch.ones_like(discr_a_onrly)), use_gpu)
    discr_b_rly_loss = discr_loss(
        discr_b_onrly,
        aduc(torch.ones_like(discr_b_onrly)), use_gpu)

    discr_a_fke_loss = discr_loss(
        discr_a_onfke,
        aduc(torch.zeros_like(discr_a_onfke)), use_gpu)
    discr_b_fke_loss = discr_loss(
        discr_b_onfke,
        aduc(torch.zeros_like(discr_a_onfke)), use_gpu)

    # discrementators loss
    discr_a_loss = discr_a_rly_loss + discr_a_fke_loss
    discr_b_loss = discr_b_rly_loss + discr_b_fke_loss

    # generators fool loss
    gener_a_fool = F.mse_loss(
        discr_a_onfke,
        aduc(torch.ones_like(discr_a_onfke)), use_gpu)
    gener_b_fool = F.mse_loss(discr_b_onfke,
        aduc(torch.ones_like(discr_b_onfke)), use_gpu)

    # cyclic loss
    gener_a_cyc_loss = F.l1_loss(cyclic_a, batch_a)
    gener_b_cyc_loss = F.l1_loss(cyclic_b, batch_b)

    # generators loss
    gener_a_loss = gener_a_fool + alpha * gener_a_cyc_loss
    gener_b_loss = gener_b_fool + alpha * gener_b_cyc_loss

    return discr_a_loss, discr_b_loss, gener_a_loss, gener_b_loss

def compute_combined_loss(gener_a, gener_b, 
                          discr_a, discr_b,
                          batch_a, batch_b,
                          alpha,
                          discr_loss = 'mse',
                          use_gpu=None):
    if discr_loss.lower() == 'mse':
        discr_loss = F.mse_loss
    elif discr_loss.lower() == 'bce':
        discr_loss = F.binary_cross_entropy

    # outputs
    discr_a_onrly = discr_a(batch_a)
    discr_b_onrly = discr_b(batch_b)

    fake_a = gener_a(batch_b)
    fake_b = gener_b(batch_a)

    discr_a_onfke = discr_a(fake_a)
    discr_b_onfke = discr_b(fake_b)
    
    cyclic_a = gener_a(fake_b)
    cyclic_b = gener_b(fake_a)

    # parts of discrimentators loss
    discr_a_rly_loss = discr_loss(
        discr_a_onrly,
        aduc(torch.ones_like(discr_a_onrly)))
    discr_b_rly_loss = discr_loss(
        discr_b_onrly,
        aduc(torch.ones_like(discr_b_onrly)))

    discr_a_fke_loss = discr_loss(
        discr_a_onfke,
        aduc(torch.zeros_like(discr_a_onfke)))
    discr_b_fke_loss = discr_loss(
        discr_b_onfke,
        aduc(torch.zeros_like(discr_a_onfke)))

    # discrementators loss
    discr_a_loss = discr_a_rly_loss + discr_a_fke_loss
    discr_b_loss = discr_b_rly_loss + discr_b_fke_loss

    discr_loss = (discr_a_loss + discr_b_loss) / 2

    # generators fool loss
    gener_a_fool = F.mse_loss(
        discr_a_onfke,
        aduc(torch.ones_like(discr_a_onfke), use_gpu))
    gener_b_fool = F.mse_loss(
        discr_b_onfke,
        aduc(torch.ones_like(discr_b_onfke), use_gpu))

    # cyclic loss
    gener_a_cyc_loss = F.l1_loss(cyclic_a, aduc(Variable(batch_a.data)))
    gener_b_cyc_loss = F.l1_loss(cyclic_b, aduc(Variable(batch_b.data)))

    # generators loss
    gener_a_loss = gener_a_fool + alpha * gener_a_cyc_loss
    gener_b_loss = gener_b_fool + alpha * gener_b_cyc_loss

    gener_loss = gener_a_loss + gener_b_loss

    losses = [discr_loss, gener_loss,
              discr_a_loss, discr_b_loss,
              gener_a_loss, gener_b_loss,
              gener_a_fool, gener_b_fool]

    return losses

def consensus_loss(loss, net):
    grad_params = torch.autograd.grad(loss,
                                      net.parameters(),
                                      create_graph=True)
    grad_norm = 0
    for grad in grad_params:
        grad_norm += grad.pow(2).sum()
    return grad_norm