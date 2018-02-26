# gener -- is GENERATE, i.e. gener_a is
# mapping form A to B

# instances:
# gener_a - from B to A
# gener_b - from A to B
# discr_a from A find A, should return 1
# discr_b from B find B, should return 1
# discr_gen_a - from gen_a(B) find B, fool it (return 0)
# discr_gen_b - from gen_b(A) find A, fool it (return 0)
# batch_one - A
# batch_two - B
import torch.nn.functional as F
import torch
from torch.autograd import Variable

def compute_loss(gener_a, gener_b, 
                 discr_a, discr_b,
                 batch_a, batch_b,
                 alpha):
    var_1 = Variable(torch.Tensor(1), requires_grad=False)
    var_0 = Variable(torch.Tensor(0), requires_grad=False)
    
    # outputs    
    discr_a_onrly = discr_a(batch_a)
    discr_b_onrly = discr_b(batch_b)

    fake_a = gener_a(batch_b)
    fake_b = gener_b(batch_a)

    discr_a_onfke = discr_a(fake_a)
    discr_b_onfke = discr_b(fake_b)
    
    cyclic_a = gener_a(fake_b)
    cyclic_b = gener_a(fake_a)

    # parts of discrimentators loss
    discr_a_rly_loss = F.mse_loss(discr_a_onrly, var_1)
    discr_b_rly_loss = F.mse_loss(discr_b_onrly, var_1)

    discr_a_fke_loss = F.mse_loss(discr_a_onfke, var_0)
    discr_b_fke_loss = F.mse_loss(discr_b_onfke, var_0)

    # discrementators loss
    discr_a_loss = discr_a_rly_loss + discr_a_fke_loss
    discr_b_loss = discr_b_rly_loss + discr_b_fke_loss

    # generators fool loss
    gener_a_fool = F.mse_loss(discr_a_onfke, 1)
    gener_b_fool = F.mse_loss(discr_b_onfke, 1)

    # cyclic loss
    gener_a_cyc_loss = F.l1_loss(batch_a, cyclic_a)
    gener_b_cyc_loss = F.l1_loss(batch_b, cyclic_b)

    # generators loss
    gener_a_loss = gener_a_fool + alpha * gener_a_cyc_loss
    gener_b_loss = gener_b_fool + alpha * gener_b_cyc_loss

    return discr_a_loss, discr_b_loss, gener_a_loss, gener_b_loss
