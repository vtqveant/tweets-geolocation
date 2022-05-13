import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geometry import to_euclidean


class MvMFLayer(nn.Module):
    """This works more like a loss, i.e. it evaluates an expression to be minimized given a point on a sphere
    (e.g. using a MAE w.r.t. zero), but it does not infer any points itself. A prediction will be done afterwards using
    the parameters trained with this network."""

    def __init__(self, in_features, num_distributions):
        super().__init__()
        self.kappa = nn.Parameter(torch.Tensor(num_distributions))
        self.mu = nn.Parameter(torch.Tensor(num_distributions, 3))  # each mu must be of length 1, i.e. $||\mu_i||_2 = 1$
        self.fc1 = nn.Linear(in_features=in_features, out_features=num_distributions)

    def forward(self, weights, euclidean_coord):
        vmf_weights = self.fc1(weights)
        vmf_weights = F.softmax(vmf_weights, dim=1)

        d = torch.transpose(torch.matmul(self.mu, torch.transpose(euclidean_coord, 0, 1)), 0, 1)
        exponent = torch.exp(torch.mul(self.kappa, d))

        coeff = torch.div(self.kappa, torch.sinh(self.kappa))
        denom = torch.full_like(coeff, 4 * 3.1415926536)
        coeff = torch.div(coeff, denom)

        vmf = torch.mul(coeff, exponent)
        mvmf = torch.sum(torch.mul(vmf_weights, vmf), dim=1)
        return mvmf


def MvMF_loss(output, target):
    """The loss (negative logarithm of a weighted sum of vMF components) equals zero when the MvMF is
    equal to 1, in which case MvMF can be interpreted as a probability distribution over separate vMFs treated
    as classes (probability of a tweet being submitted from a particular city, if vMFs are initialized with
    coordinates of cities)."""
    return F.l1_loss(torch.neg(torch.log(output)), target)


def init_mvmf_weights(module):
    """
    This function should be passed to nn.Module.apply()

        model = MvMFLayer(in_features=1024, num_distributions=10000)
        model.apply(init_mvmf_weights)
    """
    if isinstance(module, MvMFLayer):
        # value from the paper empirically observed to correspond to a st.d. about the size of a large city
        module.kappa.data.fill_(10.0)

        # TODO pick coordinates of most populated cities
        cs = []
        for i in range(module.mu.data.size(dim=0)):
            c = to_euclidean(np.random.randint(-90, 90), np.random.randint(-180, 180))
            cs.append(c)
        module.mu.data.copy_(torch.tensor(np.array(cs)))


def unit_norm_mu_clipper(module):
    """
    This function should be passed to nn.Module.apply() after optimizer.step()

    Keeps norm2(mu) == 1

    https://discuss.pytorch.org/t/restrict-range-of-variable-during-gradient-descent/1933/3
    """
    if isinstance(module, MvMFLayer) and hasattr(module, 'mu'):
        w = module.mu.data
        w.div_(torch.linalg.vector_norm(w, 2, 1).reshape(-1, 1).expand_as(w))
