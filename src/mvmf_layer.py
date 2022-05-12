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
        self._num_distributions = num_distributions

        self.kappa = nn.Parameter(torch.Tensor(num_distributions))
        self.mu = nn.Parameter(torch.Tensor(num_distributions, 3))  # each mu must be of length 1, i.e. $||\mu_i||_2 = 1$

        self.fc1 = nn.Linear(in_features=in_features, out_features=self._num_distributions)

    def forward(self, raw_weights, euclidean_coord):
        w = self.fc1(raw_weights)
        w = F.softmax(w, dim=0)

        coeff = torch.div(self.kappa, torch.sinh(self.kappa))
        inner = torch.matmul(self.mu, torch.transpose(euclidean_coord, 0, 1))
        inner = torch.transpose(inner, 0, 1)
        m = torch.mul(self.kappa, inner)
        exponent = torch.exp(m)
        vmf = torch.mul(coeff, exponent)
        mvmf = torch.sum(torch.mul(w, vmf), dim=1)
        return torch.neg(torch.log(mvmf))


def init_mvmf_weights(module):
    """
    This function should be passed to nn.Module.apply()

        model = MvMFLayer(in_features=10, num_distributions=7)
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
