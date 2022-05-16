import torch
import torch.nn as nn
import torch.nn.functional as F


class MvMFLayer(nn.Module):
    """
    This works more like a loss, i.e. it evaluates an expression to be minimized given a point on a sphere
    (e.g. using a MAE w.r.t. zero), but it does not infer any points itself. A prediction will be done afterwards using
    the parameters trained with this network.
    """

    def __init__(self, in_features, num_distributions):
        super().__init__()
        self.kappa = nn.Parameter(torch.Tensor(num_distributions))
        self.mu = nn.Parameter(
            torch.Tensor(num_distributions, 3))  # each mu must be of length 1, i.e. $||\mu_i||_2 = 1$
        self.fc1 = nn.Linear(in_features=in_features, out_features=num_distributions)

    def forward(self, weights, euclidean_coord):
        weights = self.fc1(weights)
        vmf_weights = F.softmax(weights, dim=1)

        d = torch.matmul(euclidean_coord, torch.transpose(self.mu, 0, 1))
        exponent = torch.exp(torch.mul(self.kappa, d))

        coeff = torch.div(self.kappa, torch.sinh(self.kappa))
        coeff = coeff.div_(4 * 3.1415926536)

        vmf = torch.mul(coeff, exponent)
        t = torch.mul(vmf_weights, vmf)
        mvmf = torch.sum(t, dim=1)
        return mvmf


def MvMF_loss(output, target):
    """
    The loss (negative logarithm of a weighted sum of vMF components) equals zero when the MvMF is
    equal to 1, in which case MvMF can be interpreted as a probability distribution over separate vMFs treated
    as classes (probability of a tweet being submitted from a particular city, if vMFs are initialized with
    coordinates of cities).
    """
    return F.mse_loss(torch.neg(torch.log(output)), target)
