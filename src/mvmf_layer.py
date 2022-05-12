import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MvMFLayer(nn.Module):

    def __init__(self, in_features, num_distributions):
        super().__init__()
        self._num_distributions = num_distributions

        self.kappa = nn.Parameter(torch.Tensor(num_distributions))
        self.mu = nn.Parameter(torch.Tensor(num_distributions, 3))  # each mu nust be of length 1, i.e. $||\mu_i||_2 = 1$

        self.fc1 = nn.Linear(in_features=in_features, out_features=self._num_distributions)

    def forward(self, raw_weights, euclidean_coord):
        w = self.fc1(raw_weights)
        w = F.softmax(w, dim=0)

        coeff = torch.div(self.kappa, torch.sinh(self.kappa))
        inner = torch.inner(self.mu, euclidean_coord)
        inner = torch.reshape(inner, (-1,))
        m = torch.mul(self.kappa, inner)
        exponent = torch.exp(m)
        vmf = torch.mul(coeff, exponent)
        mvmf = torch.sum(torch.mul(w, vmf), dim=0)
        return mvmf


def to_euclidean(lat, lon):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    x1 = math.cos(lat_rad) * math.cos(lon_rad)
    x2 = math.cos(lat_rad) * math.sin(lon_rad)
    x3 = math.sin(lat_rad)
    return [x1, x2, x3]


def to_geographical(x):
    lat_rad = math.asin(x[2])
    lon_rad = math.atan2(x[1], x[0])
    return math.degrees(lat_rad), math.degrees(lon_rad)


def norm2(x):
    return math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])


def init_weights(module):
    if isinstance(module, MvMFLayer):
        # value from the paper empirically observed to correspond to a st.d. about the size of a large city
        module.kappa.data.fill_(10.0)

        # TODO pick coordinates of most populated cities
        cs = []
        for i in range(module.mu.data.size(dim=0)):
            c = to_euclidean(np.random.randint(-90, 90), np.random.randint(-180, 180))
            cs.append(c)
        module.mu.data.copy_(torch.tensor(np.array(cs)))


def main():
    # https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates

    lat = 0.508498
    lon = -73.8294
    print(lat, lon)

    x = to_euclidean(lat, lon)
    print(x)
    print(norm2(x))

    lat, lon = to_geographical(x)
    print(lat, lon)

    # try the model
    torch.manual_seed(0)  # for repeatable results
    model = MvMFLayer(in_features=10, num_distributions=7)
    model.apply(init_weights)

    # the layer combines two inputs
    # a point on $S^2$ converted to euclidean coordinates...
    input = np.array(to_euclidean(lat, lon))
    eucl_coord = torch.tensor(np.array([input]), dtype=torch.float32)

    # and the weights to be passed through a softmax
    weights = torch.tensor(np.array([0.5, 0.1, 0.1, 0.2, 0., 0., 0.1, 0., 0., 0.]), dtype=torch.float32)

    # the output is the probability of these coordinates w.r.t. to MvMF given weights
    y = model(weights, eucl_coord)
    print('Forward pass:', y)


if __name__ == '__main__':
    main()
