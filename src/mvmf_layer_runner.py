"""
Usage example for MvMFLayer
"""

import torch
import numpy as np
from mvmf_layer import MvMFLayer, MvMF_loss
from geometry import to_euclidean, to_geographical, norm2
from train import init_mvmf_weights


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
    model.apply(init_mvmf_weights)

    # the layer combines two inputs
    # a point on $S^2$ converted to euclidean coordinates...
    in1 = np.array(to_euclidean(0.508498, -73.8294))
    in2 = np.array(to_euclidean(-10.5, 34.342))
    eucl_coord = torch.tensor(np.array([in1, in2]), dtype=torch.float32)

    # and the weights to be passed through a softmax
    weights = torch.tensor(np.array([
        [0.5, 0.1, 0.1, 0.2, 0., 0., 0.1, 0., 0., 0.],
        [1.5, 0.1, 0.1, -2., 0., 0., 0.1, 0., 3., 0.]
    ]), dtype=torch.float32)

    # the output is (should be) the probability of these coordinates w.r.t. to MvMF given weights
    mvmf_score, vmf_weights = model(weights, eucl_coord)
    print('Forward pass:', mvmf_score)

    target = torch.zeros(len(eucl_coord))
    loss = MvMF_loss(mvmf_score, target)
    loss.backward()

    print('MvMF result: ', mvmf_score)
    print('loss: ', loss)


if __name__ == '__main__':
    main()
