import torch
import torch.nn as nn
import numpy as np


# kappa = nn.Parameter(torch.Tensor(2))
# print(kappa.shape)
#
# mu = nn.Parameter(torch.Tensor(2, 3))
# print(mu.shape)

# need to convert shape (2, 1) to (2)
t = torch.Tensor(np.array([[1], [2]]))
print(t)
print(t.shape)

r = torch.reshape(t, (-1,))
print(r)
print(r.shape)