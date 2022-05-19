from typing import List
import numpy as np
import torch

import character_encoder
from character_encoder import CharacterEncoder
from geometry import to_euclidean
from unicode_cnn import UnicodeCNN


# TODO refac (s. dataset_processor.py)
def convert_unicode_features_to_tensor(matrix):
    """a matrix is actually a list of bytestring consisting of b'0' and b'1', so we need to offset by 48"""
    return torch.transpose(torch.tensor([[float(i - 48) for i in s] for s in matrix], dtype=torch.float32), 0, 1)


def predict_coord_center_of_mass(model: UnicodeCNN, texts: List[str]):
    """
    Computes a center of mass of top-n $\\mu$-s of an MvMF layer using vMF weights computed for the text.
    """
    encoder = CharacterEncoder(character_encoder.ENCODING_SIZE_SMALL)
    unicode_features = torch.stack([convert_unicode_features_to_tensor(encoder.encode(t)) for t in texts])

    model.eval()
    with torch.no_grad():
        # predict w.r.t. to dummy coordinates 'cause vMF weights do not depend on coordinates
        dummy_coordinates = torch.zeros([len(texts), 3])
        _, _, score, vmf_weights = model(unicode_features, dummy_coordinates)

    result = []
    weights = vmf_weights.detach().numpy()
    mu = model.get_parameter('mvmf.mu').detach()
    top_n = 1
    for w in weights:
        indices = np.argpartition(w, -top_n)[-top_n:]
        weights_top_n = w[indices]
        mu_top_n = mu[indices]
        center_of_mass = np.average(mu_top_n, axis=0, weights=weights_top_n)
        result.append(center_of_mass)

    return result


def predict_coord_grid_search(snapshot: str, text: str, num_lat_samples: int, num_lon_samples: int):
    """
    Computes a grid of scores for visualization
    """

    # 1. define BBox for South America
    lat_min = -56.0
    lat_max = 13.0
    lon_min = -82.0
    lon_max = -33.0

    # 2. load model
    model = UnicodeCNN()
    model.load_state_dict(torch.load(snapshot))
    model.eval()

    # 3. unicode encoder
    encoder = CharacterEncoder(character_encoder.ENCODING_SIZE_SMALL)
    unicode_features = convert_unicode_features_to_tensor(encoder.encode(text))
    features = torch.stack([unicode_features] * num_lon_samples)

    # 4. sample coordinates, batch along longitude axis and compute a list of (x, y, z, score) entries
    results = []
    with torch.no_grad():
        for lat in np.linspace(lat_min, lat_max, num=num_lat_samples):
            euclidean_coordinates = []
            for lon in np.linspace(lon_min, lon_max, num=num_lon_samples):
                euclidean_coordinates.append(torch.tensor(to_euclidean(lat, lon), dtype=torch.float32))

            # create a batch from all points sampled from the longitude range
            coordinates = torch.stack(euclidean_coordinates)
            _, _, score, _ = model(features, coordinates)

            # join coordinates with MvMF scores to get a list of (x, y, z, score) entries
            b = torch.cat([coordinates, torch.reshape(score, (-1, 1))], dim=1).numpy()
            results.extend(b)

    return np.array(results)
