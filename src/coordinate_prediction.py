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


def predict_coord_grid_search(snapshot, text, num_lat_samples, num_lon_samples):
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
            _, _, score = model(features, coordinates)

            # join coordinates with MvMF scores to get a list of (x, y, z, score) entries
            b = torch.cat([coordinates, torch.reshape(score, (-1, 1))], dim=1).numpy()
            results.extend(b)

    return np.array(results)


def main():
    results = predict_coord_grid_search(
        '../snapshots/16-05-2022_03:50:17_large.pth',
        '😉😁 “La vida es un viaje y quien viaja vive dos veces”.  – Omar Khayyam https://t.co/TPvK5BYZ2x;1429968360448610313',
        num_lat_samples=100,
        num_lon_samples=100
    )
    print(results)


if __name__ == '__main__':
    main()
