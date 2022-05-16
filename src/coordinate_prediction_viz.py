import math
import random
import matplotlib.pyplot as plt

from coordinate_prediction import predict_coord_grid_search
from geometry import to_geographical


def main():
    results = predict_coord_grid_search(
        '../snapshots/16-05-2022_03:50:17_large.pth',
        'Diego Carrasco? Como decía el Bonva, cuando era weno era malo!',
        num_lat_samples=100,
        num_lon_samples=150
    )

    xs, ys, ss = [], [], []
    for result in results:
        if random.random() > 0.5:
            lat, lon = to_geographical(result[:3])
            score = result[3]
            xs.append(lon)
            ys.append(lat)
            ss.append(1/(1 + math.exp(1 - score)))

    plt.scatter(x=xs, y=ys, c=ss, s=0.5, cmap='hot')
    plt.show()


if __name__ == '__main__':
    main()
