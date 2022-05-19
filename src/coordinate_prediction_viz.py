import math
import random
import matplotlib.pyplot as plt

from coordinate_prediction import predict_coord_grid_search
from geometry import to_geographical


def main():
    results = predict_coord_grid_search(
        '../snapshots/19-05-2022_09:16:16.pth',
        # '@JLGamboa_Velez @MaytaTristan @CayetanoHeredia @INS_Peru Jorge, si te fijas en las redes, esa opinión',
        # 'Um cabelo arrumado quem não gosta😍',
        'Essa máquina é do meu irmão em Cristo @glenio.josafa @vera.corcino em Vilhena, Rondonia, Brazil',
        # '@___nobodyknows Ayyy yo estoy caducada ya desde el 2018 y no renuevo 😢',
        num_lat_samples=100,
        num_lon_samples=100
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
