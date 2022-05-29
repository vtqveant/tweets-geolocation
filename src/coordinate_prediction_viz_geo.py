import math

from plotly.graph_objs import Layout
from plotly import offline
from coordinate_prediction import Predictor
from geometry import to_geographical


def plot(lats, lons, scores):
    data = [{
        'type': 'scattergeo',
        'lon': lons,
        'lat': lats,
        'marker': {
            'size': 8.0,
            'color': scores,
            'opacity': 0.5,
            'colorscale': 'Inferno',
            'reversescale': True,
            'colorbar': {'title': 'Brightness'},
        },
    }]
    my_layout = Layout(title='South America Tweet Geolocation')

    fig = {'data': data, 'layout': my_layout}
    offline.plot(fig, filename='south_america_tweet_geolocation.html')


def main():
    predictor = Predictor()
    results = predictor.predict_coord_grid_search(
        '../snapshots/weights.pth',
        '*#19Feb-* La Alcaldía Bolivariana del municipio Bermúdez, liderada por Nircia Villegas, a través de la Dirección de Infraestructura',
        num_lat_samples=60,
        num_lon_samples=60
    )

    lats, lons, scores = [], [], []
    for result in results:
        lat, lon = to_geographical(result[:3])
        lats.append(lat)
        lons.append(lon)
        score = result[3]
        scores.append(math.exp(5 * score))

    plot(lats, lons, scores)


if __name__ == '__main__':
    main()
