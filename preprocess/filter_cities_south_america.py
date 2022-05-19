import pandas as pd

df = pd.read_csv('../worldcities/simplemaps_worldcities_basicv1.75/worldcities.csv')

# BBox for South America
lat_min = -56.0
lat_max = 13.0
lon_min = -82.0
lon_max = -33.0

cities_south_america = df[(df['lat'] > lat_min) & (df['lat'] < lat_max) & (df['lng'] > lon_min) & (df['lng'] < lon_max)]
cities_south_america[['city', 'lat', 'lng', 'population']].to_csv('cities_south_america.csv', sep=',')

