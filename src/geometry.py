import math


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


class Coordinate:
    """A class to represent a point on the Earth"""

    def __init__(self, lat, lon):
        self.lat = float(lat)
        self.lon = float(lon)
