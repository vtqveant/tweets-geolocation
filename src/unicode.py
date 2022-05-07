from os import listdir
from os.path import isfile, join
import csv

path = "../data"
filenames = [f for f in listdir(path) if isfile(join(path, f))]

# a subset to play with
filenames = filenames[0:10]

# print(filenames)


class Coordinate:
    """A class to represent a point on the Earth"""

    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def __str__(self):
        return lat + ', ' + lon


entries = {}
field_counts = {}
for filename in filenames:
    idx = filename.find('_')
    lat = filename[0:idx]
    lon = filename[idx + 1:filename.find('.csv')]
    coord = Coordinate(lat, lon)
    if coord not in entries:
        entries[coord] = []
    with open('../data/' + filename, newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            # entries[coord].append(row)
            for key in row.keys():
                if key in field_counts:
                    field_counts[key] += 1
                else:
                    field_counts[key] = 1

# some_key = list(entries.keys())[0]
# some_entry = entries[some_key]
# print(some_key)
# print(some_entry)
# print(some_entry[0].keys())

print(field_counts)