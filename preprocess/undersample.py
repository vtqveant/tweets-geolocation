"""
As the initial dataset is hugely imbalanced, and the main task for optimization is a form of classification,
the dataset is under-sampled to contain up to 10k entries for a particular location.

The resulting data contains about 3.8M records, which is then split into training and testing sets
in proportion 10:1 (35 resulting files go to the testing set, remaining 353 to the training set).
"""

import csv
from os import listdir
from os.path import isfile, join
import random


def parse_coordinates(filename):
    idx = filename.find('_')
    lat = filename[0:idx]
    lon = filename[idx + 1:filename.find('.csv')]
    return lat, lon


if __name__ == '__main__':
    source_directory = '../data'
    target_directory = '../undersample_data'
    max_entries_per_file = 10000

    filenames = [f for f in listdir(source_directory) if isfile(join(source_directory, f))]

    target = []
    for filename in filenames:
        lat, lon = parse_coordinates(filename)
        with open(join(source_directory, filename), 'r') as f:
            source = csv.DictReader(f, delimiter=';')
            counter = 0
            for row in source:
                if counter == max_entries_per_file:
                    break
                r = {
                    'text': row['text'],
                    'lang': row['lang'],
                    'geo_country_code': row['geo_country_code'],
                    'lat': lat,
                    'lon': lon
                }
                target.append(r)
                counter += 1

    random.shuffle(target)

    counter = 0
    chunks = [target[i: i + max_entries_per_file] for i in range(0, len(target), max_entries_per_file)]
    for chunk in chunks:
        target_filename = str(counter) + '.csv'
        with open(join(target_directory, target_filename), 'w', newline='') as outcsv:
            writer = csv.DictWriter(outcsv, fieldnames=['text', 'lang', 'geo_country_code', 'lat', 'lon'], delimiter=';')
            writer.writeheader()
            for r in chunk:
                writer.writerow(r)
        counter += 1
