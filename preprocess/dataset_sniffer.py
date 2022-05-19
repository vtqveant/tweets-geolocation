"""
This script scans an entire dataset to collect statistics and label dictionaries
"""

from os import listdir
from os.path import isfile, join
import csv
import json


langs = {}
countries = {}
filename_row_count = {}

path = '../data'
filenames = [f for f in listdir(path) if isfile(join(path, f))]

file_counter = 0
row_counter = 0
lang_counter = 0
country_counter = 0

for filename in filenames:
    with open(join(path, filename), newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        rows_in_file = 0
        for row in reader:
            row_counter += 1
            rows_in_file += 1
            if row['lang'] not in langs.keys():
                langs[row['lang']] = lang_counter
                lang_counter += 1
            if row['geo_country_code'] not in countries.keys():
                countries[row['geo_country_code']] = country_counter
                country_counter += 1
        filename_row_count[filename] = rows_in_file
    if file_counter % 300 == 0:
        print('{:.0f}% done'.format(100. * file_counter / len(filenames)))
    file_counter += 1


print('files (coordinates)', file_counter)
print('rows', row_counter)
print('langs', len(langs.keys()))
print('countries', len(countries.keys()))
print(langs)
print(countries)


# save for future analysis
with open('inca_dataset_stats.json', 'w') as f:
    f.write(json.dumps(filename_row_count))
with open('inca_dataset_langs.json', 'w') as f:
    f.write(json.dumps(langs))
with open('inca_dataset_geo_country_codes.json', 'w') as f:
    f.write(json.dumps(countries))
