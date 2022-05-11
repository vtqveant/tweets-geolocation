from os import listdir
from os.path import isfile, join
import csv


path = '../data'
filenames = [f for f in listdir(path) if isfile(join(path, f))]

langs = {}
countries = {}

counter = 0
for filename in filenames:
    with open(join(path, filename), newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            if row['lang'] not in langs.keys():
                langs[row['lang']] = 0
            if row['geo_country_code'] not in countries.keys():
                countries[row['geo_country_code']] = 0
    if counter % 100 == 0:
        print('{:.0f}% done'.format(100. * counter / len(filenames)))
    counter += 1

print('langs', len(langs.keys()))
print('countries', len(countries.keys()))
