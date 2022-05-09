from os import listdir
from os.path import isfile, join
import csv

import character_encoder
from character_encoder import CharacterEncoder


PATH = '../data'


class DatasetPreprocessor:

    def __init__(self):
        self.filePreprocessor = FilePreprocessor()

    def process(self):
        filenames = [f for f in listdir(PATH) if isfile(join(PATH, f))]
        # a subset to play with
        filenames = filenames[0:1]
        print(filenames)

        entries = {}
        for filename in filenames:
            entries.update(self.filePreprocessor.process(filename))
        return entries


class FilePreprocessor:

    def __init__(self):
        self._characterEncoder = CharacterEncoder(character_encoder.ENCODING_SIZE_SMALL)

    def process(self, filename):
        coord = self._parse_coordinates(filename)
        entries = {coord: []}
        with open('../data/' + filename, newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                lang = row['lang']
                text = row['text']
                matrix = self._characterEncoder.encode(text)
                entries[coord].append(matrix)
        return entries

    @staticmethod
    def _parse_coordinates(filename):
        idx = filename.find('_')
        lat = filename[0:idx]
        lon = filename[idx + 1:filename.find('.csv')]
        return Coordinate(lat, lon)


class Coordinate:
    """A class to represent a point on the Earth"""

    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def __str__(self):
        return self.lat + ', ' + self.lon


def main():
    preprocessor = DatasetPreprocessor()
    entries = preprocessor.process()
    print(entries)


if __name__ == '__main__':
    main()
