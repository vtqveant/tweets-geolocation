from os import listdir
from os.path import isfile, join
import csv

from typing import List

import character_encoder
from character_encoder import CharacterEncoder


PATH = '../data'


class DatasetProcessor:
    """A generator for training examples constructed from files in the dataset"""

    def __init__(self):
        self.fileProcessor = FileProcessor()

    def __iter__(self):
        filenames = [f for f in listdir(PATH) if isfile(join(PATH, f))]
        for filename in filenames:
            entries = self.fileProcessor.process(filename)
            for entry in entries:
                yield entry


class FileProcessor:

    def __init__(self):
        self._characterEncoder = CharacterEncoder(character_encoder.ENCODING_SIZE_SMALL)

    def process(self, filename) -> List:
        coord = self._parse_coordinates(filename)
        entries: List = []
        with open('../data/' + filename, newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                matrix = self._characterEncoder.encode(row['text'])
                training_example = TrainingExample(matrix, row['lang'], coord)
                entries.append(training_example)
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


class TrainingExample:

    def __init__(self, matrix: List[List[str]], lang: str, coordinates: Coordinate):
        self.matrix = matrix
        self.lang = lang
        self.coordinates = coordinates


def main():
    processor = DatasetProcessor()
    train_set = [x for _, x in zip(range(50), processor)]
    print(train_set)


if __name__ == '__main__':
    main()
