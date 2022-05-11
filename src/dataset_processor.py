import torch
from torch.utils.data.dataset import IterableDataset
from os import listdir
from os.path import isfile, join
import csv
import random

from typing import List

import character_encoder
from character_encoder import CharacterEncoder


class IncaTweetsDataset(IterableDataset):
    """A generator for training examples constructed from files in the dataset"""

    def __init__(self, path, label_tracker):
        super(IterableDataset, self).__init__()
        self._path = path
        self._label_tracker = label_tracker
        self._file_processor = FileProcessor()

        self._filenames = [f for f in listdir(self._path) if isfile(join(self._path, f))]
        random.shuffle(self._filenames)

        self._num_samples = None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            print("ERROR: multiple workers are not supported, DataLoader needs to be initialized with num_workers=0")
            quit()

        for filename in self._filenames:
            entries = self._file_processor.process(self._path, filename)
            for entry in entries:
                yield {
                    "matrix": self._to_tensor(entry.matrix),
                    "lang": self._label_tracker.get_language_index(entry.lang),
                    "geo_country_code": self._label_tracker.get_country_index(entry.geo_country_code)
                }

    def __len__(self):
        """TODO don't use it in the final solution, use it only to play with the dataset a little bit"""
        if self._num_samples is None:
            self._num_samples = 0
            for filename in self._filenames:
                with open(join(self._path, filename), newline='') as f:
                    reader = csv.DictReader(f, delimiter=';')
                    for _ in reader:
                        self._num_samples += 1
        return self._num_samples

    @staticmethod
    def _to_tensor(matrix):
        """TODO: this is a temporary solution"""
        return torch.transpose(torch.tensor([[float(i) for i in s] for s in matrix]), 0, 1)


class FileProcessor:
    def __init__(self):
        self._characterEncoder = CharacterEncoder(character_encoder.ENCODING_SIZE_SMALL)

    def process(self, path, filename) -> List:
        coord = self._parse_coordinates(filename)
        entries: List = []
        with open(join(path, filename), newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                matrix = self._characterEncoder.encode(row['text'])
                training_example = TrainingExample(matrix, row['lang'], row['geo_country_code'], coord)
                entries.append(training_example)
        return entries

    @staticmethod
    def _parse_coordinates(filename):
        idx = filename.find('_')
        lat = filename[0:idx]
        lon = filename[idx + 1:filename.find('.csv')]
        return Coordinate(lat, lon)


class LabelTracker:
    """A container for labels with lazy registration"""

    def __init__(self):
        self.language_index = 0
        self.country_code_index = 0
        self.languages = {}
        self.geo_country_codes = {}

    def get_language_index(self, language):
        if language not in self.languages.keys():
            self.languages[language] = self.language_index
            self.language_index += 1
        return self.languages[language]

    def get_country_index(self, country_code):
        if country_code not in self.geo_country_codes.keys():
            self.geo_country_codes[country_code] = self.country_code_index
            self.country_code_index += 1
        return self.geo_country_codes[country_code]


class Coordinate:
    """A class to represent a point on the Earth"""

    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def __str__(self):
        return self.lat + ', ' + self.lon


class TrainingExample:
    def __init__(self, matrix: List[List[str]], lang: str, geo_country_code: str, coordinates: Coordinate):
        self.matrix = matrix
        self.lang = lang
        self.geo_country_code = geo_country_code
        self.coordinates = coordinates


def main():
    label_tracker = LabelTracker()
    dataset = IncaTweetsDataset(path='../splits/train', label_tracker=label_tracker)
    # train_set = [x for _, x in zip(range(10), dataset)]
    # print(train_set)

    # train_set = [x for _, x in zip(range(1000), torch.utils.data.DataLoader(dataset, num_workers=0))]
    train_set = [x for x in torch.utils.data.DataLoader(dataset, num_workers=0)]
    print(len(dataset))
    # print(train_set)

    print(label_tracker.languages)
    print(label_tracker.geo_country_codes)


if __name__ == '__main__':
    main()
