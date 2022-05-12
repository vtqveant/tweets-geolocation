import torch
from torch.utils.data.dataset import IterableDataset
from os import listdir
from os.path import isfile, join
import csv
import random

from typing import List

import character_encoder
from character_encoder import CharacterEncoder
from label_tracker import LabelTracker, FileLabelTracker
from geometry import Coordinate, to_euclidean


class IncaTweetsDataset(IterableDataset):
    """An iterator for training examples constructed from files in the dataset"""

    def __init__(self, path: str, label_tracker: LabelTracker):
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
                    "coordinates": torch.tensor(to_euclidean(entry.coord.lat, entry.coord.lon), dtype=torch.float32),
                    "lang": self._label_tracker.get_language_index(entry.lang),
                    "geo_country_code": self._label_tracker.get_country_index(entry.geo_country_code)
                }

    def __len__(self):
        """TODO don't use it in the final solution, use it only to play with the dataset a little bit"""
        return 10815072
        # if self._num_samples is None:
        #     self._num_samples = 0
        #     for filename in self._filenames:
        #         with open(join(self._path, filename), newline='') as f:
        #             reader = csv.DictReader(f, delimiter=';')
        #             for _ in reader:
        #                 self._num_samples += 1
        # return self._num_samples

    @staticmethod
    def _to_tensor(matrix):
        """a matrix is actually a list of bytestring consisting of b'0' and b'1', so we need to offset by 48"""
        return torch.transpose(torch.tensor([[float(i - 48) for i in s] for s in matrix], dtype=torch.float32), 0, 1)


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


class TrainingExample:
    def __init__(self, matrix: List[List[str]], lang: str, geo_country_code: str, coordinate: Coordinate):
        self.matrix = matrix
        self.lang = lang
        self.geo_country_code = geo_country_code
        self.coord = coordinate


def main():
    label_tracker = FileLabelTracker(
        languages_filename='inca_dataset_langs.json',
        country_codes_filename='inca_dataset_geo_country_codes.json'
    )
    dataset = IncaTweetsDataset(path='../data', label_tracker=label_tracker)
    # train_set = [x for _, x in zip(range(10), dataset)]
    # print(train_set)

    train_set = [x for _, x in zip(range(10), torch.utils.data.DataLoader(dataset, num_workers=0))]
    # train_set = [x for x in torch.utils.data.DataLoader(dataset, num_workers=0)]
    # print(len(dataset))
    print(train_set)


if __name__ == '__main__':
    main()
