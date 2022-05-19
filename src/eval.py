import torch
from torch.utils.data import DataLoader
from vincenty import vincenty

from coordinate_prediction import predict_coord_center_of_mass
from dataset_processor import IncaTweetsDataset
from geometry import to_geographical
from label_tracker import FileLabelTracker
from unicode_cnn import UnicodeCNN


def evaluate(test_loader, snapshot):
    model = UnicodeCNN()
    model.load_state_dict(torch.load(snapshot))
    model.eval()

    distances = []
    for batch in test_loader:
        text = batch['text']
        true_coordinates = map(to_geographical, batch['coordinates'])

        result = predict_coord_center_of_mass(model, text)
        prediction_coordinates = map(to_geographical, result)

        batch_distances = [vincenty(t, p) for t, p in zip(true_coordinates, prediction_coordinates)]
        print(batch_distances)

        distances.extend(batch_distances)

    average_distance = sum(distances) / len(distances)
    print('\nMAE (km): {:.4f}\n'.format(average_distance))


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {device} device")

    test_kwargs = {'batch_size': 100}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0, 'pin_memory': True, 'shuffle': False}
        test_kwargs.update(cuda_kwargs)

    label_tracker = FileLabelTracker(
        languages_filename='inca_dataset_langs.json',
        country_codes_filename='inca_dataset_geo_country_codes.json'
    )
    test_dataset = IncaTweetsDataset(path='../splits/test', label_tracker=label_tracker, shuffle=False)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    evaluate(test_loader, '../snapshots/19-05-2022_09:59:01.pth')
