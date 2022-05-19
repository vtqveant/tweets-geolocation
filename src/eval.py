import torch
from torch.utils.data import DataLoader
from vincenty import vincenty

from coordinate_prediction import predict_coord_center_of_mass
from dataset_processor import IncaTweetsDataset
from geometry import to_geographical
from label_tracker import FileLabelTracker
from unicode_cnn import UnicodeCNN


def evaluate(data_loader: DataLoader, snapshot: str, max_batches: int):
    """Computes an average distance between true and predicted coordinates using Vincenty distance algorithm."""
    model = UnicodeCNN()
    model.load_state_dict(torch.load(snapshot))
    model.eval()

    distances = []
    for i, batch in enumerate(data_loader):
        if i == max_batches:
            break

        text = batch['text']
        true_coordinates = map(to_geographical, batch['coordinates'])

        result = predict_coord_center_of_mass(model, text)
        predicted_coordinates = map(to_geographical, result)

        batch_distances = [vincenty(t, p) for t, p in zip(true_coordinates, predicted_coordinates)]
        distances.extend(batch_distances)
        print('{:.0f}% done'.format(100.0 * (i + 1) / max_batches))

    average_distance = sum(distances) / len(distances)
    print('\nMAE (km): {:.4f}\n'.format(average_distance))


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {device} device")

    eval_kwargs = {'batch_size': 1000}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0, 'pin_memory': True, 'shuffle': False}
        eval_kwargs.update(cuda_kwargs)

    label_tracker = FileLabelTracker(
        languages_filename='inca_dataset_langs.json',
        country_codes_filename='inca_dataset_geo_country_codes.json'
    )
    eval_dataset = IncaTweetsDataset(path='../splits/eval', label_tracker=label_tracker, shuffle=False)
    loader = DataLoader(eval_dataset, **eval_kwargs)

    evaluate(loader, '../snapshots/weights.pth', max_batches=10)
