import torch
from torch.utils.data import DataLoader

from dataset_processor import IncaTweetsDataset
from label_tracker import FileLabelTracker
from unicode_cnn import UnicodeCNN

model = UnicodeCNN()
model.load_state_dict(torch.load('../snapshots/14-05-2022_00:26:07.pth'))
model.eval()

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

test_kwargs = {'batch_size': 4}

label_tracker = FileLabelTracker(
    languages_filename='inca_dataset_langs.json',
    country_codes_filename='inca_dataset_geo_country_codes.json'
)
test_dataset = IncaTweetsDataset(path='../splits/test', label_tracker=label_tracker)
test_loader = DataLoader(test_dataset, **test_kwargs)

with torch.no_grad():
    for sample in test_loader:
        unicode_features = sample['matrix']
        euclidean_coordinates = sample['coordinates']
        output = model(unicode_features, euclidean_coordinates)
        print(output)
