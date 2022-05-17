import torch
from torch.utils.data import DataLoader

from dataset_processor import IncaTweetsDataset
from label_tracker import FileLabelTracker
from unicode_cnn import UnicodeCNN

model = UnicodeCNN()
model.load_state_dict(torch.load('../snapshots/17-05-2022_01:44:55_1000dist_large_dataset.pth'))
model.eval()

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

test_kwargs = {'batch_size': 10}

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
        country_raw_weights, lang_pred_raw_weights, mvmf_pred = model(unicode_features, euclidean_coordinates)

        # print('country raw weights (apply argmax to obtain the country index)\n\t', country_raw_weights)
        # print('language raw weights\n\t', lang_pred_raw_weights)

        print('MvMF score (interpreted as a probability of the tweet being posted from this location\n\t', mvmf_pred)

        country = [label_tracker.get_country(i) for i in torch.argmax(country_raw_weights, dim=1).numpy()]
        print(country)

        language = [label_tracker.get_language(i) for i in torch.argmax(lang_pred_raw_weights, dim=1).numpy()]
        print(language)
