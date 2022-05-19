import torch
from torch.utils.data import DataLoader

from dataset_processor import IncaTweetsDataset
from label_tracker import FileLabelTracker
from unicode_cnn import UnicodeCNN

model = UnicodeCNN()
model.load_state_dict(torch.load('../snapshots/weights.pth'))
model.eval()

test_kwargs = {'batch_size': 10}

label_tracker = FileLabelTracker(
    languages_filename='inca_dataset_langs.json',
    country_codes_filename='inca_dataset_geo_country_codes.json'
)
test_dataset = IncaTweetsDataset(path='../splits/eval', label_tracker=label_tracker)
test_loader = DataLoader(test_dataset, **test_kwargs)

with torch.no_grad():
    for sample in test_loader:
        unicode_features = sample['matrix']
        euclidean_coordinates = sample['coordinates']
        country_raw_weights, lang_pred_raw_weights, mvmf_pred, _ = model(unicode_features, euclidean_coordinates)

        # print('country raw weights (apply argmax to obtain the country index)\n\t', country_raw_weights)
        # print('language raw weights\n\t', lang_pred_raw_weights)

        print('MvMF score\t', mvmf_pred)

        country = [label_tracker.get_country(i) for i in torch.argmax(country_raw_weights, dim=1).numpy()]
        print(country)

        language = [label_tracker.get_language(i) for i in torch.argmax(lang_pred_raw_weights, dim=1).numpy()]
        print(language)
