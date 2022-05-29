"""
An example of model usage. Consult eval.py for batch processing.
"""

import torch
from coordinate_prediction import predict_coord_center_of_mass
from geometry import to_geographical
from unicode_cnn import UnicodeCNN


def predict(text):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {device} device")

    eval_kwargs = {'batch_size': 1}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0, 'pin_memory': True, 'shuffle': False}
        eval_kwargs.update(cuda_kwargs)

    model = UnicodeCNN()
    model.load_state_dict(torch.load('../snapshots/weights.pth'))
    model.eval()

    [result] = predict_coord_center_of_mass(model, [text])
    return to_geographical(result)


if __name__ == '__main__':
    # 0.5094617011385917;-51.05876773901514
    lat, lon = predict("@edsonsaless Não sei, mas que os dois fazem um casal lindo sim!! Mato ")
    print('({}, {})'.format(lat, lon))
