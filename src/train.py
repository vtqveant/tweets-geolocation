import csv
import numpy as np
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

from dataset_processor import IncaTweetsDataset
from label_tracker import FileLabelTracker
from mvmf_layer import MvMFLayer, MvMF_loss
from src.geometry import to_euclidean
from unicode_cnn import UnicodeCNN


def init_conv_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
    if isinstance(module, nn.Conv1d):
        torch.nn.init.xavier_uniform_(module.weight)


def init_mvmf_weights(module):
    """
    This function should be passed to nn.Module.apply()

        model = MvMFLayer(in_features=1024, num_distributions=10000)
        model.apply(init_mvmf_weights)
    """
    if isinstance(module, MvMFLayer):
        # value from the paper empirically observed to correspond to a st.d. about the size of a large city
        module.kappa.data.fill_(10.0)

        cs = []
        with open('cities_south_america.csv', newline='') as f:
            reader = csv.DictReader(f, delimiter=',')
            rows = list(reader)

            # MvMF Layer expects module.mu.data.size(dim=0) values for $\mu$
            if len(rows) < module.mu.data.size(dim=0):
                print('WARNING: not enough coordinates for MvMF Layer initialization')
                quit()

            for i, row in enumerate(rows):
                if i == module.mu.data.size(dim=0):
                    break
                c = to_euclidean(float(row['lat']), float(row['lon']))
                cs.append(c)

        module.mu.data.copy_(torch.tensor(np.array(cs)))


def unit_norm_mu_clipper(module):
    """
    This function should be passed to nn.Module.apply() after optimizer.step()

    Keeps norm2(mu) == 1

    https://discuss.pytorch.org/t/restrict-range-of-variable-during-gradient-descent/1933/3
    """
    if isinstance(module, MvMFLayer) and hasattr(module, 'mu'):
        w = module.mu.data
        w.div_(torch.linalg.vector_norm(w, 2, 1).reshape(-1, 1).expand_as(w))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        unicode_features = sample['matrix'].to(device)
        euclidean_coordinates_target = sample['coordinates'].to(device)
        geo_country_code_target = sample['geo_country_code'].to(device)
        language_target = sample['lang'].to(device)

        optimizer.zero_grad()
        country_prediction_output, language_prediction_output, mvmf_output = model(unicode_features, euclidean_coordinates_target)

        # Task 0 - language prediction
        language_prediction_loss = F.cross_entropy(language_prediction_output, language_target, reduction='mean')

        # Task 1 - country prediction
        country_prediction_loss = F.cross_entropy(country_prediction_output, geo_country_code_target, reduction='mean')

        # Task 2 - MvMF loss
        target = torch.zeros_like(mvmf_output).to(device)
        mvmf_loss = MvMF_loss(mvmf_output, target)

        # combined loss
        combined_output = torch.add(language_prediction_loss, torch.add(country_prediction_loss, mvmf_loss))
        zero = torch.zeros_like(combined_output).to(device)
        loss = F.mse_loss(combined_output, zero)

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)

        # all mean directions ($\mu$) of component MvMF distributions must stay on the unit sphere
        # (cities don't fly away)
        model.apply(unit_norm_mu_clipper)

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.12f}'.format(
                epoch, batch_idx * len(unicode_features), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

        if batch_idx > 0 and batch_idx % args.snapshot_interval == 0:
            if args.dry_run:
                break
            print('Saving snapshot for epoch {}\n'.format(epoch))
            torch.save(model.state_dict(), '../snapshots/' + datetime.now().strftime("%d-%m-%Y_%H:%M:%S_1000dist_large_dataset") + '.pth')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='UnicodeCNN + MvMF')
    parser.add_argument('--batch-size', type=int, default=400, metavar='N',
                        help='input batch size for training (default: 400)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--clip', type=float, default=4.0, metavar='CL',
                        help='max_norm (clipping threshold) (default: 4.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--snapshot-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before saving a snapshot')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {device} device")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    model = UnicodeCNN().to(device)
    print(model)

    # initialization of linear and convolutional layers
    model.apply(init_conv_weights)

    # initialization for MvMF layer
    model.apply(init_mvmf_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    label_tracker = FileLabelTracker(
        languages_filename='inca_dataset_langs.json',
        country_codes_filename='inca_dataset_geo_country_codes.json'
    )
    train_dataset = IncaTweetsDataset(path='../data', label_tracker=label_tracker, use_cache=False)  # TODO this is not a proper split, just to overfit once
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_dataset = IncaTweetsDataset(path='../splits/test', label_tracker=label_tracker)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    # start where we ended last time
    # model.load_state_dict(torch.load('../snapshots/16-05-2022_19:08:13_1000dist_large_dataset.pth'))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        torch.save(model.state_dict(), '../snapshots/' + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.pth')
        # test(model, device, test_loader)


if __name__ == '__main__':
    main()
