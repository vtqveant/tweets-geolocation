import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

from dataset_processor import IncaTweetsDataset
from label_tracker import FileLabelTracker
from mvmf_layer import MvMFLayer, init_mvmf_weights, MvMF_loss, unit_norm_mu_clipper

# Based on MNIST implementation from git@github.com:pytorch/examples.git

NUM_COUNTRY_CODES = 19  # 247 country codes defined by Twitter API, 19 in dataset
NUM_VMF_DISTRIBUTIONS = 10000


class UnicodeCNN(nn.Module):
    def __init__(self):
        super(UnicodeCNN, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3)

        # language estimator (66 languages supported by Twitter API, including Unknown)
        self.fc1 = nn.Linear(in_features=256 * 6, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=66)

        # feature mixing
        self.fc3 = nn.Linear(in_features=1602, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=1024)

        # country cross-entropy prediction (Twitter API defines a total of 247 unique country codes)
        self.fc5 = nn.Linear(in_features=1024, out_features=NUM_COUNTRY_CODES)

        # MvMF
        self.mvmf = MvMFLayer(in_features=1024, num_distributions=NUM_VMF_DISTRIBUTIONS)

    def forward(self, unicode_features, euclidean_coordinates):
        # convolutional layers
        x = self.conv1(unicode_features)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = torch.flatten(x, start_dim=1)

        # language estimator
        t = self.fc1(x)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.softmax(t, dim=1)

        # feature mixing
        q = torch.cat((x, t), 1)
        q = self.fc3(q)
        q = F.relu(q)
        q = self.fc4(q)
        mixed_features = F.relu(q)

        # Task 1: country prediction (this goes to a cross-entropy loss)
        # In PyTorch, the input is expected to contain raw, unnormalized scores for each class, so softmax
        # after y1 is not needed
        y1 = self.fc5(mixed_features)

        # Task 2: MvMF layer (this goes to a MvMF loss)
        y2 = self.mvmf(mixed_features, euclidean_coordinates)

        return y1, y2


def init_conv_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
    if isinstance(module, nn.Conv1d):
        torch.nn.init.xavier_uniform_(module.weight)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        unicode_features = sample['matrix'].to(device)
        euclidean_coordinates = sample['coordinates'].to(device)
        geo_country_code = sample['geo_country_code'].to(device)

        optimizer.zero_grad()
        output1, output2 = model(unicode_features, euclidean_coordinates)

        # Task 1 - country prediction
        loss1 = F.cross_entropy(output1, geo_country_code, reduction='mean')

        # Task 2 - MvMF loss
        target = torch.zeros_like(output2).to(device)
        loss2 = MvMF_loss(output2, target)

        # combined loss
        combined_output = torch.add(loss1, loss2)
        zero = torch.zeros_like(combined_output).to(device)
        loss = F.l1_loss(combined_output, zero)

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
            torch.save(model.state_dict(), '../snapshots/' + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.pth')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            unicode_features = sample['matrix'].to(device)
            euclidean_coordinates = sample['coordinates'].to(device)
            # geo_country_code = sample['geo_country_code'].to(device)

            # Task 1 - country code prediction
            # test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

            # Task 2 = MvMF loss
            target = torch.zeros(len(euclidean_coordinates)).to(device)
            output = model(unicode_features, euclidean_coordinates)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='UnicodeCNN + MvMF')
    parser.add_argument('--batch-size', type=int, default=400, metavar='N',
                        help='input batch size for training (default: 400)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
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
    train_dataset = IncaTweetsDataset(path='../splits/train', label_tracker=label_tracker, use_cache=True)  # TODO this is not a proper split, just to overfit once
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_dataset = IncaTweetsDataset(path='../splits/test', label_tracker=label_tracker)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    # start where we ended last time
    # model.load_state_dict(torch.load('../snapshots/15-05-2022_02:32:54.pth'))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        torch.save(model.state_dict(), '../snapshots/' + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.pth')
        # test(model, device, test_loader)


if __name__ == '__main__':
    main()
