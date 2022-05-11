import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from dataset_processor import LabelTracker, IncaTweetsDataset


# Based on MNIST implementation from git@github.com:pytorch/examples.git


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)

        # language estimator (66 languages supported by Twitter API, including Unknown)
        self.fc1 = nn.Linear(in_features=1536, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=66)

        # feature mixing
        self.fc3 = nn.Linear(in_features=1602, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=1024)

        # country cross-entropy prediction (Twitter API defines a total of 247 unique country codes)
        self.fc5 = nn.Linear(in_features=1024, out_features=247)

    def forward(self, x):
        # convolutional layers
        x = self.conv1(x)
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
        x = torch.flatten(x, 1)

        # language estimator
        t = self.fc1(x)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = F.softmax(t, dim=1)

        # feature mixing
        q = torch.cat((x, t), 1)
        q = self.fc3(q)
        q = F.relu(q)
        q = self.fc4(q)
        q = F.relu(q)

        # country cross-entropy prediction
        y = self.fc5(q)
        y = F.softmax(y, dim=1)

        return y


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        data, target = sample['matrix'].to(device), sample['geo_country_code'].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{} samples]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                loss.item()
            ))
            if args.dry_run:
                break


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {device} device")

    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)

    model = NeuralNetwork().to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    label_tracker = LabelTracker()
    dataset = IncaTweetsDataset(label_tracker=label_tracker)
    train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)

    # a singe epoch to play with
    train(args, model, device, train_loader, optimizer, epoch=0)


if __name__ == '__main__':
    main()
