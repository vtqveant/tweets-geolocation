import torch
from torch import nn
import torch.nn.functional as F

from mvmf_layer import MvMFLayer

NUM_COUNTRY_CODES = 19  # 247 country codes defined by Twitter API, 19 in dataset
NUM_VMF_DISTRIBUTIONS = 1000


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
        language_prediction_raw_scores = self.fc2(t)
        # t = F.softmax(lang_pred_raw_scores, dim=1)

        # feature mixing
        q = torch.cat((x, language_prediction_raw_scores), 1)
        q = self.fc3(q)
        q = F.relu(q)
        q = self.fc4(q)
        mixed_features = F.relu(q)

        # Task 1: country prediction (this goes to a cross-entropy loss)
        # In PyTorch, the input is expected to contain raw, unnormalized scores for each class, so softmax
        # after y1 is not needed
        country_prediction_raw_scores = self.fc5(mixed_features)

        # Task 2: MvMF layer (this goes to a MvMF loss)
        mvmf_score = self.mvmf(mixed_features, euclidean_coordinates)

        return country_prediction_raw_scores, language_prediction_raw_scores, mvmf_score
