import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class FFNet(torch.nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()
        self.ffnet_hidden_dim = 32
        self.ffnet_out_dim = 2
        self.ffnet = nn.Sequential(
            nn.Linear(4, self.ffnet_hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(self.ffnet_hidden_dim, self.ffnet_out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, 512 * 512 * 512)
        x = self.ffnet(x)
        return x


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1_dim = 32
        self.conv2_dim = 64
        self.conv3_dim = 128
        self.ffnet_hidden_dim = 512
        self.ffnet_out_dim = 124
        self.out_dim = 2

        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=self.conv1_dim, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.conv1_dim),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=self.conv1_dim, out_channels=self.conv2_dim, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.conv2_dim),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=self.conv2_dim, out_channels=self.conv3_dim, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.conv3_dim),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.ffnet = nn.Sequential(
            nn.Linear(self.conv3_dim * 2 * 2, self.ffnet_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.ffnet_hidden_dim, self.ffnet_out_dim),
            nn.Sigmoid()
        )

        self.regression = nn.Linear(self.ffnet_out_dim + 4, 2)

    def forward(self, x, scalars):
        x = self.conv_pool(x)
        batch_size, c_out, h_out, w_out = x.size()
        # print(batch_size, c_out, h_out, w_out)
        x = x.view(-1, c_out * h_out * w_out)
        x = self.ffnet(x)
        x = torch.cat([x, scalars], dim=1)
        x = self.regression(x)
        return x


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.reduce = nn.Linear(1000, 124)
        self.regression = nn.Linear(128, 2)

    def forward(self, input_image, scalars):
        out = self.resnet(input_image)
        out = F.relu(self.reduce(out))
        out = torch.cat([out, scalars], dim=1)
        out = self.regression(out)
        return out
