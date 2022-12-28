import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        # self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.3)
        # self.bn1 = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        # self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.3)
        # self.bn2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        # self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.3)
        # self.bn3 = nn.BatchNorm2d(num_features=128)
        #
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        # self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(p=0.3)
        # self.bn4 = nn.BatchNorm2d(num_features=256)

        # self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(in_features=256 * 12 * 12, out_features=1000)
        # self.bn5 = nn.BatchNorm1d(num_features=1000)
        # self.dense1 = nn.Linear(in_features=128 * 25 * 25, out_features=640)
        # self.dense1 = nn.Linear(in_features=64 * 53 * 53, out_features=1000)

        # self.act5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(in_features=1000, out_features=640)
        # self.bn6 = nn.BatchNorm1d(num_features=640)
        # self.act6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.3)
        self.dense3 = nn.Linear(in_features=640, out_features=68 * 2)

    def forward(self, x):
        # x = self.dropout1(self.pool1(self.act1(self.conv1(x))))
        # x = self.dropout2(self.pool2(self.act2(self.conv2(x))))
        # x = self.dropout3(self.pool3(self.act3(self.conv3(x))))

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.dropout4(x)

        # print("before flatten", x.shape)
        # x = self.flatten(x)
        # print("after flatten", x.shape)
        x = x.view(x.size(0), -1)

        x = self.dropout5(F.relu(self.dense1(x)))
        x = self.dropout6(F.relu(self.dense2(x)))
        x = self.dense3(x)
        return x
