import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.act1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.act2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.act3 = nn.ELU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout()
        #
        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        # self.act4 = nn.ReLU()
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout4 = nn.Dropout()

        self.flatten = nn.Flatten()

        # self.dense1 = nn.Linear(in_features=256 * 12 * 12, out_features=1000)
        self.dense1 = nn.Linear(in_features=128 * 25 * 25, out_features=512)
        # self.dense1 = nn.Linear(in_features=64 * 53 * 53, out_features=512)

        self.act5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.1)
        # self.dense2 = nn.Linear(in_features=1000, out_features=1000)
        # self.act6 = nn.ReLU()
        # self.dropout6 = nn.Dropout()
        self.dense3 = nn.Linear(in_features=512, out_features=68 * 2)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.act1(self.conv1(x))))
        x = self.dropout2(self.pool2(self.act2(self.conv2(x))))
        x = self.dropout3(self.pool3(self.act3(self.conv3(x))))
        # x = self.dropout4(self.pool4(self.act4(self.conv4(x))))

        # print("before flatten", x.shape)
        x = self.flatten(x)
        # print("after flatten", x.shape)

        x = self.dropout5(self.act5(self.dense1(x)))
        # x = self.dropout6(self.act6(self.dense2(x)))
        x = self.dense3(x)
        return x
