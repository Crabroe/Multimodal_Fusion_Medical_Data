import torch, torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.net_in = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4005, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
        )

        self.res1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
        )

        self.res2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
        )

        self.res3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
        )

        self.res4 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
        )

        self.res5 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
        )

        self.net_out = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
        )

    def ResOut(self, x):
        x1 = self.net_in(x)
        x2 = self.res1(x1)
        x3 = self.res2(x1 + x2)
        x4 = self.res3(x2 + x3)

        return x4 + x3

    def forward(self, x):
        x34 = self.ResOut(x)
        x5 = self.net_out(x34)

        return x5