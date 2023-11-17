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

        self.net_out = nn.Sequential(
            nn.Linear(128, 32),
            nn.Dropout(0.2),
            nn.BatchNorm1d(32),
            nn.Linear(32, 8),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.net_in(x)
        x2 = self.res1(x1)
        x3 = self.res2(x1 + x2)
        x4 = self.res3(x2 + x3)
        x5 = self.res4(x3 + x4)
        x6 = self.net_out(x4 + x5)

        return x6