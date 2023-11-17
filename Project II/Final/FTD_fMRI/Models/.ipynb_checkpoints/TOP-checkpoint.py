import torch, torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.res1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.Dropout(0.2),
        )

        self.res2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.Dropout(0.2),
        )

        self.res3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.Dropout(0.2),
        )

        self.res4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.Dropout(0.2),
        )

        self.net_out = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        # x = torch.cat((x1, x2), dim=1)
        x3 = self.res1(x2)
        x4 = self.res2(x1 + x3)
        x5 = self.res3(x3 + x4)
        x6 = self.res4(x4 + x5)
        x7 = self.net_out(x5 + x6)
        return x7