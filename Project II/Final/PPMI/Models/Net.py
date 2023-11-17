import torch, torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.sequence1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(98, 64),
            nn.Dropout(0.15),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
        )

        self.sequence2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(98, 64),
            nn.Dropout(0.15),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
        )

        self.sequence3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(98, 64),
            nn.Dropout(0.15),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
        )

        self.out = nn.Sequential(
            nn.Linear(32, 32),
            nn.Dropout(0.15),
            nn.Linear(32, 32),
            nn.Dropout(0.15),
            nn.Linear(32, 8),
            nn.Dropout(0.15),
            nn.Linear(8, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2, x3):
        a = x1 - x2
        b = x2 - x3
        c = x1 + x2 + x3
        out1 = self.sequence1(a)
        out2 = self.sequence2(b)
        out3 = self.sequence3(c)
        out = self.out(out1 + out2 + out3)
        return out

if __name__ == '__main__':
    x1 = torch.rand((10, 1, 98))
    x2 = torch.rand((10, 1, 98))

    net = Model()

    print(net(x1, x2, x2))