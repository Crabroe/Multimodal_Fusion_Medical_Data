import torch, torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.sequence = nn.Sequential(
            nn.Linear(8, 6),
            nn.Dropout(0.3),    # 0.2
            nn.Linear(6, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        # print(x1.shape, x2.shape)
        # X = torch.cat((x1, x2), dim=1)
        X = x1 + x2
        return self.sequence(X)