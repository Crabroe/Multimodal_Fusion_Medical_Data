import torch, torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers=1, batch_size=1):
        super(Model, self).__init__()

        self.batch_size = batch_size    # 一次性输入的句子量，这里通常置为1
        self.hidden_size = hidden_size  # 隐藏层的宽度
        self.n_layers = n_layers        # 是否是多重LSTM连接
        self.input_size = input_size
        self.out_size = out_size        # 最后层神经网络的输出宽度

        self.lstm = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=False)
        self.out = nn.Linear(hidden_size, out_size)
        self.softmax = nn.Softmax()

        self.sequence = nn.Sequential(
            nn.Linear(hidden_size, 8),
            nn.Dropout(0.25),
            nn.Linear(8, 8),
            # nn.Softmax(dim=1)
        )

    def forward(self, input):
        # input = input.squeeze()  # 去掉 (batch, 1, 28, 28) 中的 1，变成 (batch, 28, 28)
        input = input.permute(2, 0, 1)

        h = torch.autograd.Variable(torch.zeros([self.n_layers, input.size(0), self.hidden_size])).to(device)

        output, hidden = self.lstm(input, h)
        output = self.sequence(output[-1, :, :])

        return output