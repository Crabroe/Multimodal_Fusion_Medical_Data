# 分类ADNI_90_120的循环神经网络

import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision, torch.nn as nn
import Loader_ADNI_RNN as Loader
import Judge

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers=1, batch_size=1):
        super(RNN, self).__init__()

        self.batch_size = batch_size    # 一次性输入的句子量，这里通常置为1
        self.hidden_size = hidden_size  # 隐藏层的宽度
        self.n_layers = n_layers        # 是否是多重LSTM连接
        self.input_size = input_size
        self.out_size = out_size        # 最后层神经网络的输出宽度

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, out_size)
        self.softmax = nn.Softmax()

        self.sequence = nn.Sequential(
            nn.Linear(hidden_size, 4),
            nn.Dropout(0.15),
            nn.Linear(4, out_size),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        # input = input.squeeze()  # 去掉 (batch, 1, 28, 28) 中的 1，变成 (batch, 28, 28)
        input = input.permute(2, 0, 1)

        h = torch.autograd.Variable(torch.zeros([self.n_layers, input.size(0), self.hidden_size]))
        c = torch.autograd.Variable(torch.zeros([self.n_layers, input.size(0), self.hidden_size]))

        output, hidden = self.lstm(input, (h, c))
        output = self.sequence(output[-1, :, :])

        return output

# # 测试代码
# model = RNN(90, 16, 2).cuda()
# X = torch.rand([3, 1, 90, 160]).cuda()
#
# out = model(X)
# print(out.shape)
# quit()

LR = 2e-4
EPOCH = 500
n_fold = 10
if_PRINT = False

datasets = Loader.RNN_YTL_Fold(n_fold)

acc_list_total = []
best_roc_list, best_sen_list, best_spe_list = [], [], []

for dataset in datasets:
    datas_train, datas_test = dataset
    model = RNN(90, 48, 2)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    acc_list_turn = []

    # 作图需要的参数
    _X = list(range(EPOCH))
    _Y_train = []
    _Y_test = []

    # 最终评价指标
    good_acc = 0
    good_con_mat = None
    good_epoch = 0

    # 开始训练
    for i in range(EPOCH):

        # a) train part
        model.train()
        acc_for_train, count = 0, 0
        for data in datas_train:
            samples, labels = data
            outputs = model(samples)
            # recording acc
            preds = outputs.argmax(1)
            acc = (preds == labels).sum()
            count += len(labels)
            acc_for_train += acc

            # loss computing, backward
            labels = labels.long()
            loss = loss_fn(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
        _Y_train.append(acc_for_train / count)

        # b) test part
        model.eval()
        acc_for_test, count = 0, 0
        con_mat = np.zeros((2, 2))

        for data in datas_test:
            samples, labels = data
            outputs = model(samples).cpu()
            labels = labels.cpu()

            preds = outputs.argmax(1)
            acc = (preds == labels).sum()
            count += len(labels)
            acc_for_test += acc

            for one in range(preds.shape[0]):
                con_mat[int(labels[one].item()), preds[one]] += 1

        _Y_test.append(acc_for_test / count)

        # c) Summarize
        print('\rin epoch {}, test acc = {} / {}(epoch:{})'.format(i, acc_for_test / count, good_acc, good_epoch), end='')
        # print(con_mat)
        acc_list_turn.append(acc_for_test / count)

        # d) 记录最好的值
        if acc_for_test / count >= good_acc:
            good_acc = acc_for_test / count
            good_con_mat = con_mat
            good_epoch = i
    print()

    # 记录所有的评价参数
    good_roc = Judge.mat_roc(good_con_mat.astype(dtype=int))
    good_sensitivity = Judge.mat_sensitivity(good_con_mat.astype(dtype=int))
    good_specificity = Judge.mat_specify(good_con_mat.astype(dtype=int))
    best_roc_list.append(good_roc)
    best_sen_list.append(good_sensitivity)
    best_spe_list.append(good_specificity)
    acc_list_total.append(max(acc_list_turn))

    if if_PRINT:
        # plt
        plt.figure()
        plt.plot(_X, _Y_test, c='red')
        plt.plot(_X, _Y_train, c='blue')
        plt.show()
print(np.mean(np.asarray(acc_list_total)))
print('roc:{}, sensitivity:{}, specificity:{}'.format(np.mean(np.asarray(best_roc_list)),
                                                      np.mean(np.asarray(best_sen_list)),
                                                      np.mean(np.asarray(best_spe_list))))