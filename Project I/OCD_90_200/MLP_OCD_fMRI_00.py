import matplotlib.pyplot as plt
import torch, torch.nn as nn, torchvision
import Loader_OCD_MLP as Loader
import numpy as np
import Judge

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4005, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.Dropout(0.1),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

ave_acc = 0 # 多次time的平均训练时间
fold_num = 10
train_time = fold_num
EPOCH = 1000
learning_rate = 0.0005

mat_sum = np.zeros((2, 2), dtype=int)
dataSets = Loader.YTL_Fold(fold_num)

each_time = 0
time_list = list(range(EPOCH))
for dataset in dataSets:
    each_time += 1
    datas_train, datas_test = dataset
    acc_list = []
    acc_list_train = []
    loss_list_train = []

    Model = NET()

    high_epoch = 0  # 测试得最高准确率的回合
    high_acc = 0    # 测试的最高准确率
    high_mat = 0    # 最高准确率对应的测试集混淆矩阵

    lost_fn = nn.CrossEntropyLoss()
    # optim = torch.optim.RMSprop(Model.parameters(), lr=learning_rate)
    optim = torch.optim.SGD(Model.parameters(), lr=learning_rate)

    # 1.每一次训练
    for i in range(EPOCH):
        total_loss = 0
        total_train_step = 0    # 被取出的batch次数
        total_acc_train = 0 # 在训练过程中的准确率之和
        count_train = 0     # 在训练过程中的样本数量

        # 1.1 这是训练的部分
        Model.train()
        for data in datas_train:
            imgs, targets = data
            outs = Model(imgs)

            preds = Model(imgs).argmax(1)
            acc = (preds == targets).sum()
            count_train += len(targets)
            total_acc_train += acc

            targets = targets.long()

            loss = lost_fn(outs, targets)
            total_loss += loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_train_step += 1

        # 1.1.1 将训练部分的准确率加入列表中 [作图]
        acc_list_train.append(total_acc_train / count_train)
        loss_list_train.append(total_loss)

        total_acc_test = 0  # 在测试过程中的准确率之和
        count_test = 0      # 在测试过程中的样本数量
        mat = np.zeros((2, 2))  # 初始化的混淆矩阵

        # 1.2 这是测试部分
        Model.eval()
        for imgs, targets in datas_test:
            preds = Model(imgs).argmax(1)
            acc = (preds == targets).sum()
            count_test += len(targets)
            total_acc_test += acc
            for one in range(targets.shape[0]):
                mat[int(targets[one].item()), preds[one]] += 1

        # 1.2.1 测试集的准确率
        acc_list.append(total_acc_test / count_test)
        if total_acc_test / count_test >= high_acc:
            high_acc = total_acc_test / count_test
            high_epoch = i
            best = Model
            high_mat = mat

        print('\rin fold time: {}, epoch: {}, train_acc:{}, test_acc->[{} / {}]<-highest_acc (epoch{})'.format(each_time,
                                                                                           i,
                                                                                           round(float(total_acc_train/count_train), 2),
                                                                                           round(float(total_acc_test/count_test), 2),
                                                                                           round(float(high_acc), 2),
                                                                                           high_epoch),
              end='')
    # 2.计算本次训练的相关评价参数
    mat_sum += high_mat.astype(dtype=int)

    print()

    if train_time <= 10:
        plt.figure()
        plt.subplot2grid((1, 2), (0, 0))
        plt.plot(time_list, acc_list, c='red')
        plt.plot(time_list, acc_list_train, c='blue')
        plt.title('accuracy')
        plt.subplot2grid((1, 2), (0, 1))
        plt.title('loss')
        plt.plot(time_list, loss_list_train, c='black')
        plt.show()


    print('in time: {}, highest epoch: {}, with highest acc: {}'.format(each_time + 1, high_epoch, high_acc))
    print(high_mat)
    ave_acc += high_acc

print('average accuracy of this model is', ave_acc/train_time)
print('roc:{}, sensitivity:{}, specificity:{}'.format(Judge.mat_roc(mat_sum),
                                                      Judge.mat_sensitivity(mat_sum),
                                                      Judge.mat_specify(mat_sum)))

