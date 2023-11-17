import random

import scipy.io as scio
import numpy as np
import torch, torch.nn as nn
from sklearn import preprocessing
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

min_max_scaler = preprocessing.MinMaxScaler()

class GetLoader(torch.utils.data.Dataset):

    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

def load(file_root='', file_order=1, binary=True):
    # 该函数将输入file_root + file_name[file_order]的文件，按两类(binary=True)或多类(binary=False)读取
    # return：(ndarray)3维样本矩阵, (ndarray)1维标签向量

    # 0.1 所有数据集的.mat文件名
    file_name = ['ADNI.mat',
                 'ADNI_90_120_fMRI.mat',
                 'FTD_90_200_fMRI.mat',
                 'OCD_90_200_fMRI.mat',
                 'PPMI.mat']

    # 0.2 所有数据集的样本类型名
    Sample_catagories = [['AD', 'MCI', 'MCIn', 'MCIp', 'NC'],
                         ['AD', 'EMCI', 'LMCI', 'NC'],
                         ['FTD', 'NC'],
                         ['OCD', 'NC'],
                         ['PD', 'NC']]

    # 1.读取数据
    dataSet = scio.loadmat(file_root + file_name[file_order])
    Sample_catagory = Sample_catagories[file_order]

    # 2.将数据打包装入
    Datas, Labels = [], []
    if binary:
        data_OP = dataSet[Sample_catagory[0]]
        data_NC = dataSet[Sample_catagory[-1]]

        for i in range(data_OP.shape[0]):
            Datas.append(data_OP[i])
        for i in range(data_NC.shape[0]):
            Datas.append(data_NC[i])

        Labels_OP = np.ones((1, data_OP.shape[0]))
        Labels_NC = np.zeros((1, data_NC.shape[0]))

        Labels = np.asarray(Labels_OP.tolist()[0] + Labels_NC.tolist()[0], dtype=float)
    else:
        catagory_number = 0
        for key in Sample_catagory:
            datas = dataSet[key]
            for i in range(datas.shape[0]):
                Datas.append(datas[i])
                Labels.append(catagory_number)
            catagory_number += 1
        Labels = np.asarray(Labels, dtype=float)

    # 3.Datas和Labels都是ndarray类型的数据
    return Datas, Labels

datas, labels = load(file_order=4)
need_datas, need_labels = [], []
OP_num, NG_num = 0, 0
for i in range(len(datas)):

    print('\r OP{}, NG{}'.format(OP_num, NG_num), end='')
    need_datas.append(datas[i])
    need_labels.append(labels[i])
datas = np.asarray(need_datas).astype(np.float32)
labels = need_labels

# 1.将每一个样本分开，分别处理后装入一个列表当中
new_datas = []
for each in range(datas.shape[0]):
    dimensions = []
    for i in range(3):
        # 数据的预处理部分
        each_dimension = datas[each][i*98: (i+1)*98]
        # each_dimension = (each_dimension - each_dimension.min()) / (each_dimension.max() - each_dimension.min())
        dimensions.append(each_dimension)
    '''
    data = np.corrcoef(data)
    data = np.tril(data)
    length = data.shape[0]
    new_data = []
    for i in range(length):
        for j in range(i):
            if data[i, j] >= np.mean(data):
                new_data.append(data[i, j])
            else:
                new_data.append(0)
    data = np.asarray(new_data)
    data = data.astype(np.float32)
    dimensions.append(data)
    '''
    new_datas.append(dimensions)

# 2.将样本按k折交叉验证的方式装入dataloader，再装入列表当中
len_sample = len(new_datas)
order_sample = []
for i in range(len_sample):
    order_sample.append(i)

# 2.1 封装过程，包括训练集样本平衡过程
def get(batch_size=49, n_splits=10, drop=0.6):
    Folds = []
    kf = KFold(n_splits=n_splits, shuffle=True)
    for train_index, test_index in kf.split(order_sample):
        samples_train, labels_train = [], []
        samples_test, labels_test = [], []

        for i in train_index.tolist():
            R = random.random()
            if labels[i] == 1 and R > drop:
                # samples_test.append(new_datas[i])
                # labels_test.append(labels[i])
                continue
            else:
                samples_train.append(new_datas[i])
                labels_train.append(labels[i])

        for i in test_index.tolist():
            R = random.random()

            samples_test.append(new_datas[i])
            labels_test.append(labels[i])

        train_dataset = GetLoader(samples_train, labels_train)
        train_dataset = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, drop_last=True)
        test_dataset = GetLoader(samples_test, labels_test)
        test_dataset = DataLoader(test_dataset, batch_size=batch_size,
                                   shuffle=True, drop_last=False)

        Folds.append([train_dataset, test_dataset])

    return Folds


Folds = get(10)
if __name__ == '__main__':
    for fold in Folds:
        train_iter, test_iter = fold
        for train_batch in train_iter:
            print(train_batch[0][0].shape)
            quit()