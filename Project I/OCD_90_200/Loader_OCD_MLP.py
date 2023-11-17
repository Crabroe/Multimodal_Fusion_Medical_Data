# 这个函数包是标准的函数读取库

import scipy.io as scio
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
import torch
from torch.utils.data import DataLoader
import torchvision


def load(file_root, file_order, binary=True):
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
    Datas = np.asarray(Datas)

    return Datas, Labels

def sample_split(Datas, Labels, test_size=0.1):
    # 该函数用于训练样本和测试样本的切割，接受ndarray形式的Datas，ndarray形式的Labels
    # return: (ndarray)X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(Datas, Labels, test_size=test_size)
    return X_train, X_test, y_train, y_test

def sample_kFold_split(Datas, Labels, split_num=10):
    kf = KFold(n_splits=split_num, shuffle=True)

    datasets = []
    for train_index, test_index in kf.split(Datas, Labels):
        X_train, X_test = Datas[train_index], Datas[test_index]
        y_train, y_test = Labels[train_index], Labels[test_index]

        X_train, X_test, y_train, y_test = np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)

        dataset_train = get_dataloader(X_train, y_train)
        dataset_test = get_dataloader(X_test, y_test)
        datasets.append([dataset_train, dataset_test])
    return datasets

def process_all(Datas: np.ndarray, Labels: np.ndarray):
    # 该函数用于[整组]样本数据的处理，将样本数据读取为ndarray的形式，即可传入该函数进行样本数据处理
    # return: (ndarray)Datas_alter, (ndarray)Labels_alter
    # 该函数根据需要重写，但对于单个样本的处理，可以改变progress_single函数
    Datas_alter = []
    Labels_alter = []
    for each in zip(Datas, Labels):
        data, label = each
        data, label = process_single(data, label)   # 该步骤即处理data和label
        Datas_alter.append(data)
        Labels_alter.append(label)

    Datas_alter = np.asarray(Datas_alter)
    Labels_alter = np.asarray(Labels_alter)

    return Datas_alter, Labels_alter

# 若更改单个样本的处理方式，修改该函数↓
def process_single(Data: np.ndarray, Label: np.ndarray):
    # 该函数用于[单个]样本数据的处理，对于单个样本的处理，可以在次函数中编写
    # return: (ndarray, float32)Data, (ndarray)Label
    # 注意，返回的单个样本应为2维矩阵的形式
    data = Data[:, 40:]

    data = min_max_scaler.fit_transform(data[:, 0:])

    data = np.corrcoef(data)

    data = np.tril(data)

    len = data.shape[0]
    new_data = []
    for i in range(len):
        for j in range(i):
            new_data.append(data[i, j])

    data = np.asarray([new_data])
    Data = data.astype(np.float32)

    return Data, Label

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

def get_dataloader(Datas, Labels, batch_size=31, shuffle=True, drop_last=False):

    # 该函数用于返回用于batch批量迭代训练形式的数据，在训练神经网络时可直接使用
    # 接收ndarray形式的Datas和Labels
    # return (dataloader)data_pack
    # 对返回的数据包使用for循环，每一个循环中可X, y = each解包得到tensor数据类型的样本和标签
    dataset = GetLoader(Datas, Labels)
    data_pack = DataLoader(dataset, batch_size=batch_size,
                           shuffle=shuffle, drop_last=drop_last)
    return data_pack

# def YiTiaoLong():
#     # 集读取数据、数据预处理、样本拆分、转换张量所有功能的一条龙服务[/滑稽]
#     Datas, Labels = load('../datasets_class/', 1, binary=True)
#     Datas, Labels = process_all(Datas, Labels)
#     X_train, X_test, y_train, y_test = sample_split(Datas, Labels)
#     dataset_train = get_dataloader(X_train, y_train)
#     dataset_test = get_dataloader(X_test, y_test)
#
#     return dataset_train, dataset_test
#
# def RNN_YiTiaoLong(test_size=0.1):
#     Datas, Labels = load('../datasets_class/', 1, binary=True)  # 从文件夹中取数据
#     Datas, Labels = process_all(Datas, Labels)  # 数据处理
#     X_train, X_test, y_train, y_test = sample_split(Datas, Labels, test_size)   # 数据分割
#
#     dataset_train = get_dataloader(X_train, y_train)
#     datasets_class = get_dataloader(X_test, y_test)
#
#     return dataset_train, datasets_class

def YTL_Fold(n_split=10):
    Datas, Labels = load('', 3, binary=True)
    Datas, Labels = process_all(Datas, Labels)
    datasets = sample_kFold_split(Datas, Labels, n_split)

    return datasets

# ------------实例化后的工具库----------------
min_max_scaler = preprocessing.MinMaxScaler()
# -------------------------------------------
