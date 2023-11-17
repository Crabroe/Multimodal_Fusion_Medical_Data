import random
import scipy.io as scio
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
import torch
from torch.utils.data import DataLoader
import torchvision

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

class Data():
    def __init__(self, file_root='', setNumber=1, method=None, fold=10, valiSet=True, batch_size=21):
        if method is None:
            method = ['Seq', 'Cor']
        self.fold = fold
        self.file_root = ''
        self.Samples, self.Labels = self.load(file_root, setNumber)
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.method = method
        self.valiSet = valiSet
        self.batch_size = batch_size
        self.Fold_Datasets, self.Test_Iter = self.init_Datasets()


    def load(self, file_root, file_order, binary=True):
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

    def process(self, method: list):
        # 这是一个数据处理函数，将返回处理后的Samples和Labels，类型为List
        def process_single(data: np.ndarray, label: np.ndarray, method):
            if method == 'Seq':
                data = data[:, 20:180]
                data = self.min_max_scaler.fit_transform(data[:, 0:])
                data = data.astype(np.float32)

            if method == 'Cor':
                data = data[:, 20:]
                data = self.min_max_scaler.fit_transform(data[:, 0:])
                data = np.corrcoef(data)
                data = np.tril(data)

                len = data.shape[0]
                new_data = []
                for i in range(len):
                    for j in range(i):
                        new_data.append(data[i, j])

                data = np.asarray([new_data])
                data = data.astype(np.float32)

            return data, label

        Datas_alter, Labels_alter = [], []

        for twin in zip(self.Samples, self.Labels):
            # 处理数据集中的每一对样本和标签
            data = []
            first = True
            for each_method in method:
                data_one_method, label = process_single(twin[0], twin[1], each_method)  # 该步骤即处理data和label

                if len(method) > 1:
                    data.append(data_one_method)
                else:
                    data = data_one_method

                if first:
                    first = False
                    Labels_alter.append(label)

            Datas_alter.append(data)


        return Datas_alter, Labels_alter

    def split_test(self, Samples, Labels):

        X_train, X_test, y_train, y_test = train_test_split(Samples, Labels, test_size=1/self.fold)

        return X_train, X_test, y_train, y_test

    def get_fold(self, Samples, Labels):
        print(type(Samples))
        print(type(Labels))
        if self.valiSet:
            kf = KFold(n_splits=self.fold-1, shuffle=True)
        else:
            kf = KFold(n_splits=self.fold, shuffle=True)

        datasets = []
        for train_index, test_index in kf.split(Samples, Labels):
            # print(train_index, test_index)
            X_train, X_test = np.asarray(Samples, dtype=object)[train_index].tolist(), np.asarray(Samples, dtype=object)[test_index].tolist()
            y_train, y_test = np.asarray(Labels, dtype=object)[train_index].tolist(), np.asarray(Labels, dtype=object)[test_index].tolist()

            dataset_train = self.get_dataloader(X_train, y_train)
            dataset_test = self.get_dataloader(X_test, y_test)
            datasets.append([dataset_train, dataset_test])
        return datasets

    def init_Datasets(self):
        Samples, Labels = self.process(self.method)

        if self.valiSet:
            X_tv, X_test, y_tv, y_test = self.split_test(Samples, Labels)
            test_iter = self.get_dataloader(X_test, y_test)
            fold_datasets = self.get_fold(X_tv, y_tv)
        else:
            test_iter = None
            fold_datasets = self.get_fold(Samples, Labels)

        return fold_datasets, test_iter

    def get_dataloader(self, Datas, Labels, drop_last=False):
        # 该函数用于返回用于batch批量迭代训练形式的数据，在训练神经网络时可直接使用
        # 接收ndarray形式的Datas和Labels
        # return (dataloader)data_pack
        # 对返回的数据包使用for循环，每一个循环中可X, y = each解包得到tensor数据类型的样本和标签
        dataset = GetLoader(Datas, Labels)
        data_pack = DataLoader(dataset, batch_size=self.batch_size,
                               shuffle=True, drop_last=drop_last)
        return data_pack

if __name__ == '__main__':
    data = Data(file_root='', setNumber=2, fold=10, valiSet=False, batch_size=21)

    Fold, test_iter = data.Fold_Datasets, data.Test_Iter

    for fold in Fold:
        train_iter, vali_iter = fold
        for train in train_iter:
            [X1, X2], y = train
            print(X1.shape, X2.shape)
        print('-------------------')
        for test in vali_iter:
            [X1, X2], y = test
            print(y)
        print('===================')