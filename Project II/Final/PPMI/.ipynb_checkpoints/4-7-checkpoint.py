# 分类ADNI_90_120的多模态神经网络 - 三模型合并

import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import classification_report

import Loader_forPPMI as Loader
import Judge
import time
from Models.GRU import Model as RNN
from Models.Net import Model as MLP
from Models.TOP import Model as TOP
from sklearn.svm import SVC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


LR = 0.0005
EPOCH = 300
n_fold = 20
TURN = 5
if_PRINT = True
if_valiSet = False

acc_list_total = []
best_roc_list, best_sen_list, best_spe_list = [], [], []
acc_TURN = []
for turn in range(TURN):
    # I.准备数据
    Fold_Datasets = Loader.get(n_splits=n_fold, drop=0.9)

    acc_list_test = []
    for each_Fold in Fold_Datasets:
        time_ST = time.perf_counter()

        # 0.模型实例化
        model1 = RNN(90, 45, 2).to(device)
        model2 = MLP().to(device)
        model3 = TOP().to(device)

        loss_fn = nn.CrossEntropyLoss().to(device)
        optim = torch.optim.Adam(list(model2.parameters()), lr=LR)   # 此处做修改
        acc_list_turn = []

        # 1.作图需要的参数
        _X = list(range(EPOCH))
        _Y_train = []
        _Y_test = []

        # 2.最终评价指标
        good_acc = 0
        good_con_mat = None
        good_epoch = 0
        best_model_RNN = None


        # 3.数据集
        train_iter, vali_iter = each_Fold

        # 4.开始训练
        for i in range(EPOCH):

            # 4.1 ------TRAIN PART-------
            model1.train()
            model2.train()
            model3.train()

            acc_for_train, count = 0, 0
            for train_batch in train_iter:
                [X1, X2, X3], labels = train_batch
                X1, X2, X3,  labels = X1.to(device),\
                                      X2.to(device),\
                                      X3.to(device),\
                                      labels.to(device)

                # outputs_seq = model1(samples_seq)
                outputs_cor = model2(X1, X2, X3)
                # outputs = model3(outputs_seq, outputs_cor)
                outputs = outputs_cor

                # recording acc
                preds = outputs.cpu().argmax(1)
                acc = (preds == labels.cpu()).sum()
                count += len(labels)
                acc_for_train += acc

                # loss computing, backward
                labels = labels.long()
                loss = loss_fn(outputs, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
            _Y_train.append(torch.true_divide(acc_for_train, count))

            # 4.2 ------VALIDATION PART-------
            model1.eval()
            model2.eval()
            model3.eval()
            acc_for_vali, count = 0, 0
            con_mat_vali = np.zeros((2, 2))

            for vali_batch in vali_iter:
                [X1, X2, X3], labels = vali_batch
                X1, X2, X3 = X1.to(device), X2.to(device), X3.to(device)

                # outputs_seq = model1(samples_seq)
                outputs_cor = model2(X1, X2, X3)
                # outputs = model3(outputs_seq, outputs_cor)
                outputs = outputs_cor   # TODO:去掉

                preds = outputs.cpu().argmax(1)
                acc = (preds == labels).sum()
                count += len(labels)
                acc_for_vali += acc
                for one in range(preds.shape[0]):
                    con_mat_vali[int(labels[one].item()), preds[one]] += 1

            _Y_test.append(torch.true_divide(acc_for_vali, count))

            # 4.2.1 打印总结
            print('\rin epoch {}, vali acc = {} / {}(epoch:{})'.format(i, round(float(torch.true_divide(acc_for_vali, count)), 2),
                                                                       round(float(good_acc), 2), good_epoch), end='')
            # print(con_mat)
            acc_list_turn.append(torch.true_divide(acc_for_vali, count))

            # 4.2.2 记录最好的值
            if torch.true_divide(acc_for_vali, count) >= good_acc:
                good_acc = torch.true_divide(acc_for_vali, count)
                good_con_mat = con_mat_vali
                good_epoch = i
                best_model_RNN = model2
        print('\n', good_con_mat)

        # II.训练SVM
        SVMmodel = SVC(kernel='linear')

        acc_list_test.append(-1)

        time_ED = time.perf_counter()
        print(' time: {}s'.format(round((time_ED - time_ST), 4)))

        # 6.记录所有的评价参数
        try:
            good_roc = Judge.mat_roc(good_con_mat.astype(dtype=int))
            good_sensitivity = Judge.mat_sensitivity(good_con_mat.astype(dtype=int))
            good_specificity = Judge.mat_specify(good_con_mat.astype(dtype=int))

            best_roc_list.append(good_roc)
            best_sen_list.append(good_sensitivity)
            best_spe_list.append(good_specificity)
        except:
            print('本次AUC参数计算有误，因为随机分配的测试集仅含一类样本')

        acc_list_total.append(max(acc_list_turn))

        if if_PRINT:
            # plt
            plt.figure()

            plt.ylim([0.0, 1.0])

            plt.plot(_X, _Y_test, c='red')
            plt.plot(_X, _Y_train, c='blue')

            plt.show()
    print(np.mean(np.asarray(acc_list_total)))
    print('roc:{}, sensitivity:{}, specificity:{}, acc_test:{}'.format(np.mean(np.asarray(best_roc_list)),
                                                          np.mean(np.asarray(best_sen_list)),
                                                          np.mean(np.asarray(best_spe_list)),
                                                                       np.mean(np.asarray(acc_list_test))))
    acc_TURN.append(np.mean(np.asarray(acc_list_total)))
print('\nof {} turns, the average acc is {}'.format(TURN, np.mean(np.asarray(acc_TURN))))