import torch
import os
import numpy as np
import pandas as pd
from torch.utils import data
from config import config
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SVMSMOTE
from Smote import Smote
eps = 1e-8


class DataPro(data.Dataset):
    def __init__(self, FILE_NAME):
        '''
        myData_ = pd.read_csv(FILE_NAME, header=None)

        #print(myData_.describe())
        #print(myData_.info())
        myData_ = myData_.to_numpy()
        myData_ = myData_[:1000, :]
        print(myData_.shape)

        X = myData_[:, :-1]
        y = myData_[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y)


        #smo = SVMSMOTE()
        #X_smo, y_smo = smo.fit_resample(X_train, y_train)
        #print(type(X_train))

        self.train_data = X_train
        self.train_label = y_train
        self.test_data = X_test
        self.test_label = y_test
        self.train_data = (self.train_data - self.train_data.mean(axis=0, keepdims=True)) \
                          / (self.train_data.std(axis=0, keepdims=True) + eps)
        self.test_data = (self.test_data - self.test_data.mean(axis=0, keepdims=True)) \
                         / (self.test_data.std(axis=0, keepdims=True) + eps)
        '''

        Data = pd.read_csv(FILE_NAME, header=None).to_numpy()
        #Data = Data[:2000,:]

        np.random.shuffle(Data)

        Pos = Data[Data[:, -1] == 1].astype(np.float32)
        Neg = Data[Data[:, -1] == 0].astype(np.float32)

        P_len = Pos.shape[0]
        N_len = Neg.shape[0]

        if (P_len < N_len):
            tmp = Pos[:, :-1]
            tmp = Smote(tmp, N=int(N_len/P_len)).over_sampling()
            len1 = tmp.shape[0]
            len2 = tmp.shape[1]
            tmp2 = np.zeros((len1, len2 + 1))
            for i in range(len1):
                for j in range(len2):
                    tmp2[i][j] = tmp[i][j]
                tmp2[i][len2] = 1
            Pos = np.r_[Pos, tmp2]
        else:
            tmp = Neg[:, :-1]
            tmp = Smote(tmp, N=int(P_len/N_len)).over_sampling()
            len1 = tmp.shape[0]
            len2 = tmp.shape[1]
            tmp2 = np.zeros((len1, len2 + 1))
            for i in range(len1):
                for j in range(len2):
                    tmp2[i][j] = tmp[i][j]
                tmp2[i][len2] = 0
            Neg = np.r_[Neg, tmp2]
        Pos = Pos.astype(np.float32)
        Neg = Neg.astype(np.float32)

        self.num = Pos.shape[0] + Neg.shape[0]
        data1 = int(Pos.shape[0] * 0.8)
        data2 = int(Neg.shape[0] * 0.8)

        self.train_data = np.r_[Pos[:data1, :-1], Neg[:data2, :-1]]
        self.train_label = np.r_[Pos[:data1, -1], Neg[:data2, -1]]
        self.test_data = np.r_[Pos[data1:, :-1], Neg[data2:, :-1]]
        self.test_label = np.r_[Pos[data1:, -1], Neg[data2:, -1]]
        self.train_data = (self.train_data - self.train_data.mean(axis=0, keepdims=True)) \
                          / (self.train_data.std(axis=0, keepdims=True) + eps)
        self.test_data = (self.test_data - self.test_data.mean(axis=0, keepdims=True)) \
                         / (self.test_data.std(axis=0, keepdims=True) + eps)


    def __getitem__(self, index):
        return self.train_data[index], self.train_label[index]

    def __len__(self):
        return len(self.train_data)
