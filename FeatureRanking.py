import pandas as pd
from tqdm import tqdm
import numpy as np
from config import config
from DataProcess import DataPro
from torch.utils import data
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor

knn = 2

def sigma(i, NIFW, N, w):
    sum = 0
    for j in range(N):
        sum = sum + w[j] * NIFW[i][j]
    return float(sum)
def Rank(X, y, w):
    X_list = X.tolist()
    Transpose = list(map(list, zip(*X_list)))  # 训练数据转置
    neighbors = NearestNeighbors().fit(Transpose)
    tmp = np.array(Transpose)
    Max = -0x7f7f7f
    N = tmp.shape[0]  # 特征数量
    IFW = np.zeros((N, N), dtype=float)
    FR = []
    for m in tqdm(range(N), desc='计算特征秩FR', colour='green'):
        Kneighbors, dis = neighbors.kneighbors(np.array(Transpose[m]).reshape(1, -1), n_neighbors=knn,
                                               return_distance=True)
        sum = dis.sum()
        ave = sum / float(knn)
        neigh_dist, neigh_ind = neighbors.radius_neighbors(np.array(Transpose[m]).reshape(1, -1), radius=ave,
                                                           return_distance=True, sort_results=True)
        num = neigh_ind[0].shape[0]
        x1 = neigh_ind[0]
        for i in range(num):
            if (x1[i] != m):
                X_train, X_test, y_train, y_test = train_test_split(X, y)
                svcIn = SVC()
                svcOut = SVC()
                nbayesIn = GaussianNB()
                nbayesOut = GaussianNB()
                knnIn = KNeighborsClassifier(n_neighbors=knn)
                knnOut = KNeighborsClassifier(n_neighbors=knn)
                X_train_new = np.delete(X_train, [m, x1[i]], axis=1)
                X_test_new = np.delete(X_test, [m, x1[i]], axis=1)

                svcIn.fit(X_train, y_train)
                svcOut.fit(X_train_new, y_train)
                nbayesIn.fit(X_train, y_train)
                nbayesOut.fit(X_train_new, y_train)
                knnIn.fit(X_train, y_train)
                knnOut.fit(X_train_new, y_train)
                accIn = (svcIn.score(X_test, y_test) + nbayesIn.score(X_test, y_test) + knnIn.score(X_test,
                                                                                                    y_test)) / float(3)
                accOut = (svcOut.score(X_test_new, y_test) + nbayesOut.score(X_test_new, y_test) + knnOut.score(
                    X_test_new, y_test)) / float(3)
                IFWi = accIn - accOut
                IFW[m][x1[i]] = IFWi
                Max = max(Max, IFWi)
    NIFW = IFW / Max
    for i in range(N):
        FRi = config.lambd * w[i] + config.xi * sigma(i, NIFW, N, w)
        FRi = FRi * 100
        FR.append(FRi)
    return FR
