import numpy as np
import pandas as pd
from tqdm import tqdm
from config import config
from torch.utils import data
from DataProcess import DataPro
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def selection(X, y):
    w = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)
    num = X_train.shape[1]
    for i in tqdm(range(num), desc='特征选择', colour='green'):
        svcIn = SVC()
        svcOut = SVC()
        nbayesIn = GaussianNB()
        nbayesOut = GaussianNB()
        knnIn = KNeighborsClassifier(n_neighbors=config.n_neighbors)
        knnOut = KNeighborsClassifier(n_neighbors=config.n_neighbors)
        X_train_new = np.delete(X_train, i, axis=1)
        X_test_new = np.delete(X_test, i, axis=1)
        svcIn.fit(X_train, y_train)
        svcOut.fit(X_train_new, y_train)
        nbayesIn.fit(X_train, y_train)
        nbayesOut.fit(X_train_new, y_train)
        knnIn.fit(X_train, y_train)
        knnOut.fit(X_train_new, y_train)
        accIn = (svcIn.score(X_test, y_test) + nbayesIn.score(X_test, y_test) +
                 knnIn.score(X_test, y_test)) / float(3)
        accOut = (svcOut.score(X_test_new, y_test) + nbayesOut.score(X_test_new, y_test) +
                  knnOut.score(X_test_new, y_test)) / float(3)
        wi = accIn - accOut
        w.append(wi)
    w = np.array(w).reshape(-1)
    selected = np.where(w > 0)
    selected = np.array(selected).reshape(-1)
    w = w[selected]
    return selected, w
