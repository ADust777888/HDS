import numpy as np
import torch
from torch import nn

eps = 1e-8


class Net(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.hidden1 = nn.Linear(in_dim, 32)
        self.hidden2 = nn.Linear(32, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.hidden4 = nn.Linear(32, 16)
        self.hidden5 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x.view(-1)

def pearsonr(y_pred, y_true):
    # y_pred: [batch, seq]
    # y_true: [batch, seq]
    y_pred, y_true = y_pred.view(-1), y_true.view(-1)
    centered_pred = y_pred - torch.mean(y_pred)
    centered_true = y_true - torch.mean(y_true)
    covariance = torch.sum(centered_pred * centered_true)
    bessel_corrected_covariance = covariance / (y_pred.size(0) - 1)
    std_pred = torch.std(y_pred, dim=0)
    std_true = torch.std(y_true, dim=0)
    corr = bessel_corrected_covariance / (std_pred * std_true)
    return corr

def Loss(y, y_hat):
    # MSE
    Loss_1 = nn.MSELoss(reduction='mean')
    # MAE
    Loss_2 = nn.L1Loss()
    # Pearson Correlation
    Loss_3 = pearsonr(y_hat, y)
    loss = Loss_1(y_hat, y)
    return loss
