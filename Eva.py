import numpy as np


def Evaluation(y, y_output):
    Acc = ((y[y_output > 0.5] == 1).sum() + (y[y_output < 0.5] == 0).sum()) / len(y)
    TP = (y[y_output > 0.5] == 1).sum()
    TN = (y[y_output < 0.5] == 0).sum()
    FP = (y[y_output > 0.5] == 0).sum()
    FN = (y[y_output < 0.5] == 1).sum()
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    mF1 = 2 * P * R / (P + R)
    TP = TP.detach().cpu().numpy()
    TN = TN.detach().cpu().numpy()
    FP = FP.detach().cpu().numpy()
    FN = FN.detach().cpu().numpy()
    G_mean = np.sqrt(TP / (TP + FN) * TN / (TN + FP))
    # G_mean = np.sqrt(TP / (TP + FN) * TN / (TN + FP))
    return round(Acc.item(), 4), round(G_mean.item(), 4), round(mF1.item(), 4), round(R.item(), 4), round(P.item(), 4)


def Evaluation2(y, y_output):
    s = 0
    num_L = 0
    num_M = 0
    num_H = 0
    num_N = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    num = len(y)
    for i in range(num):
        if(y_output[i] >= 7.0 and y_output[i] < 8.4):
            num_L += 1
            if(y[i] >= 7.0 and y[i] < 8.4):
                s = s + 1
                TP += 1
            else:
                FP += 1
        elif(y_output[i] >= 8.4 and y_output[i] < 11.1):
            num_M += 1
            if(y[i] >= 8.4 and y[i] < 11.1):
                s = s + 1
                TP += 1
            else:
                FP += 1
        elif(y_output[i] >= 11.1):
            num_H += 1
            if(y[i] >= 11.1):
                s = s + 1
                TP += 1
            else:
                FP += 1
        elif(y_output[i] < 7.0 and y_output[i] > 0):
            num_N += 1
            if(y[i] > 0 and y[i] < 7.0):
                s = s + 1
                TN += 1
            else:
                FN += 1

    Acc = s / len(y)
    #Acc = ((y[y_output > 0.5] == 1).sum() + (y[y_output < 0.5] == 0).sum()) / len(y)
    #num_L = max(1, (y_output >= 7.0 and y_output < 8.4).sum())
    #num_M = max(1, (y_output >= 8.4 and y_output < 11.1).sum())
    #num_H = max(1, (y_output >= 11.1).sum())
    #num_N = max(1, (y_output < 7.0).sum())
    num_P = num_L + num_M + num_H
    #num_P = max(1, (y_output > 0.5).sum())
    #num_N = max(1, (y_output < 0.5).sum())
    #TP = ((y[y_output >= 7.0 and y_output < 8.4] >= 7.0 and y[y_output >= 7.0 and y_output < 8.4] < 8.4)).sum() +\
    #      ((y[y_output >= 8.4 and y_output < 11.1] >= 8.4 and y[y_output >= 8.4 and y_output < 11.1] < 11.1)).sum() + \
    #      (y[y_output >= 11.1] >= 11.1).sum()
    #TN = (y[y_output < 7.0] < 7.0).sum()

    #FP = ((y[y_output < 7.0 or y_output >= 8.4] >= 7.0 and y[y_output < 7.0 or y_output >= 8.4] < 8.4)).sum() +\
    #      ((y[y_output < 8.4 or y_output >= 11.1] >= 8.4 and y[y_output < 8.4 or y_output >= 11.1] < 11.1)).sum() + \
    #      (y[y_output < 11.1] >= 11.1).sum()
    #FN = (y[y_output < 7.0] >= 7.0).sum()
    if(TP + FP == 0):
        P = 0
        print("TP + FP为0")
    else:
        P = TP / (TP + FP)
    if(TP + FN == 0):
        R = 0
        print("TP + FN为0")
    else:
        R = TP / (TP + FN)
    if(P + R == 0):
        mF1 = 0
        print("P + R为0")
    else:
        mF1 = 2 * P * R / (P + R)

    #TP = TP.detach().cpu().numpy()
    #TN = TN.detach().cpu().numpy()
    #FP = FP.detach().cpu().numpy()
    #FN = FN.detach().cpu().numpy()
    #G_mean = sqrt(TP / (TP + FN) * TN / (TN + FP)).detach().cpu().numpy()
    if(TP + FN == 0 or TN + FP == 0):
        G_mean = 0
        print("TP + FN == 0 or TN + FP == 0")
    else:
        G_mean = np.sqrt(TP / (TP + FN) * TN / (TN + FP))
    # G_mean = np.sqrt(TP / (TP + FN) * TN / (TN + FP))
    return round(Acc.item(), 4), round(G_mean.item(), 4), round(mF1, 4), round(R, 4), round(P.item(), 4)