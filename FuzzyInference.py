import numpy as np
import pandas as pd

alpha = [1, 1, 1, 1, 1, 1]
beta = [2, 2, 2, 2, 2, 2]
gamma = [3, 3, 3, 3, 3, 3]

# 最后一列0为Low, 1为medium, 2为high的min()
# 前面的0为Low, 1为medium, 2为high,从mu_N_Low, mu_N_medium, mu_N_high中读数
rule = [[0,0,0,0,0],
        [0,0,0,1,0],
        [0,0,0,2,1],
        [0,0,1,0,0],
        [0,0,1,1,1],
        [0,0,1,2,0],
        [0,0,2,0,1],
        [0,0,2,1,0],
        [0,0,2,2,1],
        [0,1,0,0,0],

        [0,1,0,1,1],
        [0,1,0,2,0],
        [0,1,1,0,1],
        [0,1,1,1,1],
        [0,1,1,2,1],
        [0,1,2,0,0],
        [0,1,2,1,1],
        [0,1,2,2,2],
        [0,2,0,0,1],
        [0,2,0,1,0],

        [0,2,0,2,2],
        [0,2,1,0,0],
        [0,2,1,1,1],
        [0,2,1,2,2],
        [0,2,2,0,2],
        [0,2,2,1,2],
        [0,2,2,2,2],
        [1,0,0,0,0],
        [1,0,0,1,0],
        [1,0,0,2,0],

        [1,0,1,0,0],
        []]

# rows为模糊规则的条数
rows = len(rule)


# n为特征个数, X为一个sample的特征
def Fuzzy(n, X, FR):
    mu_low = []
    mu_medium = []
    mu_high = []
    for i in range(n):
        if (X[i] <= alpha[i]):
            mu_low.append(1.0)
            mu_medium.append(0.0)
            mu_high.append(0.0)
        elif (alpha[i] < X[i] < beta[i]):
            low = (beta[i] - X[i]) * float(1) / (beta[i] - alpha[i])
            mu_low.append(low)
            mu_medium.append(1 - low)
            mu_high.append(0.0)
        elif (beta[i] <= X[i] < gamma[i]):
            mu_low.append(0.0)
            medium = (gamma[i] - X[i]) * float(1) / (gamma[i] - beta[i])
            mu_medium.append(medium)
            mu_high.append(1 - medium)
        else:
            mu_low.append(0.0)
            mu_medium.append(0.0)
            mu_high.append(1.0)
    mu_R_low = [mu_low[i] * FR[i] for i in range(n)]
    mu_R_medium = [mu_medium[i] * FR[i] for i in range(n)]
    mu_R_high = [mu_high[i] * FR[i] for i in range(n)]
    mu_R_low = np.array(mu_R_low)
    mu_R_medium = np.array(mu_R_medium)
    mu_R_high = np.array(mu_R_high)
    mu_N_low = mu_R_low / float(max(mu_R_low))
    mu_N_medium = mu_R_medium / float(max(mu_R_medium))
    mu_N_high = mu_R_high / float(max(mu_R_high))

    # 最后一列0为Low, 1为medium, 2为high的min()
    # 前面的0为Low, 1为medium, 2为high,从mu_N_Low, mu_N_medium, mu_N_high中读数
    low_output = []
    medium_output = []
    high_output = []
    for i in range(rows):
        tmp = []
        for j in range(n):
            if (rule[i][j] == 0):
                tmp.append(mu_N_low[j])
            elif (rule[i][j] == 1):
                tmp.append(mu_N_medium[j])
            else:
                tmp.append(mu_N_high[j])
        Min = min(tmp)
        if (rule[rows - 1][n] == 0):
            low_output.append(Min)
        elif (rule[rows - 1][n] == 1):
            medium_output.append(Min)
        else:
            high_output.append(Min)

    lowMembership = max(low_output)
    mediumMembership = max(medium_output)
    highMembership = max(high_output)

    COG = (lowMembership * (1 + 2 + 3 + 4) + mediumMembership * (5 + 6 + 7 + 8) +
           highMembership * (9 + 10 + 11 + 12)) / (4 * (lowMembership + mediumMembership + highMembership))
    a, b, c = 3.835, 5.836, 7.836

    if (COG <= a):
        Pnormal = 1
        Pinfected = 0
    elif (a < COG < c):
        Pnormal = (c - COG) / float(c - a)
        Pinfected = 1 - Pnormal
    else:
        Pnormal = 0
        Pinfected = 1
    return Pnormal, Pinfected
