import math
import numpy as np
import torch
from DNN_model import do_train
from data_generator import generate_samples_random
from statsmodels.distributions.empirical_distribution import ECDF
import BET
import scipy.stats as stats
from scipy.stats import rankdata

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def get_m_out_n_tuples(m, n, B):
    n_groups = n // m
    n_perms = math.ceil(B / n_groups)
    temp = [np.random.choice(n, size=(n_groups, m), replace=False) for i in range(B)]
    result = np.vstack(temp)
    return result[0:B, ]


def sim_worker(n_sample, sType, dx, dy, dz, nstd, alpha_x, norm, dist_z):
    X_h0, Y_h0, Z_h0 = generate_samples_random(size=n_sample, sType=sType, dx=dx, dy=dy, dz=dz, nstd=nstd,
                                                           alpha_x=alpha_x,
                                                           normalize=norm, seed=None, dist_z=dist_z)

    Y_h0 = torch.tensor(Y_h0, dtype=torch.float32).squeeze()
    X_h0 = torch.tensor(X_h0, dtype=torch.float32).squeeze()
    Z_h0 = torch.tensor(Z_h0, dtype=torch.float32).squeeze()

    B = 30
    n = 1000
    num = 0.8 * n
    L = 30
    m = 300
    m_sample = 0.8 * m

    sample_index = get_m_out_n_tuples(m, n, B)
    result = []
    for b in range(B):
        bet_max = []
        for l in range(L):
            Yh0_pred, Xh0_pred, Yh0_test, Xh0_test = do_train(Y_h0[sample_index[b]], X_h0[sample_index[b]],
                                                              Z_h0[sample_index[b]])
            epsilon_Yh0 = Yh0_test - Yh0_pred
            epsilon_Xh0 = Xh0_test - Xh0_pred
            U = ECDF(epsilon_Xh0)
            V = ECDF(epsilon_Yh0)
            bet_max_abs = BET.BETs(U(epsilon_Xh0), V(epsilon_Yh0), max_depth=2, print_res=False).bets_max
            bet_max.append(bet_max_abs)
        result.append(bet_max)
    result = np.array(result)
    Tn = ((result + m_sample) / 4 - m_sample / 4) / np.sqrt(m_sample ** 2 / (16 * (m_sample - 1)))

    H_tilda = np.abs(stats.norm.ppf((rankdata(Tn, method='dense').reshape(B, L) - 1 / 2) / (B * L), loc=0, scale=1))
    S_b_tilda = np.mean(H_tilda, axis=1)

    ###### calculate stats on full data

    bet_max = []
    for l in range(L):
        Yh0_pred, Xh0_pred, Yh0_test, Xh0_test = do_train(Y_h0, X_h0, Z_h0)
        epsilon_Yh0 = Yh0_test - Yh0_pred
        epsilon_Xh0 = Xh0_test - Xh0_pred
        U = ECDF(epsilon_Xh0)
        V = ECDF(epsilon_Yh0)
        bet_max_abs = BET.BETs(U(epsilon_Xh0), V(epsilon_Yh0), max_depth=2, print_res=False).bets_max
        bet_max.append(bet_max_abs)
    result_all = np.array(bet_max)
    result_n = (result_all + num) / 4
    std_Tn_all = np.abs((result_n - num / 4) / np.sqrt(num ** 2 / (16 * (num - 1))))
    test_stat = np.mean(std_Tn_all)

    ###### make decision
    final_result = 1 if test_stat > np.percentile(S_b_tilda, 97.5) else 0
    return final_result


def sim_worker_h1(n_sample, sType, dx, dy, dz, nstd, alpha_x, norm, dist_z):
    X_h1, Y_h1, Z_h1 = generate_samples_random(size=n_sample, sType=sType, dx=dx, dy=dy, dz=dz, nstd=nstd,
                                                           alpha_x=alpha_x,
                                                           normalize=norm, seed=None, dist_z=dist_z)
    Y_h1 = torch.tensor(Y_h1, dtype=torch.float32).squeeze()
    X_h1 = torch.tensor(X_h1, dtype=torch.float32).squeeze()
    Z_h1 = torch.tensor(Z_h1, dtype=torch.float32).squeeze()

    B = 30
    n = 500
    num = 0.8 * n
    L = 30
    m = 200
    m_sample = 0.8 * m

    sample_index = get_m_out_n_tuples(m, n, B)
    result = []
    for b in range(B):
        bet_max = []
        for l in range(L):
            Yh1_pred, Xh1_pred, Yh1_test, Xh1_test = do_train(Y_h1[sample_index[b]], X_h1[sample_index[b]],
                                                              Z_h1[sample_index[b]])
            epsilon_Yh1 = Yh1_test - Yh1_pred
            epsilon_Xh1 = Xh1_test - Xh1_pred
            U = ECDF(epsilon_Xh1)
            V = ECDF(epsilon_Yh1)
            bet_max_abs = BET.BETs(U(epsilon_Xh1), V(epsilon_Yh1), max_depth=2, print_res=False).bets_max
            bet_max.append(bet_max_abs)
        result.append(bet_max)
    result = np.array(result)
    Tn = ((result + m_sample) / 4 - m_sample / 4) / np.sqrt(m_sample ** 2 / (16 * (m_sample - 1)))

    H_tilda = np.abs(stats.norm.ppf((rankdata(Tn, method='dense').reshape(B, L) - 1 / 2) / (B * L), loc=0, scale=1))
    S_b_tilda = np.mean(H_tilda, axis=1)

    ###### calculate stats on full data

    bet_max = []
    for l in range(L):
        Yh1_pred, Xh1_pred, Yh1_test, Xh1_test = do_train(Y_h1, X_h1, Z_h1)
        epsilon_Yh1 = Yh1_test - Yh1_pred
        epsilon_Xh1 = Xh1_test - Xh1_pred
        U = ECDF(epsilon_Xh1)
        V = ECDF(epsilon_Yh1)
        bet_max_abs = BET.BETs(U(epsilon_Xh1), V(epsilon_Yh1), max_depth=2, print_res=False).bets_max
        bet_max.append(bet_max_abs)
    result_all = np.array(bet_max)
    result_n = (result_all + num) / 4

    std_Tn_all = np.abs((result_n - num / 4) / np.sqrt(num ** 2 / (16 * (num - 1))))
    test_stat = np.mean(std_Tn_all)

    ###### make decision
    final_result = 1 if test_stat > np.percentile(S_b_tilda, 97.5) else 0
    return final_result