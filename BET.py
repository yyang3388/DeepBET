#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import List
import numpy as np
import pandas as pd
from scipy import stats
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def fraction_to_binary(a: float, d: int) -> List[int]:
    count = 0
    res = []
    while count < d:
        if a * 2 >= 1:
            res.append(1)
            a = a * 2 - 1
            count += 1
        else:
            a *= 2
            res.append(0)
            count += 1
    return res


# In[3]:


def bex_centers(depth=3):
    centers = np.vstack(
        (np.array([[x / (2 ** (depth + 1)) for x in range(2 ** (depth + 1) - 1, -1, -2)] * (2 ** depth)]),
         np.array([[x / (2 ** (depth + 1))] * (2 ** depth) for x in range(2 ** (depth + 1) - 1, -1, -2)]).reshape(1, (
                     2 ** depth) ** 2))).T
    return centers


# In[4]:


def bet_plot(beidx1: str, beidx2: str, depth=3):
    xyc = bex_centers(depth)
    BEy = [fraction_to_binary(x, depth) for x in xyc[:, 0]]
    BEx = [fraction_to_binary(y, depth) for y in xyc[:, 1]]

    RDx = 2 * np.array(BEx) - 1
    RDy = 2 * np.array(BEy) - 1

    beidx1_num = [int(x) - 1 for x in beidx1.split(':')]
    x_prod = np.prod(RDx[:, beidx1_num], axis=1)
    beidx2_num = [int(x) - 1 for x in beidx2.split(':')]
    y_prod = np.prod(RDy[:, beidx2_num], axis=1) * (-1) ** (len(beidx2_num) - 1)

    col_idx = x_prod * y_prod

    N = 2 ** depth + 1
    X, Y = np.mgrid[0:1:complex(0, N), 0:1:complex(0, N)]
    Z = col_idx.reshape(N - 1, N - 1)

    fig, ax0 = plt.subplots(figsize=(5, 5))
    c = ax0.pcolor(X, Y, Z, cmap=ListedColormap(['white', np.array([0, 0, 1, 1 / 4])]), edgecolors=None)
    ax0.set_xlabel(r'$U_x$')
    ax0.set_ylabel(r'$U_y$')

    return ax0


# In[2]:


class BET:
    def __init__(self, x, y, depth=3, plot=False, print_res=True):

        n = len(x)
        BEx = [fraction_to_binary(a, d=depth) for a in x]
        BEy = [fraction_to_binary(a, d=depth) for a in y]

        RDx = 2 * np.array(BEx) - 1
        RDy = 2 * np.array(BEy) - 1

        BEx_complete, BEy_complete = np.zeros(shape=(n, 1)), np.zeros(shape=(n, 1))
        BEx_complete_colnames, BEy_complete_colnames = [], []

        for ind in range(1, depth + 1):
            sets = list(itertools.combinations(list(range(1, depth + 1)), ind))
            if not sets:
                nint = 1
                sets = np.array(sets)  # reshape
            else:
                for cb in sets:
                    temp1 = np.prod(RDx[np.ix_(list(range(n)), [(x - 1) for x in list(cb)])], axis=1)
                    BEx_complete = np.c_[BEx_complete, temp1]
                    BEx_complete_colnames.append(":".join(tuple(str(i) for i in cb)))

                    temp2 = np.prod(RDy[np.ix_(list(range(n)), [(x - 1) for x in list(cb)])], axis=1)
                    BEy_complete = np.c_[BEy_complete, temp2]
                    BEy_complete_colnames.append(":".join(tuple(str(i) for i in cb)))

        self.BEx_complete = np.delete(BEx_complete, 0, axis=1)
        self.BEy_complete = np.delete(BEy_complete, 0, axis=1)

        count_interaction = np.dot(self.BEx_complete.T, self.BEy_complete)
        self.count_interaction_df = pd.DataFrame(count_interaction)  # return
        self.count_interaction_df.columns, self.count_interaction_df.index = BEy_complete_colnames, BEx_complete_colnames

        #         if count_interaction_df.shape[0] == 1:
        #             count_interaction_df.columns, count_interaction_df.index = "1", "1"

        max_abs_count_interaction = np.max(np.abs(count_interaction))
        max_ind = np.where(abs(count_interaction) == max_abs_count_interaction)
        self.bet_max = count_interaction[max_ind[0][0], max_ind[1][0]]
        self.max_index = [self.count_interaction_df.index[max_ind[0][0]],
                          self.count_interaction_df.columns[max_ind[1][0]]]  # return
        self.strongest_asymmetry = count_interaction[max_ind[0][0], max_ind[1][0]]  # return

        self.p_value = min(
            (2 ** depth - 1) ** 2 * 2 * (1 - stats.binom.cdf((max_abs_count_interaction + n) / 2 - 1, n, 0.5)), 1)
        # return

        table22 = [[max_abs_count_interaction / 4 + n / 4, -max_abs_count_interaction / 4 + n / 4],
                   [-max_abs_count_interaction / 4 + n / 4, max_abs_count_interaction / 4 + n / 4]]
        odd_ratio, fpvalue = stats.fisher_exact(table22)
        self.FE_p_value = min(
            (2 ** depth - 1) ** 2 * (fpvalue - stats.hypergeom.pmf(table22[0][0], n, n / 2, n / 2) / 2), 1)  # return

        self.chisq_stat = np.sum(count_interaction ** 2) / n  # return
        self.chisq_pvalue = 1 - stats.chi2.cdf(self.chisq_stat, (2 ** depth - 1) ** 2)  # return

        cell_BEx = np.dot(BEx, np.array([2 ** x for x in range(depth - 1, -1, -1)]).reshape((-1, 1)))
        cell_BEy = np.dot(BEy, np.array([2 ** x for x in range(depth - 1, -1, -1)]).reshape((-1, 1)))
        self.cell_store = np.zeros(shape=(2 ** depth, 2 ** depth))

        for i in range(n):
            self.cell_store[cell_BEx[i][0]][cell_BEy[i][0]] += 1

        cell_c = np.sum(self.cell_store, axis=1)[np.sum(self.cell_store, axis=1) > 0]
        cell_r = np.sum(self.cell_store, axis=0)[np.sum(self.cell_store, axis=0) > 0]
        self.LRT = np.sum(self.cell_store[self.cell_store > 0] * np.log(self.cell_store[self.cell_store > 0])) - np.sum(
            cell_c * np.log(cell_c)) - np.sum(cell_r * np.log(cell_r)) + n * np.log(n)
        self.LRT_pvalue = 1 - stats.chi2.cdf(2 * self.LRT, (2 ** depth - 1) ** 2)

        if plot:
            ax = bet_plot(depth=depth, beidx1=self.max_index[0], beidx2=self.max_index[1])
            ax.plot(x, y, 'ro', markersize=1.5)
            ax.axes.set_xlim(0, 1)
            ax.axes.set_ylim(0, 1)
            plt.show()

        def prn_obj(self):
            print('\n'.join(['%s: \n %s' % item for item in self.__dict__.items()]))

        if print_res:
            prn_obj(self)


# In[6]:


class BETd:
    def __init__(self, u: List[float], v: List[float], d1=2, d2=2, plot=False, print_res=True):
        n = len(u)
        BEx = [fraction_to_binary(a, d=d1) for a in u]
        BEy = [fraction_to_binary(a, d=d2) for a in v]

        RDx = 2 * np.array(BEx) - 1
        RDy = 2 * np.array(BEy) - 1

        index1, index_name1, index2, index_name2 = [], [], [], []
        count_interaction = np.array([[0] * (2 ** (d1 - 1)) * (2 ** (d2 - 1))]).reshape(2 ** (d1 - 1), 2 ** (d2 - 1))
        for i in range(d1):
            temp1 = list(itertools.combinations([x for x in range(1, d1)], i))
            index1 += temp1
            index_name1 += [":".join(list(str(n) for n in idx) + [str(d1)]) for idx in temp1]

        for j in range(d2):
            temp2 = list(itertools.combinations([x for x in range(1, d2)], j))
            index2 += temp2
            index_name2 += [":".join(list(str(n) for n in idx) + [str(d2)]) for idx in temp2]

        for i in range(len(index1)):
            for j in range(len(index2)):
                count_interaction[i][j] = np.sum(np.prod(np.c_[RDx[:, [(x - 1) for x in list(index1[i]) + [d1]]], RDy[:,
                                                                                                                  [(
                                                                                                                               x - 1)
                                                                                                                   for x
                                                                                                                   in
                                                                                                                   list(
                                                                                                                       index2[
                                                                                                                           j]) + [
                                                                                                                       d2]]]],
                                                         axis=1))

        self.count_interaction_df = pd.DataFrame(count_interaction)  # return
        self.count_interaction_df.columns, self.count_interaction_df.index = index_name2, index_name1

        max_abs_count_interaction = np.max(np.abs(count_interaction))
        max_ind = np.where(abs(count_interaction) == max_abs_count_interaction)
        self.bet_max_abs = max_abs_count_interaction
        self.max_index = [self.count_interaction_df.index[max_ind[0][0]],
                          self.count_interaction_df.columns[max_ind[1][0]]]  # return
        self.strongest_asymmetry = count_interaction[max_ind[0][0], max_ind[1][0]]  # return

        self.p_value = min(
            (2 ** (d1 + d2 - 2)) * 2 * (1 - stats.binom.cdf((max_abs_count_interaction + n) / 2 - 1, n, 0.5)),
            1)  # return

        table22 = [[max_abs_count_interaction / 4 + n / 4, -max_abs_count_interaction / 4 + n / 4],
                   [-max_abs_count_interaction / 4 + n / 4, max_abs_count_interaction / 4 + n / 4]]
        odd_ratio, fpvalue = stats.fisher_exact(table22)
        self.FE_p_value = min(2 ** (d1 + d2 - 2) * (fpvalue - stats.hypergeom.pmf(table22[0][0], n, n / 2, n / 2) / 2),
                              1)  # return

        if plot:
            ax = bet_plot(depth=max(d1, d2), beidx1=self.max_index[0], beidx2=self.max_index[1])
            ax.plot(u, v, 'ro', markersize=1.5)
            ax.axes.set_xlim(0, 1)
            ax.axes.set_ylim(0, 1)
            plt.show()

        def prn_obj(self):
            print('\n'.join(['%s: \n %s' % item for item in self.__dict__.items()]))

        if print_res:
            prn_obj(self)


# In[7]:


class BETs:
    def __init__(self, u, v, max_depth=4, print_res=True):
        n = len(u)
        temp = BET(u, v, depth=1, print_res=False)

        bet_adj_pvalue = [temp.p_value]
        bet_max_idx = [temp.max_index]
        bets_max = [temp.bet_max]
        temp_col = list(temp.count_interaction_df.columns)
        temp_row = list(temp.count_interaction_df.index)

        if max_depth == 1:

            self.bets_pvalue = bet_adj_pvalue[0]
            self.bets_index = bet_max_idx[0]
            self.bets_max = bets_max[0]
        else:
            for d in range(2, max_depth + 1):
                tempa = BET(u, v, depth=d, print_res=False)
                #                 tempa_col = list(tempa.count_interaction_df.columns)
                #                 tempa_row = list(tempa.count_interaction_df.index)
                tempa_count_interaction = tempa.count_interaction_df

                tempa_count_interaction.loc[temp_col, temp_row] = 0
                count_interaction = np.array(tempa_count_interaction)

                max_abs_count_interaction = np.max(np.abs(count_interaction))
                max_count_interaction = np.max(count_interaction)
                max_ind = np.where(abs(count_interaction) == max_abs_count_interaction)
                max_index = [tempa_count_interaction.index[max_ind[0][0]],
                             tempa_count_interaction.columns[max_ind[1][0]]]
                strongest_asymmetry = count_interaction[max_ind[0][0], max_ind[1][0]]  # return

                p_value = min(((2 ** d - 1) ** 2 - (2 ** (d - 1) - 1) ** 2) * 2 * (
                            1 - stats.binom.cdf((max_abs_count_interaction + n) / 2 - 1, n, 0.5)), 1)

                table22 = [[max_abs_count_interaction / 4 + n / 4, -max_abs_count_interaction / 4 + n / 4],
                           [-max_abs_count_interaction / 4 + n / 4, max_abs_count_interaction / 4 + n / 4]]
                odd_ratio, fpvalue = stats.fisher_exact(table22)
                FE_p_value = min(((2 ** d - 1) ** 2 - (2 ** (d - 1) - 1) ** 2) * (
                            fpvalue - stats.hypergeom.pmf(table22[0][0], n, n / 2, n / 2) / 2), 1)

                bet_adj_pvalue.append(FE_p_value)
                bet_max_idx.append(max_index)
                bets_max.append(max_count_interaction)
                temp_col = list(tempa.count_interaction_df.columns)
                temp_row = list(tempa.count_interaction_df.index)

        self.bets_max = count_interaction[max_ind[0][0], max_ind[1][0]]
        self.bets_pvalue = min(min(bet_adj_pvalue) * max_depth, 1)
        self.bets_index = bet_max_idx[bet_adj_pvalue.index(min(bet_adj_pvalue))]

        def prn_obj(self):
            print('\n'.join(['%s: \n %s' % item for item in self.__dict__.items()]))

        if print_res:
            prn_obj(self)

# In[ ]:




