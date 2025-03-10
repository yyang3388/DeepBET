import argparse
import torch
from statsmodels.distributions.empirical_distribution import ECDF
import BET
from data_generator import generate_samples_random
from DNN_model import do_train
from joblib import Parallel, delayed
from multiple_split import sim_worker, sim_worker_h1

parser = argparse.ArgumentParser(description='DeepBet')
parser.add_argument('-t', '--test', type=str, default='type1error', choices=['type1error', 'power'])
parser.add_argument('-s', '--split', type=str, default='single', choices=['single', 'multiple'])
parser.add_argument('-ss', '--sim_size', type=int, default= 500)
parser.add_argument('-n', '--n_sample', type=int, default=1000)
parser.add_argument('-st', '--sType', type=str, default='CI', choices=['CI','I','dependent'])
parser.add_argument('-dx', '--x_dims', type=int, default=1)
parser.add_argument('-dy', '--y_dims', type=int, default=1)
parser.add_argument('-dz', '--z_dims', type=int, default=100)
parser.add_argument('-ns', '--nstd', type=float, default=0.05)
parser.add_argument('-zd', '--z_dist', type=str, default='gaussian', choices=['gaussian', 'laplace'])
parser.add_argument('-ax', '--alpha_x', type=float, default=0.9)
parser.add_argument('-nm', '--normalize', type=bool, default=True)
args = parser.parse_args()

def main():
    test = args.test
    split = args.split
    sim_size = args.sim_size
    n_sample = args.n_sample
    sType = args.sType
    dx = args.x_dims
    dy = args.y_dims
    dz = args.z_dims
    dist_z = args.z_dist
    nstd = args.nstd
    alpha_x = args.alpha_x
    norm = args.normalize

    if test == 'type1error':
        if split == 'single':
            s = 0
            for i in range(sim_size):
                X_h0, Y_h0, Z_h0 = generate_samples_random(size=n_sample, sType=sType, dx=dx, dy=dy, dz=dz, nstd=nstd,
                                                           alpha_x=alpha_x,
                                                           normalize=norm, seed=None, dist_z=dist_z)
                Y_h0 = torch.tensor(Y_h0, dtype=torch.float32).squeeze()
                X_h0 = torch.tensor(X_h0, dtype=torch.float32).squeeze()
                Z_h0 = torch.tensor(Z_h0, dtype=torch.float32).squeeze()
                Y_pred_h0, X_pred_h0, Y_test_h0, X_test_h0 = do_train(Y_h0, X_h0, Z_h0)
                epsilon_Yh0 = Y_test_h0 - Y_pred_h0
                epsilon_Xh0 = X_test_h0 - X_pred_h0
                U = ECDF(epsilon_Xh0)
                V = ECDF(epsilon_Yh0)
                p = BET.BETs(U(epsilon_Xh0), V(epsilon_Yh0), print_res=False).bets_pvalue
                s += (p < 0.05)
            type1_error = s / 500
            print(type1_error)
        elif split == 'multiple':
            sim_res = Parallel(n_jobs=6)(delayed(sim_worker)(n_sample, sType, dx, dy, dz, nstd, alpha_x, norm, dist_z) for _ in range(500))
            type1error = sum(sim_res) / len(sim_res)
            print(type1error)
        else:
            raise ValueError('split type is not valid.')
    elif test == 'power':
        if split == 'single':
            s = 0
            p_values = []
            for i in range(500):
                X_h1, Y_h1, Z_h1 = generate_samples_random(size=n_sample, sType=sType, dx=dx, dy=dy, dz=dz, nstd=nstd,
                                                           alpha_x=alpha_x,
                                                           normalize=norm, seed=None, dist_z=dist_z)
                Y_h1 = torch.tensor(Y_h1, dtype=torch.float32).squeeze()
                X_h1 = torch.tensor(X_h1, dtype=torch.float32).squeeze()
                Z_h1 = torch.tensor(Z_h1, dtype=torch.float32).squeeze()
                Yh1_pred, Xh1_pred, Yh1_test, Xh1_test = do_train(Y_h1, X_h1, Z_h1)
                epsilon_Yh1 = Yh1_test - Yh1_pred
                epsilon_Xh1 = Xh1_test - Xh1_pred
                U = ECDF(epsilon_Xh1)
                V = ECDF(epsilon_Yh1)
                p = BET.BET(U(epsilon_Xh1), V(epsilon_Yh1), print_res=False).p_value
                p_values.append(p)
                s += (p < 0.05)
            power = s / 500
            print(power)
        elif split == 'multiple':
            sim_res_h1 = Parallel(n_jobs=6)(delayed(sim_worker_h1)(n_sample, sType, dx, dy, dz, nstd, alpha_x, norm, dist_z) for _ in range(500))
            power = sum(sim_res_h1) / len(sim_res_h1)
            print(power)
        else:
            raise ValueError('split type is not valid.')
    else:
        raise ValueError('Test is not supported.')

if __name__ == '__main__':
    main()
