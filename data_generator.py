import numpy as np

def generate_samples_random(size=1000, sType='CI', dx=1, dy=1, dz=100, nstd=0.05, alpha_x=0.05,
                            normalize=True, seed=None, dist_z='gaussian'):
    '''
    Generate CI,I or NI post-nonlinear samples
    1. Z is independent Gaussian or Laplace
    2. X = f1(<a,Z> + b + noise) and Y = f2(<c,Z> + d + noise) in case of CI
    Arguments:
        size : number of samples
        sType: CI, I, or NI
        dx: Dimension of X
        dy: Dimension of Y
        dz: Dimension of Z
        nstd: noise standard deviation
        we set f1 to be sin function and f2 to be cos function.
    Output:
        Samples X, Y, Z
    '''
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

    num = size

    if dist_z == 'gaussian':
        cov = np.eye(dz)
        mu = np.zeros(dz)
        Z = np.random.multivariate_normal(mu, cov, num)

    elif dist_z == 'laplace':
        Z = np.random.laplace(loc=0.0, scale=1.0, size=num*dz)
        Z = np.reshape(Z, (num, dz))

    Ax = np.random.rand(dz, dx)
    for i in range(dx):
        Ax[:, i] = Ax[:, i] / np.linalg.norm(Ax[:, i], ord=1)
    Ay = np.random.rand(dz, dy)
    for i in range(dy):
        Ay[:, i] = Ay[:, i] / np.linalg.norm(Ay[:, i], ord=1)

    Axy = np.ones((dx, dy)) * alpha_x

    if sType == 'CI':
        X = np.sin(np.matmul(Z, Ax)) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num)
        Y = np.cos(np.matmul(Z, Ay)) + nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num)
    elif sType == 'I':
        X = nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num)
        Y = nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num)
    else:
        X = np.sin(np.matmul(Z, Ax)) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dy), num)
        #e = nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num)
        Y = np.cos(np.matmul(X, Axy) + np.matmul(Z, Ay)) + nstd * np.random.multivariate_normal(np.zeros(dy),
                                                                                               np.eye(dy), num)

    if normalize:
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())

    return np.array(X), np.array(Y), np.array(Z)