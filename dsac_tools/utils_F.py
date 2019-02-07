import torch
import numpy as np

def homo_py(x):
    # input: x [N, 2]
    # output: x_homo [N, 3]
    N = list(x.size())[0]
    x_homo = torch.cat((x, torch.ones(N, 1, dtype=x.dtype)), 1)
    return x_homo

def de_homo_py(x_homo):
    # input: x_homo [N, 3]
    # output: x [N, 2]
    N = list(x_homo.size())[0]
    epi = 1e-10
    x = torch.cat((x_homo[:, 0:1]/(x_homo[:, 2:3]+epi), x_homo[:, 1:2]/(x_homo[:, 2:3]+epi)), 1)
    return x

def E_from_XY(X, Y):
    # X, Y: [N, 2]
    xx = torch.cat([X.t(), Y.t()], dim=0)
    # print(xx.size())
    X = torch.stack([
        xx[2, :] * xx[0, :], xx[2, :] * xx[1, :], xx[2, :],
        xx[3, :] * xx[0, :], xx[3, :] * xx[1, :], xx[3, :],
        xx[0, :], xx[1, :], torch.ones_like(xx[0, :])
    ], dim=0).t()
    XwX = torch.matmul(X.t(), X)
    # print("XwX shape = {}".format(XwX.shape))

    # Recover essential matrix from self-adjoing eigen
    e, v = torch.eig(XwX, eigenvectors=True)
    # print(t)
    # print('----E_gt', E.numpy())
    E_recover = v[:, 8].reshape((3, 3))
    # E_recover_rescale = E_recover / torch.norm(E_recover) * torch.norm(E)
    # print('-E_recover', E_recover_rescale.numpy())
    U, D, V = torch.svd(E_recover)
    diag_sing = torch.diag(torch.tensor([1., 1., 0.], dtype=torch.float64))
    E_recover_hat = torch.mm(U, torch.mm(diag_sing, V.t()))
    # E_recover_hat_rescale = E_recover_hat / torch.norm(E_recover_hat) * torch.norm(E)
    # print('--E_recover_hat', E_recover_hat_rescale.numpy())

    return E_recover_hat

def F_from_XY(X, Y):
    # X, Y: [N, 2]
    xx = torch.cat([X.t(), Y.t()], dim=0)
    # print(xx.size())
    X = torch.stack([
        xx[2, :] * xx[0, :], xx[2, :] * xx[1, :], xx[2, :],
        xx[3, :] * xx[0, :], xx[3, :] * xx[1, :], xx[3, :],
        xx[0, :], xx[1, :], torch.ones_like(xx[0, :])
    ], dim=0).t()
    U, D, V = torch.svd(X)
    F_recover = torch.reshape(V[:, -1], (3, 3))
    # F_recover_rescale = F_recover / torch.norm(F_recover) * torch.norm(F)
    # print('-', F_recover_rescale.numpy())
    FU, FD, FV= torch.svd(F_recover);
    FDnew = torch.diag(FD);
    FDnew[2, 2] = 0;
    F_recover_sing = torch.mm(FU, torch.mm(FDnew, FV.t()))
    # F_recover_sing_rescale = F_recover_sing / torch.norm(F_recover_sing) * torch.norm(F)
    return F_recover_sing

def YFX(F, X, Y, if_homo=False):
    if not if_homo:
        X = homo_py(X)
        Y = homo_py(Y)
    should_zeros = torch.diag(torch.matmul(torch.matmul(Y, F), X.t()))
    return should_zeros

def sampson_dist(F, X, Y, if_homo=False):
    if not if_homo:
        X = homo_py(X)
        Y = homo_py(Y)
    nominator = (torch.diag(torch.matmul(torch.matmul(Y, F), X.t())))**2
    Fx1 = torch.mm(F, X.t())
    Fx2 = torch.mm(F, Y.t())
    denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
    errors = nominator/denom
    return errors

def F_to_E(F, K):
    return torch.matmul(torch.matmul(K.t(), F), K)

def E_to_F(E, K):
    return torch.matmul(torch.matmul(torch.inverse(K).t(), E), torch.inverse(K))

def get_M2s(E):
    # Getting 4 possible poses from E
    U, S, V = torch.svd(E)
    W = torch.tensor([[0,-1,0], [1,0,0], [0,0,1]], dtype=torch.float64)
    if torch.det(torch.mm(U, torch.mm(W, V.t())))<0:
        W = -W
    # print('-- delta_t_gt', delta_t_gt)

    t_recover = U[:, 2:3]/torch.norm(U[:, 2:3])
    # t_recover_rescale = U[:, 2]/torch.norm(U[:, 2])*np.linalg.norm(t_gt) # -t_recover_rescale is also an option
    R_recover_1 = torch.mm(U, torch.mm(W, V.t()))
    R_recover_2 = torch.mm(U, torch.mm(W.t(), V.t())) # also an option
    # print('-- t_recover', t_recover.numpy())
    # print('-- R_recover_1', R_recover_1.numpy(), torch.det(R_recover_1).numpy())
    # print('-- R_recover_2', R_recover_2.numpy(), torch.det(R_recover_2).numpy())

    R2s = [R_recover_1, R_recover_2]
    t2s = [t_recover, -t_recover]
    M2s = [torch.cat((x, y), 1) for x, y in [(x,y) for x in R2s for y in t2s]]
    return R2s, t2s, M2s



# ------ For homography ------

def H_from_XY(X, Y):
    N = list(X.size())[0]
    A = torch.zeros(2*N, 9, dtype=torch.float32)
    A[0::2, 0:2] = X
    A[0::2, 2:3] = torch.ones(N, 1)
    A[1::2, 3:5] = X
    A[1::2, 5:6] = torch.ones(N, 1)
    A[0::2, 6:8] = X
    A[1::2, 6:8] = X
    A[:, 8:9] = torch.ones(2*N, 1)
    Y_vec = torch.reshape(Y, (2*N, 1))
    A[:, 6:7] = -A[:, 6:7] * Y_vec
    A[:, 7:8] = -A[:, 7:8] * Y_vec
    A[:, 8:9] = -A[:, 8:9] * Y_vec
    U, S, V = torch.svd(A)
    H = torch.reshape(V[:, -1], (3, 3))
    H = H / H[2, 2]
    return H

def H_from_XY_np(X, Y):
    N = X.shape[0]
    A = np.zeros((2*N, 9))
    A[0::2, 0:2] = X
    A[0::2, 2:3] = np.ones((N, 1))
    A[1::2, 3:5] = X
    A[1::2, 5:6] = np.ones((N, 1))
    A[0::2, 6:8] = X
    A[1::2, 6:8] = X
    A[:, 8:9] = np.ones((2*N, 1))
    y_vec = np.reshape(Y, (2*N, 1))
    A[:, 6:7] = -A[:, 6:7] * y_vec
    A[:, 7:8] = -A[:, 7:8] * y_vec
    A[:, 8:9] = -A[:, 8:9] * y_vec
    U, S, V = np.linalg.svd(A)
    H = np.reshape(V[-1, :], (3, 3))
    H = H / H[2, 2]
    return H

def reproj_error_HXY(H, X, Y):
    HX = de_homo_py(torch.matmul(H, homo_py(X).t()).t())
    errors = torch.norm(Y - HX, dim=1)
    return torch.mean(errors), errors

import operator as op
from functools import reduce
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return int(numer / denom)
