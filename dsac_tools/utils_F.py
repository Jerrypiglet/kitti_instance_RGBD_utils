import torch
import numpy as np
import cv2
import dsac_tools.utils_misc as utils_misc
import dsac_tools.utils_geo as utils_geo
import dsac_tools.utils_vis as utils_vis

# def E_from_XY(X, Y):
#     # X, Y: [N, 2]
#     xx = torch.cat([X.t(), Y.t()], dim=0)
#     # print(xx.size())
#     X = torch.stack([
#         xx[2, :] * xx[0, :], xx[2, :] * xx[1, :], xx[2, :],
#         xx[3, :] * xx[0, :], xx[3, :] * xx[1, :], xx[3, :],
#         xx[0, :], xx[1, :], torch.ones_like(xx[0, :])
#     ], dim=0).t()
#     XwX = torch.matmul(X.t(), X)
#     # print("XwX shape = {}".format(XwX.shape))

#     # Recover essential matrix from self-adjoing eigen
#     e, v = torch.eig(XwX, eigenvectors=True)
#     # print(t)
#     # print('----E_gt', E.numpy())
#     E_recover = v[:, 8].reshape((3, 3))
#     print(E_recover.numpy())
#     # E_recover_rescale = E_recover / torch.norm(E_recover) * torch.norm(E)
#     # print('-E_recover', E_recover_rescale.numpy())
#     U, D, V = torch.svd(E_recover)
#     diag_sing = torch.diag(torch.tensor([1., 1., 0.], dtype=torch.float64))
#     E_recover_hat = torch.mm(U, torch.mm(diag_sing, V.t()))
#     # E_recover_hat_rescale = E_recover_hat / torch.norm(E_recover_hat) * torch.norm(E)
#     # print('--E_recover_hat', E_recover_hat_rescale.numpy())

#     return E_recover_hat

def _E_from_XY(X, Y, K):
    F = _F_from_XY(X, Y)
    E = _F_to_E(F, K)
    return E

def _F_from_XY(X, Y, W=None):
    # X, Y: [N, 2]
    xx = torch.cat([X.t(), Y.t()], dim=0)
    # print(xx.size())
    X = torch.stack([
        xx[2, :] * xx[0, :], xx[2, :] * xx[1, :], xx[2, :],
        xx[3, :] * xx[0, :], xx[3, :] * xx[1, :], xx[3, :],
        xx[0, :], xx[1, :], torch.ones_like(xx[0, :])
    ], dim=0).t()
    if W is not None:
        X = torch.mm(W, X)
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

def _YFX(F, X, Y, if_homo=False):
    if not if_homo:
        X = homo_py(X)
        Y = homo_py(Y)
    should_zeros = torch.diag(torch.matmul(torch.matmul(Y, F), X.t()))
    return should_zeros

def _sampson_dist(F, X, Y, if_homo=False):
    if not if_homo:
        X = utils_misc._homo(X)
        Y = utils_misc._homo(Y)
    nominator = (torch.diag(torch.matmul(torch.matmul(Y, F), X.t())))**2
    Fx1 = torch.mm(F, X.t())
    Fx2 = torch.mm(F, Y.t())
    denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
    errors = nominator/denom
    return errors

def _F_to_E(F, K):
    return torch.matmul(torch.matmul(K.t(), F), K)

def E_to_F(E, K):
    return torch.matmul(torch.matmul(torch.inverse(K).t(), E), torch.inverse(K))

def _get_M2s(E):
    # Getting 4 possible poses from E
    U, S, V = torch.svd(E)
    W = torch.tensor([[0,-1,0], [1,0,0], [0,0,1]], dtype=torch.float64)
    if torch.det(torch.mm(U, torch.mm(W, V.t())))<0:
        W = -W
    # print('-- delta_t_gt', delta_t_gt)

    t_recover = U[:, 2:3]/torch.norm(U[:, 2:3])
    # print('---', E.numpy())
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

def _E_to_M(E_est_th, K, x1, x2, inlier_mask, delta_Rt_gt=None):
    R2s, t2s, M2s = _get_M2s(E_est_th)

    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    M1 = np.hstack((R1, t1))

    cheirality_checks = []
    M2_list = []
    for Rt_idx, M2 in enumerate(M2s):
        M2 = M2.numpy()
        R2 = M2[:, :3]
        t2 = M2[:, 3:4]
        # print(M2, np.linalg.det(R2))

        X_tri_homo = cv2.triangulatePoints(np.matmul(K, M1), np.matmul(K, M2), x1[inlier_mask].T, x2[inlier_mask].T)
        X_tri = X_tri_homo[:3, :]/X_tri_homo[-1, :]
        C1 = -np.matmul(R1, t1) # https://math.stackexchange.com/questions/82602/how-to-find-camera-position-and-rotation-from-a-4x4-matrix
        cheirality1 = np.matmul(R1[2:3, :], (X_tri-C1)) # https://cmsc426.github.io/sfm/

        X_tri_cam3 = np.matmul(R2, X_tri) + t2
        C2 = -np.matmul(R2, t2)
        cheirality2 = np.matmul(R2[2:3, :], (X_tri_cam3-C2))

        # Get rid of small angle points. @Manmo: you should discard points that are beyond a depth threshold (say, more than 100m), or which subtend a small angle between the two cameras (say, less than 5 degrees).
        v1s = (X_tri-C1).T
        v2s = (X_tri-C2).T
        angles_C1C2 = utils_geo.vectors_angle(v1s, v2s).T
        angles_mask = angles_C1C2 > 2.
        if not np.any(angles_mask):
            cheirality_check = False
            # print(angles_C1C2)
        else:
            cheirality_check = np.min(cheirality1[angles_mask])>0 and np.min(cheirality2[angles_mask])>0
        cheirality_checks.append(cheirality_check)
        if cheirality_check:
            print('-- Good M:', M2)
            M2_list.append(M2)
        # else:
        #     print(X_tri[-1, angles_mask.reshape([-1])])
        #     print(X_tri_cam3[-1, angles_mask.reshape([-1])])

    if np.sum(cheirality_checks)==1:
        Rt_idx = cheirality_checks.index(True)
        print('The %d_th Rt meets the Cheirality Condition! with [R|t]:'%Rt_idx)
        M_inv = utils_misc.Rt_depad(np.linalg.inv(utils_misc.Rt_pad(M2s[Rt_idx].numpy())))
    
        if delta_Rt_gt is not None:
            R2 = M2s[Rt_idx][:, :3].numpy()
            t2 = M2s[Rt_idx][:, 3:4].numpy()
            # error_R = min([utils_geo.rot12_to_angle_error(R2.numpy(), delta_R_gt) for R2 in R2s])
            # error_t = min(utils_geo.vector_angle(t2, delta_t_gt), utils_geo.vector_angle(-t2, delta_t_gt))

            R2 = M_inv[:, :3]
            t2 = M_inv[:, 3:4]
            error_R = utils_geo.rot12_to_angle_error(R2, delta_Rt_gt[:, :3])
            error_t = utils_geo.vector_angle(t2, delta_Rt_gt[:, 3:4])
            print('Recovered by ours: The rotation error (degree) %.4f, and translation error (degree) %.4f'%(error_R, error_t))

        print(M_inv)
    else:
        print('Error! %d of qualified [R|t] found!'%np.sum(cheirality_checks))

    return M2_list



# ------ For homography ------

def _H_from_XY(X, Y):
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

def _reproj_error_HXY(H, X, Y):
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

def _E_F_from_Rt(R, t, K):
    """ Better use F instead of E """
    K_th = torch.from_numpy(K).to(torch.float64)
    R_th = torch.from_numpy(R).to(torch.float64)
    t_th = torch.from_numpy(t).to(torch.float64)
    t_gt_x = utils_misc._skew_symmetric(t_th)
#     print(t_gt_x, R_th)
    E_gt_th = torch.matmul(t_gt_x, R_th)
    F_gt_th = torch.matmul(torch.matmul(torch.inverse(K_th).t(), E_gt_th), torch.inverse(K_th))
    return E_gt_th, F_gt_th

def vali_with_best10(F_gt_th, E_gt_th, x1, x2, img1_rgb_np, img2_rgb_np, kitti_two_frame_loader, DSAC_params):
    """ Validate pose estimation with best 10 corres."""
    # Validation: use best 10 corres with smalles Sampson distance to GT F to compute E and F
    print('--------------- Check with best 20 corres. ---------------')
    errors = _sampson_dist(F_gt_th, torch.from_numpy(x1).to(torch.float64), torch.from_numpy(x2).to(torch.float64), False)
    sort_index = np.argsort(errors.numpy())
    mask_index = sort_index[:20]
    print('--- Best 20 errors', errors[mask_index].numpy())

    # utils_vis.draw_corr_widths(img1_rgb_np, img2_rgb_np, x1[mask_index, :], x2[mask_index, :], np.zeros(x2[mask_index, :].shape[0])+2, '[Best 20] Sampson distance w.r.t. ground truth F (the thicker the worse corres.)', False)
    E_est_th = _E_from_XY(torch.from_numpy(x1[mask_index, :]), torch.from_numpy(x2[mask_index, :]), kitti_two_frame_loader.K_th)
    print('+++ E est&GT', (E_est_th / torch.norm(E_est_th) * torch.norm(E_gt_th)).numpy())
    print(E_gt_th.numpy())

    R2s_list, t2s_list, M2_list = _get_M2s(E_est_th)
    print('=== M', M2_list[0].numpy())

    F_est_th = _F_from_XY(torch.from_numpy(x1[mask_index, :]), torch.from_numpy(x2[mask_index, :]))
    print('--- F est&GT', (F_est_th / torch.norm(F_est_th) * torch.norm(F_gt_th)).numpy())
    print(F_gt_th.numpy())

    ## Check number of inliers w.r.t F_gt and thres
    e = np.sort(errors.numpy().tolist())
    print('--- %d/%d inliers.'%(sum(e<DSAC_params['inlier_thresh']), len(e)))

    return mask_index[:8]
