import torch
import numpy as np
from itertools import cycle

cycol = cycle('bgrcmk')

def within(x, y, xlim, ylim):
    val_inds = (x >= 0) & (y >= 0)
    val_inds = val_inds & (x < xlim) & (y < ylim)
    return val_inds

def identity_Rt():
    return np.hstack((np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)))

def _skew_symmetric(v): # v: [3, 1] or [batch_size, 3, 1]
    if len(v.size())==2:
        zero = torch.zeros_like(v[0, 0])
        M = torch.stack([
            zero, -v[2, 0], v[1, 0],
            v[2, 0], zero, -v[0, 0],
            -v[1, 0], v[0, 0], zero,
        ], dim=0)
        return M.view(3, 3)
    else:
        zero = torch.zeros_like(v[:, 0, 0])
        M = torch.stack([
            zero, -v[:, 2, 0], v[:, 1, 0],
            v[:, 2, 0], zero, -v[:, 0, 0],
            -v[:, 1, 0], v[:, 0, 0], zero,
        ], dim=1)
        return M.view(-1, 3, 3)
    
def _homo(x):
    # input: x [N, 2] or [batch_size, N, 2]
    # output: x_homo [N, 3]  or [batch_size, N, 3]
    assert len(x.size()) in [2, 3]
    if len(x.size())==2:
        ones = torch.ones(x.size()[0], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 1)
    elif len(x.size())==3:
        ones = torch.ones(x.size()[0], x.size()[1], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 2)
    return x_homo

def _de_homo(x_homo):
    # input: x_homo [N, 3] or [batch_size, N, 3]
    # output: x [N, 2] or [batch_size, N, 2]
    assert len(x_homo.size()) in [2, 3]
    epi = 1e-10
    if len(x_homo.size())==2:
        x = x_homo[:, :2]/(x_homo[:, 2:3]+epi)
    else:
        x = x_homo[:, :, :2]/(x_homo[:, :, 2:3]+epi)
    return x

def homo_np(x):
    # input: x [N, D]
    # output: x_homo [N, D+1]
    N = x.shape[0]
    x_homo = np.hstack((x, np.ones((N, 1))))
    return x_homo

def de_homo_np(x_homo):
    # input: x_homo [N, D]
    # output: x [N, D-1]
    N = x_homo.shape[0]
    epi = 1e-10
    x = np.hstack((x_homo[:, 0:1]/(x_homo[:, 2:3]+epi), x_homo[:, 1:2]/(x_homo[:, 2:3]+epi)))
    return x

def Rt_pad(Rt):
    # Padding 3*4 [R|t] to 4*4 [[R|t], [0, 1]]
    assert Rt.shape==(3, 4)
    return np.vstack((Rt, np.array([[0., 0., 0., 1.]], dtype=Rt.dtype)))

def Rt_depad(Rt01):
    # dePadding 4*4 [[R|t], [0, 1]] to 3*4 [R|t]
    assert Rt01.shape==(4, 4)
    return Rt01[:3, :]

def vis_masks_to_inds(mask1, mask2):
    val_inds_both = mask1 & mask2
    val_idxes = [idx for idx in range(val_inds_both.shape[0]) if val_inds_both[idx]] # within indexes
    return val_idxes
