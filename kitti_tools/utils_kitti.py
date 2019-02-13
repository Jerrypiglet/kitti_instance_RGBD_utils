import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch

import pykitti  # install using pip install pykitti
from kitti_tools.kitti_raw_loader import *
import dsac_tools.utils_misc as utils_misc
import dsac_tools.utils_vis as utils_vis

class KittiLoader(object):
    def __init__(self, KITTI_ROOT_PATH):
        self.KITTI_ROOT_PATH = KITTI_ROOT_PATH
        self.KITTI_PATH = KITTI_ROOT_PATH + '/raw'


    def set_drive(self, date, seq):
        self.date_name = date
        self.seq_name = seq

        # tracklet_path = KITTI_PATH+'/%s/%s/tracklet_labels.xml'%(date_name, date_name+seq_name)
        self.fdir_path = self.KITTI_PATH+'/%s/%s/'%(self.date_name, self.date_name+self.seq_name)
        # if os.path.exists(tracklet_path):
        #     print('======Tracklet Exists:', tracklet_path)
        # else:
        #     print('======Tracklet NOT Exists:', tracklet_path)
            
        ## Raw Data directory information
        path = self.fdir_path.rstrip('/')
        basedir = path.rsplit('/',2)[0]
        date = path.split('/')[-2]
        drive = path.split('/')[-1].split('_')[-2]

        self.dataset = pykitti.raw(basedir, date, drive)
        # tracklet_rects, tracklet_types, tracklet_ids, tracklet_Rs, tracklet_ts = load_tracklets_for_frames(len(list(dataset.velo)),\
        #                '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir,date, date, drive)) # (3, 3) (3,)
        self.dataset_gray = list(self.dataset.gray)
        self.dataset_rgb = list(self.dataset.rgb) 

        ## From Rui
        # Understanding calibs: https://github.com/utiasSTARS/pykitti/blob/0e5fd7fefa7cd10bbdfb5bd131bb58481d481116/pykitti/raw.py#L150
        # cam = 'leftRGB'
        self.P_rects = {'leftRGB': self.dataset.calib.P_rect_20, 'rightRGB': self.dataset.calib.P_rect_30} # cameras def.: https://github.com/utiasSTARS/pykitti/blob/19d29b665ac4787a10306bbbbf8831181b38eb38/pykitti/odometry.py#L42
        # cam2cam = {}
        self.R_cam2rect = self.dataset.calib.R_rect_00 # [cam0] R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
        P_rect = self.P_rects['leftRGB'] # P_rect_0[0-3]: 3x4 projection matrix after rectification; the reprojection matrix in MV3D
        self.velo2cam = self.dataset.calib.T_cam0_velo_unrect
        self.P_velo2im = np.dot(np.dot(P_rect, self.R_cam2rect), self.velo2cam) # 4*3
        self.im_shape = [self.dataset_gray[0][0].size[1], self.dataset_gray[0][0].size[0]]
        self.N_frames = len(list(self.dataset.velo))

        print('KITTI track loaded at %s.'%self.fdir_path)

    def load_cam_poses(self):
        oxts_path = self.fdir_path + 'oxts/data/*.txt'
        oxts = sorted(glob.glob(oxts_path))

        scene_data = {'cid': '', 'dir': self.fdir_path, 'speed': [], 'frame_id': [], 'pose':[], 'rel_path': ''}
        scale = None
        origin = None
        imu2velo_dict = read_calib_file(self.fdir_path+'../calib_imu_to_velo.txt')
        velo2cam_dict = read_calib_file(self.fdir_path+'../calib_velo_to_cam.txt')
        cam2cam_dict = read_calib_file(self.fdir_path+'../calib_cam_to_cam.txt')

        velo2cam_mat = transform_from_rot_trans(velo2cam_dict['R'], velo2cam_dict['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo_dict['R'], imu2velo_dict['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam_dict['R_rect_00'], np.zeros(3))

        imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat
        for n, f in enumerate(oxts):
            metadata = np.genfromtxt(f)
            speed = metadata[8:11]
            scene_data['speed'].append(speed)
            scene_data['frame_id'].append('{:010d}'.format(n))
            lat = metadata[0]

            if scale is None:
                scale = np.cos(lat * np.pi / 180.)

            pose_matrix = pose_from_oxts_packet(metadata[:6], scale)
            if origin is None:
                origin = pose_matrix

            odo_pose = imu2cam @ np.linalg.inv(origin) @ pose_matrix @ np.linalg.inv(imu2cam)
            odo_pose_Rt = odo_pose[:3]
            R21 = odo_pose_Rt[:, :3]
            t21 = odo_pose_Rt[:, 3:4]
            R12 = R21.T
            t12 = -np.matmul(R12, t21)
            Rt12 = np.hstack((R12, t12))
            scene_data['pose'].append(Rt12)

        self.scene_data = scene_data
        print('Scene pose loaded. First two poses:')
        print(scene_data['pose'][:2])

    def show_demo(self):
        velo_reproj_list = []
        for i in range(self.N_frames):
            velo = list(self.dataset.velo)[i] # [N, 4]
            # project the points to the camera
            velo = velo[:, :3]
            velo_reproj = utils_misc.homo_np(velo)
            velo_reproj_list.append(velo_reproj)

            for cam_iter, cam in enumerate(['leftRGB', 'rightRGB']):
                P_rect = self.P_rects[cam] # P_rect_0[0-3]: 3x4 projection matrix after rectification; the reprojection matrix in MV3D
                P_velo2im = np.dot(np.dot(P_rect, self.R_cam2rect), self.velo2cam) # 4*3

                velo_pts_im = np.dot(P_velo2im, velo_reproj.T).T # [*, 3]
                velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,2][..., np.newaxis]

                # check if in bounds
                # use minus 1 to get the exact same value as KITTI matlab code
                velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
                velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
                val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
                val_inds = val_inds & (velo_pts_im[:,0] < self.im_shape[1]) & (velo_pts_im[:,1] < self.im_shape[0])
                velo_pts_im = velo_pts_im[val_inds, :]
                
                if i == 0:
                    print('Demo: Showing left/right data (unfiltered and unrectified) of the first frame.')
                    plt.figure(figsize=(30, 8))
                    plt.imshow(self.dataset_rgb[i][cam_iter])
                    plt.scatter(velo_pts_im[:, 0].astype(np.int), velo_pts_im[:, 1].astype(np.int), s=2, c=1./velo_pts_im[:, 2])
                    plt.xlim(0, self.im_shape[1]-1)
                    plt.ylim(self.im_shape[0]-1, 0)
                    plt.title(cam)
                    plt.show()

    def get_left_right_gt(self):
        print(self.dataset.calib.P_rect_20)
        print(self.dataset.calib.P_rect_30)
        self.K = self.dataset.calib.P_rect_20[:3, :3]
        self.K_th = torch.from_numpy(self.K)
        self.Ml_gt = np.matmul(np.linalg.inv(self.K), self.dataset.calib.P_rect_20)
        self.Mr_gt = np.matmul(np.linalg.inv(self.K), self.dataset.calib.P_rect_30)

        tl_gt = self.Ml_gt[:, 3:4]
        tr_gt = self.Mr_gt[:, 3:4]
        Rl_gt = self.Ml_gt[:, :3]
        Rr_gt = self.Mr_gt[:, :3]
        print('GT camera for left/right.')
        print(Rl_gt)
        print(Rr_gt)
        print(tl_gt)
        print(tr_gt)


        self.Rtl_gt = np.vstack((np.hstack((Rl_gt, tl_gt)), np.array([0., 0., 0., 1.], dtype=np.float64)))
        self.delta_Rtlr_gt = np.matmul(np.hstack((Rr_gt, tr_gt)), np.linalg.inv(self.Rtl_gt))
        self.delta_Rlr_gt = self.delta_Rtlr_gt[:, :3]
        self.delta_tlr_gt = self.delta_Rtlr_gt[:, 3:4]
        print(self.delta_Rlr_gt, self.delta_tlr_gt)
        tlr_gt_x = utils_misc._skew_symmetric(torch.from_numpy(self.delta_tlr_gt).float())
        self.Elr_gt_th = torch.matmul(tlr_gt_x, torch.eye(3)).to(torch.float64)
        self.Flr_gt_th = torch.matmul(torch.matmul(torch.inverse(self.K_th).t(), self.Elr_gt_th), torch.inverse(self.K_th))

    def rectify(self, velo_reproj, im_l, im_r, visualize=False):
        val_inds_list = []

        X_homo = np.dot(np.dot(self.R_cam2rect, self.velo2cam), velo_reproj.T) # 4*N
        X_homo_rect = np.matmul(self.Rtl_gt, X_homo)
        X_rect = X_homo_rect[:3, :] / X_homo_rect[3:4, :]

        front_mask = X_rect[-1, :]>0
        X_rect = X_rect[:, front_mask]
        X_homo_rect = X_homo_rect[:, front_mask]

        # Plot with recfitied X and R, t
        x1_homo = np.matmul(self.K, np.matmul(np.hstack((np.eye(3), np.zeros((3, 1)))), X_homo_rect)).T
        x1 = x1_homo[:, 0:2]/x1_homo[:, 2:3]
        if visualize:
            plt.figure(figsize=(30, 8))
            plt.imshow(im_l)
            val_inds = utils_vis.scatter_xy(x1, x1_homo[:, 2], self.im_shape, 'Reprojection to cam 2 with rectified X and camera', new_figure=False)
        else:
            val_inds = utils_misc.within(x1[:, 0], x1[:, 1], self.im_shape[1], self.im_shape[0])
    #     print(val_inds.shape)
        val_inds_list.append(val_inds)

        x2_homo = np.matmul(self.K, np.matmul(np.hstack((self.delta_Rlr_gt, self.delta_tlr_gt)), X_homo_rect)).T
        x2 = x2_homo[:, :2]/x2_homo[:, 2:3]
        if visualize:
            plt.figure(figsize=(30, 8))
            plt.imshow(im_r)
            val_inds = utils_vis.scatter_xy(x2, x2_homo[:, 2], self.im_shape, 'Reprojection to cam 3 with rectified X and camera', new_figure=False)
        else:
            val_inds = utils_misc.within(x1[:, 0], x1[:, 1], self.im_shape[1], self.im_shape[0])
    #     print(val_inds.shape)
        val_inds_list.append(val_inds)

        val_inds_both = val_inds_list[0] & val_inds_list[1]
    #     print(val_inds_both.shape)
        val_idxes = [idx for idx in range(val_inds_both.shape[0]) if val_inds_both[idx]] # within indexes
        return val_idxes, X_rect

    def rectify_all(self):
        # for each frame, get the visible points on front view with identity left camera, as well as indexes of points on both left/right images
        self.val_idxes_list = []
        self.X_rect_list = []
        for i in range(self.N_frames):
            velo = list(self.dataset.velo)[i] # [N, 4]
            velo = velo[:, :3]
            velo_reproj = utils_misc.homo_np(velo)
            val_idxes, X_rect = self.rectify(velo_reproj, self.dataset_rgb[i][0], self.dataset_rgb[i][1], visualize=(i%100==0))
            self.val_idxes_list.append(val_idxes)
            self.X_rect_list.append(X_rect)