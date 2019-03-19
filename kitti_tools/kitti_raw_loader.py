# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/0caec9ed0f83cb65ba20678a805e501439d2bc25/data/kitti_raw_loader.py

from __future__ import division
import numpy as np
from path import Path
from tqdm import tqdm
import scipy.misc
from collections import Counter

import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import cv2

from kitti_tools.utils_kitti import *
from utils_good import *

class KittiRawLoader(object):
    def __init__(self,
                 dataset_dir,
                 test_scene_file=None,
                 static_frames_file=None,
                 # img_height=128,
                 # img_width=416,
                 min_speed=2,
                 get_X=False,
                 get_pose=False,
                 get_sift=False,
                 sift_num=2000,
                 BF_matcher=True):
                 # depth_size_ratio=1):
        dir_path = Path(__file__).realpath().dirname()
        # test_scene_file = dir_path/'test_scenes.txt'

        self.from_speed = static_frames_file is None
        if static_frames_file is not None:
            static_frames_file = Path(static_frames_file)
            self.collect_static_frames(static_frames_file)

        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t[:-1] for t in test_scenes]
        # self.test_scenes = []

        self.dataset_dir = Path(dataset_dir)
        # self.img_height = img_height
        # self.img_width = img_width
        self.cam_ids = ['02', '03']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        self.min_speed = min_speed
        self.get_X = get_X
        self.get_pose = get_pose
        self.get_sift = get_sift
        if self.get_sift:
            self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=sift_num, contrastThreshold=1e-5)
            self.bf = cv2.BFMatcher(normType=cv2.NORM_L2)
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)
            self.sift_matcher = self.bf if BF_matcher else self.flann

        # self.depth_size_ratio = depth_size_ratio
        self.collect_train_folders()

        # self.kitti_two_frame_loader = KittiLoader(self.dataset_dir)

    def collect_static_frames(self, static_frames_file):
        with open(static_frames_file, 'r') as f:
            frames = f.readlines()
        self.static_frames = {}
        for fr in frames:
            if fr == '\n':
                continue
            date, drive, frame_id = fr.split(' ')
            curr_fid = '%.10d' % (np.int(frame_id[:-1]))
            if drive not in self.static_frames.keys():
                self.static_frames[drive] = []
            self.static_frames[drive].append(curr_fid)
        logging.info('Static frames collected from %s.'%static_frames_file)

    def collect_train_folders(self):
        self.scenes = []
        for date in self.date_list:
            drive_set = (self.dataset_dir/date).dirs()
            for dr in drive_set:
                if dr.name[:-5] not in self.test_scenes:
                    self.scenes.append(dr)

    def get_drive_path(self, date, drive):
        drive_path = self.dataset_dir + '/%s/%s_drive_%s_sync'%(date, date, drive)
        return drive_path

    def collect_scenes(self, drive_path):
        logging.info('Collecting ' + drive_path)
        path = drive_path.rstrip('/')
        basedir = path.rsplit('/',2)[0]
        date = path.split('/')[-2]
        drive = path.split('/')[-1].split('_')[-2]

        kitti_two_frame_loader = KittiLoader(self.dataset_dir)
        kitti_two_frame_loader.set_drive(date, drive, drive_path)
        if kitti_two_frame_loader.N_frames == 0:
            logging.warning('0 frames in %s. Skipped.'%drive_path)
            return []
        kitti_two_frame_loader.get_left_right_gt()
        scene_data = kitti_two_frame_loader.load_cam_poses()
        # self.kitti_two_frame_loader.show_demo()
        assert len(scene_data['imu_pose_matrix']) == kitti_two_frame_loader.N_frames, \
            '[Error] Unequal lengths of imu_pose_matrix:%d, N_frames:%d!'%(len(scene_data['imu_pose_matrix']), kitti_two_frame_loader.N_frames)

        if self.get_X:
            val_idxes_list, X_rect_list = kitti_two_frame_loader.rectify_all(visualize=False)
            scene_data['val_idxes'] = val_idxes_list
            if len(val_idxes_list) == len(X_rect_list):
                scene_data['X_rect'] = X_rect_list
            else:
                logging.error('Unequal lengths of imu_pose_matrix:%d, val_idxes_list:%d, X_rect_list:%d! Not saving X_rect for %s-%s'%(len(scene_data['imu_pose_matrix']), len(val_idxes_list), len(X_rect_list), date, drive))
        scene_data['intrinsics'] = kitti_two_frame_loader.K
        scene_data['img_l'] = [im[0] for im in kitti_two_frame_loader.dataset_rgb]

        if self.get_sift:
            scene_data['sift_kp'] = []
            scene_data['sift_des'] = []            
            for idx in range(kitti_two_frame_loader.N_frames):
            # for idx in range(1):
                kp, des = self.sift.detectAndCompute(np.array(kitti_two_frame_loader.dataset_rgb[idx][0]), None) ## IMPORTANT: normalize these points
                x_all = np.array([p.pt for p in kp])
                scene_data['sift_kp'].append(x_all)
                scene_data['sift_des'].append(des)

        if self.get_pose:
            velo2cam_mat = kitti_two_frame_loader.dataset.calib.T_cam0_velo_unrect
            imu2velo_mat = kitti_two_frame_loader.dataset.calib.T_velo_imu
            cam_2rect_mat = kitti_two_frame_loader.dataset.calib.R_rect_00
            imu2cam = kitti_two_frame_loader.Rtl_gt @ cam_2rect_mat @ velo2cam_mat @ imu2velo_mat
            scene_data['imu2cam'] = imu2cam

        return [scene_data]

    def get_scene_imgs(self, scene_data):
        def construct_sample(scene_data, i, frame_id):
            # sample = {"img":self.load_image(scene_data, i)[0], "id":frame_id}
            sample = {"img": scene_data['img_l'][i], "id":frame_id}

            # if self.get_depth:
            #     sample['depth'] = self.generate_depth_map(scene_data, i)
            if self.get_X:
                sample['X_rect_vis'] = scene_data['X_rect'][i][:, scene_data['val_idxes'][i]]
            if self.get_pose:
                sample['imu_pose_matrix'] = scene_data['imu_pose_matrix'][i]
            if self.get_sift:
                sample['sift_kp'] = scene_data['sift_kp'][i]
                sample['sift_des'] = scene_data['sift_des'][i]
            return sample

        drive = str(scene_data['dir'].name)
        all_imgs = []
        for (i,frame_id) in enumerate(scene_data['frame_id']):
            if (drive not in self.static_frames.keys()) or (frame_id not in self.static_frames[drive]):
                # yield construct_sample(scene_data, i, frame_id)
                all_imgs.append(construct_sample(scene_data, i, frame_id))
        return all_imgs

    def dump_drive(self, args, drive_path, scene_list=None):
        if scene_list is None:
            scene_list = self.collect_scenes(drive_path)
            if not scene_list:
                return
        for scene_data in scene_list:
            dump_dir = Path(args.dump_root)/scene_data['rel_path']
            logging.info('Dumping to ' + dump_dir)
            # dump_dir = Path(args.dump_root)
            dump_dir.mkdir_p()

            intrinsics = scene_data['intrinsics']
            dump_cam_file = dump_dir/'cam'
            # np.savetxt(dump_cam_file, intrinsics)
            np.save(dump_cam_file+'.npy', intrinsics)
            
            dump_imu2cam_file = dump_dir/'imu2cam'
            np.save(dump_imu2cam_file, scene_data['imu2cam'])

            poses_file = dump_dir/'imu_pose_matrixs'
            poses = []

            scene_samples = self.get_scene_imgs(scene_data)
            for ii, sample in enumerate(scene_samples):
                img, frame_nb = sample["img"], sample["id"]
                dump_img_file = dump_dir/'{}.jpg'.format(frame_nb)
                scipy.misc.imsave(dump_img_file, img)
                if "imu_pose_matrix" in sample.keys():
                    poses.append(sample["imu_pose_matrix"].reshape(-1).tolist())
                    if len(poses) != 0:
                        # np.savetxt(poses_file, np.array(poses).reshape(-1, 16), fmt='%.20e')
                        # np.save(poses_file+'.h5', np.array(poses).reshape(-1, 16))
                        saveh5({"pose": np.array(poses).reshape(-1, 16)}, poses_file+'.h5')
                if "X_rect_vis" in sample.keys():
                    dump_X_file = dump_dir/'{}_X'.format(frame_nb)
                    # np.save(dump_X_file, sample["X_rect_vis"])
                    saveh5({"X_rect_vis": sample["X_rect_vis"]}, dump_X_file+'.h5')
                if "sift_kp" in sample.keys():
                    dump_sift_file = dump_dir/'{}_sift'.format(frame_nb)
                    # np.save(dump_sift_file, np.hstack((sample['sift_kp'], sample['sift_des'])))
                    saveh5({'sift_kp': sample['sift_kp'], 'sift_des': sample['sift_des']}, dump_sift_file+'.h5')

            if self.get_sift:
                delta_ij = 1
                for ii in range(len(scene_samples)-delta_ij):
                    jj = ii + delta_ij
                    all_ij, good_ij = self.get_sift_match_idx_pair(scene_samples[ii]['sift_des'], scene_samples[jj]['sift_des'])
                    dump_ij_idx_dict = {'all_ij': all_ij, 'good_ij': good_ij}
                    dump_ij_idx_file = dump_dir/'ij_idx_{}-{}.h5'.format(frame_nb, ii, jj)
                    saveh5(dump_ij_idx_dict, dump_ij_idx_file)

            if len(dump_dir.files('*.jpg')) < 3:
                dump_dir.rmtree()

    def get_sift_match_idx_pair(self, des1, des2):
        matches = self.sift_matcher.knnMatch(des1, des2, k=2) # another option is https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py#L309
        # store all the good matches as per Lowe's ratio test.
        good = []
        all_m = []
        for m,n in matches:
            all_m.append(m)
            if m.distance < 0.7*n.distance:
                good.append(m)

        good_ij = [[mat.queryIdx for mat in good], [mat.trainIdx for mat in good]]
        all_ij = [[mat.queryIdx for mat in all_m], [mat.trainIdx for mat in all_m]]
        return np.asarray(all_ij).T, np.asarray(good_ij).T
