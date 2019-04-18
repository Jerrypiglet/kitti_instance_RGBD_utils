""" For use in dumping single frame ground truths of KITTI Odometry Dataset
Adapted from https://github.com/ClementPinard/SfmLearner-Pytorch/blob/0caec9ed0f83cb65ba20678a805e501439d2bc25/data/kitti_raw_loader.py

Rui Zhu, rzhu@eng.ucsd.edu, 2019
"""

from __future__ import division
import numpy as np
from path import Path
from tqdm import tqdm
import scipy.misc
from collections import Counter
from pebble import ProcessPool
import multiprocessing as mp
ratio_CPU = 0.8
default_number_of_process = int(ratio_CPU * mp.cpu_count())

import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
import traceback

import coloredlogs, logging
logging.basicConfig()
logger = logging.getLogger()
coloredlogs.install(level='INFO', logger=logger)

import cv2

from kitti_tools.utils_kitti import load_velo_scan, rectify, read_calib_file, transform_from_rot_trans, scale_intrinsics, scale_P
import dsac_tools.utils_misc as utils_misc
# from utils_good import *
from glob import glob
from dsac_tools.utils_misc import crop_or_pad_choice
from utils_kitti import load_as_float, load_as_array, load_sift

class KittiOdoLoader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=375,
                 img_width=1242,
                 cam_ids=['02'],
                 get_X=False,
                 get_pose=False,
                 get_sift=False,
                 sift_num=2000,
                 if_BF_matcher=False,
                 save_npy=True):
                 # depth_size_ratio=1):
        dir_path = Path(__file__).realpath().dirname()

        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.cam_ids = cam_ids
        # assert self.cam_ids == ['02'], 'Support left camera only!'
        self.cid_to_num = {'00': 0, '01': 1, '02': 2, '03': 3}
        self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.test_seqs = [9, 10]
        # self.train_seqs = [4]
        # self.test_seqs = []
        # self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # self.test_seqs = []
        self.map_to_raw = {'00': '2011_10_03_drive_0027', '01': '2011_10_03_drive_0042', '02': '2011_10_03_drive_0034', '03': '2011_09_26_drive_0067', \
            '04': '2011_09_30_drive_0016', '05': '2011_09_30_drive_0018', '06': '2011_09_30_drive_0020', '07': '2011_09_30_drive_0027', \
            '08': '2011_09_30_drive_0028', '09': '2011_09_30_drive_0033', '10': '2011_09_30_drive_0034'}

        self.get_X = get_X
        self.get_pose = get_pose
        self.get_sift = get_sift
        self.save_npy = save_npy
        if self.save_npy:
            logging.info('+++ Dumping as npy')
        else:
            logging.info('+++ Dumping as h5')
        if self.get_sift:
            self.sift_num = sift_num
            self.if_BF_matcher = if_BF_matcher
            self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.sift_num, contrastThreshold=1e-5)
            # self.bf = cv2.BFMatcher(normType=cv2.NORM_L2)
            # FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            # search_params = dict(checks = 50)
            # self.flann = cv2.FlannBasedMatcher(index_params, search_params)
            # self.sift_matcher = self.bf if BF_matcher else self.flann

        self.scenes = {'train': [], 'test': []}
        self.collect_train_folders()
        self.collect_test_folders()

    def collect_train_folders(self):
        for seq in self.train_seqs:
            seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
            self.scenes['train'].append(seq_dir)

    def collect_test_folders(self):
        for seq in self.test_seqs:
            seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
            self.scenes['test'].append(seq_dir)

    def collect_scene_from_drive(self, drive_path):
        train_scenes = []
        logging.info('Gathering info for %s...'%drive_path)
        for c in self.cam_ids:
            scene_data = {'cid': c, 'cid_num': self.cid_to_num[c], 'dir': Path(drive_path), 'rel_path': Path(drive_path).name + '_' + c}
            img_dir = os.path.join(drive_path, 'image_%d'%scene_data['cid_num'])
            scene_data['img_files'] = sorted(glob(img_dir + '/*.png'))
            scene_data['N_frames'] = len(scene_data['img_files'])
            assert scene_data['N_frames'] != 0, 'No file found for %s!'%drive_path
            scene_data['frame_ids'] = ['{:06d}'.format(i) for i in range(scene_data['N_frames'])]

            img_shape = None
            zoom_xy = None
            show_zoom_info = True
            for idx in tqdm(range(scene_data['N_frames'])):
                img, zoom_xy = self.load_image(scene_data, idx, show_zoom_info)
                show_zoom_info = False
                if img is None and idx==0:
                    logging.warning('0 images in %s. Skipped.'%drive_path)
                    return []
                else:
                    if img_shape is not None:
                        assert img_shape == img.shape, 'Inconsistent image shape in seq %s!'%drive_path
                    else:
                        img_shape = img.shape
            # print(img_shape)
            scene_data['calibs'] = {'im_shape': [img_shape[0], img_shape[1]], 'zoom_xy': zoom_xy, 'rescale': True if zoom_xy != (1., 1.) else False}

            # Get geo params from the RAW dataset calibs
            P_rect_ori_dict = self.get_P_rect(scene_data, scene_data['calibs'])
            intrinsics = P_rect_ori_dict[c][:,:3]
            calibs_rects = self.get_rect_cams(intrinsics, P_rect_ori_dict['02'])

            drive_in_raw = self.map_to_raw[drive_path[-2:]]
            date = drive_in_raw[:10]
            seq = drive_in_raw[-4:]
            calib_path_in_raw = Path(self.dataset_dir)/'raw'/date
            imu2velo_dict = read_calib_file(calib_path_in_raw/'calib_imu_to_velo.txt')
            velo2cam_dict = read_calib_file(calib_path_in_raw/'calib_velo_to_cam.txt')
            cam2cam_dict = read_calib_file(calib_path_in_raw/'calib_cam_to_cam.txt')
            velo2cam_mat = transform_from_rot_trans(velo2cam_dict['R'], velo2cam_dict['T'])
            imu2velo_mat = transform_from_rot_trans(imu2velo_dict['R'], imu2velo_dict['T'])
            cam_2rect_mat = transform_from_rot_trans(cam2cam_dict['R_rect_00'], np.zeros(3))
            scene_data['calibs'].update({'K': intrinsics, 'P_rect_ori_dict': P_rect_ori_dict, 'cam_2rect': cam_2rect_mat, 'velo2cam': velo2cam_mat})
            scene_data['calibs'].update(calibs_rects)

            # Get pose
            poses = np.genfromtxt(self.dataset_dir/'poses'/'{}.txt'.format(drive_path[-2:])).astype(np.float64).reshape(-1, 3, 4)
            assert scene_data['N_frames']==poses.shape[0], 'scene_data[N_frames]!=poses.shape[0], %d!=%d'%(scene_data['N_frames'], poses.shape[0])
            scene_data['poses'] = poses

            scene_data['Rt_cam2_gt'] = scene_data['calibs']['Rtl_gt']

            train_scenes.append(scene_data)
        return train_scenes

    def construct_sample(self, scene_data, idx, frame_id, show_zoom_info):
        img = self.load_image(scene_data, idx, show_zoom_info)[0]
        sample = {"img":img, "id":frame_id}
        if self.get_X:
            velo = load_velo(scene_data, idx)
            if velo is None:
                logging.error('0 velo in %s. Skipped.'%scene_data['dir'])
            velo_homo = utils_misc.homo_np(velo)
            val_idxes, X_rect, X_cam0 = rectify(velo_homo, scene_data['calibs']) # list, [N, 3]
            sample['X_cam2_vis'] = X_rect[val_idxes]
            sample['X_cam0_vis'] = X_cam0[val_idxes]
        if self.get_pose:
            sample['pose'] = scene_data['poses'][idx]
        if self.get_sift:
            # logging.info('Getting sift for frame %d/%d.'%(idx, scene_data['N_frames']))
            kp, des = self.sift.detectAndCompute(img, None) ## IMPORTANT: normalize these points
            x_all = np.array([p.pt for p in kp])
            if x_all.shape[0] != self.sift_num:
                choice = crop_or_pad_choice(x_all.shape[0], self.sift_num, shuffle=True)
                x_all = x_all[choice]
                des = des[choice]
            sample['sift_kp'] = x_all
            sample['sift_des'] = des
        return sample

    def dump_drive(self, args, drive_path, split, scene_data=None):
        assert split in ['train', 'test']
        if scene_data is None:
            train_scenes = self.collect_scene_from_drive(drive_path)
            if not train_scenes:
                logging.warning('Empty scene data for %s. Skipped.'%drive_path)
                return
            assert len(train_scenes)==1, 'More than one camera not supported! %d'%len(train_scenes)
            scene_data = train_scenes[0]

        dump_dir = Path(args.dump_root)/scene_data['rel_path']
        dump_dir.mkdir_p()
        intrinsics = scene_data['calibs']['K']
        dump_cam_file = dump_dir/'cam'
        np.save(dump_cam_file+'.npy', intrinsics)
        dump_Rt_cam2_gt_file = dump_dir/'Rt_cam2_gt'
        np.save(dump_Rt_cam2_gt_file, scene_data['Rt_cam2_gt'])
        poses_file = dump_dir/'poses'
        poses = []

        logging.info('Dumping %d samples to %s...'%(scene_data['N_frames'], dump_dir))
        sample_name_list = []
        # sift_des_list = []
        for idx in tqdm(range(scene_data['N_frames'])):
            frame_id = scene_data['frame_ids'][idx]
            assert int(frame_id)==idx
            sample = self.construct_sample(scene_data, idx, frame_id, show_zoom_info=False)

            img, frame_nb = sample["img"], sample["id"]
            dump_img_file = dump_dir/'{}.jpg'.format(frame_nb)
            scipy.misc.imsave(dump_img_file, img)
            if "pose" in sample.keys():
                poses.append(sample["pose"])
            if "X_cam0_vis" in sample.keys():
                dump_X_cam0_file = dump_dir/'X_cam0_{}'.format(frame_nb)
                dump_X_cam2_file = dump_dir/'X_cam2_{}'.format(frame_nb)
                if self.save_npy:
                    np.save(dump_X_cam0_file+'.npy', sample["X_cam0_vis"])
                    np.save(dump_X_cam2_file+'.npy', sample["X_cam2_vis"])
                else:
                    saveh5({"X_cam0_vis": sample["X_cam0_vis"], "X_cam2_vis": sample["X_cam2_vis"]}, dump_X_file+'.h5')
            if "sift_kp" in sample.keys():
                dump_sift_file = dump_dir/'sift_{}'.format(frame_nb)
                if self.save_npy:
                    np.save(dump_sift_file+'.npy', np.hstack((sample['sift_kp'], sample['sift_des'])))
                else:
                    saveh5({'sift_kp': sample['sift_kp'], 'sift_des': sample['sift_des']}, dump_sift_file+'.h5')
                # sift_des_list.append(sample['sift_des'])

            sample_name_list.append('%s %s'%(dump_dir[-5:], frame_nb))

        # Get all poses    
        if "pose" in sample.keys():      
            if len(poses) != 0:
                # np.savetxt(poses_file, np.array(poses).reshape(-1, 16), fmt='%.20e')a
                if self.save_npy:
                    np.save(poses_file+'.npy', np.stack(poses).reshape(-1, 3, 4))
                else:
                    saveh5({"poses": np.array(poses).reshape(-1, 3, 4)}, poses_file+'.h5')

        # Get SIFT matches
        if self.get_sift:
            delta_ijs = [1, 2, 3, 5, 8, 10]
            # delta_ijs = [1]
            num_tasks = len(delta_ijs)
            num_workers = min(len(delta_ijs), default_number_of_process)
            # num_workers = 1
            logging.info('Getting SIFT matches on %d workers for delta_ijs = %s'%(num_workers, ' '.join(str(e) for e in delta_ijs)))

            with ProcessPool(max_workers=num_workers) as pool:
                tasks = pool.map(dump_match_idx, delta_ijs, [scene_data['N_frames']]*num_tasks, \
                    [dump_dir]*num_tasks, [self.save_npy]*num_tasks, [self.if_BF_matcher]*num_tasks)
                try:
                    for _ in tqdm(tasks.result(), total=num_tasks):
                        pass
                except KeyboardInterrupt as e:
                    tasks.cancel()
                    raise e

            # for delta_ij in delta_ijs:
            #     dump_match_idx(delta_ij, scene_data['N_frames'], sift_des_list, dump_dir, self.save_npy, self.if_BF_matcher)

        if len(dump_dir.files('*.jpg')) < 2:
            dump_dir.rmtree()

        return sample_name_list

    def load_image(self, scene_data, tgt_idx, show_zoom_info=True):
        img_file = scene_data['dir']/'image_{}'.format(scene_data['cid_num'])/scene_data['frame_ids'][tgt_idx]+'.png'
        if not img_file.isfile():
            logging.warning('Image %s not found!'%img_file)
            return None, None, None
        img = scipy.misc.imread(img_file)
        if [self.img_height, self.img_width] == [img.shape[0], img.shape[1]]:
            return img, (1., 1.)
        else:
            zoom_y = self.img_height/img.shape[0]
            zoom_x = self.img_width/img.shape[1]
            if show_zoom_info:
                logging.warning('[%s] Zooming the image (H%d, W%d) with zoom_xW=%f, zoom_yH=%f to (H%d, W%d).'%\
                    (img_file, img.shape[0], img.shape[1], zoom_x, zoom_y, self.img_height, self.img_width))
            img = scipy.misc.imresize(img, (self.img_height, self.img_width))
            return img, (zoom_x, zoom_y)

    def get_P_rect(self, scene_data, calibs, get_2cam_dict=True):
        # calib_file = scene_data['dir'].parent/'calib_cam_to_cam.txt'
        calib_file = scene_data['dir']/'calib.txt'
        if get_2cam_dict:
            P_rect = {}
            for cid in ['00', '01', '02', '03']:
                P_rect[cid], _ = read_odo_calib_file(calib_file, cid=self.cid_to_num[cid])
                if calibs['rescale']:
                    P_rect[cid] = scale_P(P_rect[cid], calibs['zoom_xy'][0], calibs['zoom_xy'][1])
            return P_rect
        else:
            P_rect, _ = read_odo_calib_file(calib_file, cid=self.cid_to_num[cid])
            if calibs['rescale']:
                P_rect = scale_P(P_rect, calibs['zoom_xy'][0], calibs['zoom_xy'][1])
        return P_rect

    def get_rect_cams(self, K, P_rect_20):
        Ml_gt = np.matmul(np.linalg.inv(K), P_rect_20)
        tl_gt = Ml_gt[:, 3:4]
        Rl_gt = Ml_gt[:, :3]
        Rtl_gt = np.vstack((np.hstack((Rl_gt, tl_gt)), np.array([0., 0., 0., 1.], dtype=np.float64)))
        calibs_rects = {'Rtl_gt': Rtl_gt}
        return calibs_rects

def dump_match_idx(delta_ij, N_frames, dump_dir, save_npy, if_BF_matcher):
    if if_BF_matcher: # OpenCV sift matcher must be created inside each thread (because it does not support sharing across threads!)
        bf = cv2.BFMatcher(normType=cv2.NORM_L2)
        sift_matcher = bf
    else:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        sift_matcher = flann

    for ii in tqdm(range(N_frames-delta_ij)):
        jj = ii + delta_ij

        _, sift_des_ii = load_sift(dump_dir, '%06d'%ii, ext='.npy' if save_npy else '.h5')
        _, sift_des_jj = load_sift(dump_dir, '%06d'%jj, ext='.npy' if save_npy else '.h5')

        # all_ij, good_ij = get_sift_match_idx_pair(sift_matcher, sift_des_list[ii], sift_des_list[jj])
        all_ij, good_ij, quality = get_sift_match_idx_pair(sift_matcher, sift_des_ii.copy(), sift_des_jj.copy())
        if all_ij is None:
            logging.warning('KNN match failed dumping %s frame %d-%d. Skipping'%(dump_dir, ii, jj))
            continue
        dump_ij_idx_file = dump_dir/'ij_idx_{}-{}'.format(ii, jj)
        if save_npy:
            np.save(dump_ij_idx_file+'_all_ij.npy', all_ij)
            np.save(dump_ij_idx_file+'_good_ij.npy', good_ij)
            np.save(dump_ij_idx_file+'_quality_ij.npy', quality)
        else:
            dump_ij_idx_dict = {'all_ij': all_ij, 'good_ij': good_ij}
            saveh5(dump_ij_idx_dict, dump_ij_idx_file+'.h5')

def get_sift_match_idx_pair(sift_matcher, des1, des2):
    try:
        matches = sift_matcher.knnMatch(des1, des2, k=2) # another option is https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py#L309
    except Exception as e:
        logging.error(traceback.format_exception(*sys.exc_info()))
        return None, None
    # store all the good matches as per Lowe's ratio test.
    good = []
    all_m = []
    quality = []
    for m,n in matches:
        all_m.append(m)
        if m.distance < 0.8*n.distance:
            good.append(m)
            quality.append([m.distance, m.distance / n.distance])
            print

    good_ij = [[mat.queryIdx for mat in good], [mat.trainIdx for mat in good]]
    all_ij = [[mat.queryIdx for mat in all_m], [mat.trainIdx for mat in all_m]]
    return np.asarray(all_ij).T.copy(), np.asarray(good_ij).T.copy(), np.asarray(quality).copy(), 

def load_velo(scene_data, tgt_idx):
    velo_file = scene_data['dir']/'velodyne'/scene_data['frame_ids'][tgt_idx]+'.bin'
    if not velo_file.isfile():
        logging.warning('Velo file %s not found!'%velo_file)
        return None
    velo = load_velo_scan(velo_file)[:, :3]
    return velo

def read_odo_calib_file(filepath, cid=2):
    # From https://github.com/tinghuiz/SfMLearner/blob/master/data/kitti/kitti_odom_loader.py#L133
    """Read in a calibration file and parse into a dictionary."""
    with open(filepath, 'r') as f:
        C = f.readlines()
    def parseLine(L, shape):
        data = L.split()
        data = np.array(data[1:]).reshape(shape).astype(np.float32)
        return data
    proj_c2p = parseLine(C[cid], shape=(3,4))
    proj_v2c = parseLine(C[-1], shape=(3,4))
    filler = np.array([0, 0, 0, 1]).reshape((1,4))
    proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
    return proj_c2p, proj_v2c
