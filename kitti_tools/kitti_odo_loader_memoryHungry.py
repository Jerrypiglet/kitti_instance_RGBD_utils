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
                 BF_matcher=False,
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
        # self.test_seqs = [10]
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
            self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.sift_num, contrastThreshold=1e-5)
            self.bf = cv2.BFMatcher(normType=cv2.NORM_L2)
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)
            self.sift_matcher = self.bf if BF_matcher else self.flann

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
        for c in self.cam_ids:
            scene_data = {'cid': c, 'cid_num': self.cid_to_num[c], 'dir': Path(drive_path), 'rel_path': Path(drive_path).name + '_' + c}
            img_dir = os.path.join(drive_path, 'image_%d'%scene_data['cid_num'])
            scene_data['img_files'] = sorted(glob(img_dir + '/*.png'))
            scene_data['N_frames'] = len(scene_data['img_files'])
            scene_data['frame_ids'] = ['{:06d}'.format(i) for i in range(scene_data['N_frames'])]

            # Check images and optionally get SIFT
            img_shape = None
            zoom_xy = None
            if self.get_sift:
                logging.info('Getting SIFT...'+drive_path)
                scene_data['sift_kp'] = []
                scene_data['sift_des'] = []
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
                if self.get_sift:
                    # logging.info('Getting sift for frame %d/%d.'%(idx, scene_data['N_frames']))
                    kp, des = self.sift.detectAndCompute(img, None) ## IMPORTANT: normalize these points
                    x_all = np.array([p.pt for p in kp])
                    if x_all.shape[0] != self.sift_num:
                        choice = crop_or_pad_choice(x_all.shape[0], self.sift_num, shuffle=True)
                        x_all = x_all[choice]
                        des = des[choice]
                    scene_data['sift_kp'].append(x_all)
                    scene_data['sift_des'].append(des)
            if self.get_sift:
                assert scene_data['N_frames']==len(scene_data['sift_kp']), 'scene_data[N_frames]!=len(scene_data[sift_kp]), %d!=%d'%(scene_data['N_frames'], len(scene_data['sift_kp']))

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

            # Get velo
            if self.get_X:
                logging.info('Getting X...'+drive_path)
                # for each frame, get the visible points on front view with identity left camera, as well as indexes of points on both left/right images
                val_idxes_list = []
                X_rect_list = []
                X_cam0_list = []
                for idx in tqdm(range(scene_data['N_frames'])):
                    velo = self.load_velo(scene_data, idx)
                    if velo is None:
                        break
                    velo_homo = utils_misc.homo_np(velo)
                    val_idxes, X_rect, X_cam0 = rectify(velo_homo, scene_data['calibs']) # list, [N, 3]
                    val_idxes_list.append(val_idxes)
                    X_rect_list.append(X_rect)
                    X_cam0_list.append(X_cam0)
                if velo is None and idx==0:
                    logging.warning('0 velo in %s. Skipped.'%drive_path)
                    return []
                scene_data['val_idxes'] = val_idxes_list
                scene_data['X_cam2'] = X_rect_list
                scene_data['X_cam0'] = X_cam0_list
                # Check number of velo frames
                assert scene_data['N_frames']==len(scene_data['X_cam2']), 'scene_data[N_frames]!=len(scene_data[X_cam2]), %d!=%d'%(scene_data['N_frames'], len(scene_data['X_cam2']))

            train_scenes.append(scene_data)
        return train_scenes

    def scene_data_to_samples(self, scene_data):
        def construct_sample(scene_data, i, frame_id, show_zoom_info):
            sample = {"img":self.load_image(scene_data, i, show_zoom_info)[0], "id":frame_id}
            if self.get_X:
                sample['X_cam2_vis'] = scene_data['X_cam2'][i][scene_data['val_idxes'][i]]
                sample['X_cam0_vis'] = scene_data['X_cam0'][i][scene_data['val_idxes'][i]]
            if self.get_pose:
                sample['pose'] = scene_data['poses'][i]
            if self.get_sift:
                sample['sift_kp'] = scene_data['sift_kp'][i]
                sample['sift_des'] = scene_data['sift_des'][i]
            return sample

        all_imgs = []
        for (i,frame_id) in enumerate(scene_data['frame_ids']):
            all_imgs.append(construct_sample(scene_data, i, frame_id, show_zoom_info=False))

        return all_imgs

    def dump_drive(self, args, drive_path, split, scene_data=None):
        assert split in ['train', 'test']
        if scene_data is None:
            train_scenes = self.collect_scene_from_drive(drive_path)
            if not train_scenes:
                logging.warning('Empty scene data for %s. Skipped.'%drive_path)
                return
            assert len(train_scenes)==1, 'More than one camera read not supported! %d'%len(train_scenes)
            scene_data = train_scenes[0]
        scene_samples = self.scene_data_to_samples(scene_data)

        dump_dir = Path(args.dump_root)/scene_data['rel_path']
        dump_dir.mkdir_p()

        intrinsics = scene_data['calibs']['K']
        dump_cam_file = dump_dir/'cam'
        np.save(dump_cam_file+'.npy', intrinsics)

        dump_Rt_cam2_gt_file = dump_dir/'Rt_cam2_gt'
        np.save(dump_Rt_cam2_gt_file, scene_data['Rt_cam2_gt'])

        poses_file = dump_dir/'poses'
        poses = []

        logging.info('Dumping %d samples to %s...'%(len(scene_samples), dump_dir))
        sample_name_list = []
        for ii, sample in enumerate(scene_samples):
            # logging.info('Dumping %d/%d.'%(ii, len(scene_samples)))
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
            for delta_ij in delta_ijs:
                for ii in range(len(scene_samples)-delta_ij):
                    jj = ii + delta_ij
                    all_ij, good_ij = self.get_sift_match_idx_pair(scene_samples[ii]['sift_des'], scene_samples[jj]['sift_des'])
                    dump_ij_idx_file = dump_dir/'ij_idx_{}-{}'.format(ii, jj)
                    if self.save_npy:
                        np.save(dump_ij_idx_file+'_all_ij.npy', all_ij)
                        np.save(dump_ij_idx_file+'_good_ij.npy', good_ij)
                    else:
                        dump_ij_idx_dict = {'all_ij': all_ij, 'good_ij': good_ij}
                        saveh5(dump_ij_idx_dict, dump_ij_idx_file+'.h5')

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

    def load_velo(self, scene_data, tgt_idx):
        velo_file = scene_data['dir']/'velodyne'/scene_data['frame_ids'][tgt_idx]+'.bin'
        if not velo_file.isfile():
            logging.warning('Velo file %s not found!'%velo_file)
            return None
        velo = load_velo_scan(velo_file)[:, :3]
        return velo

    def read_odo_calib_file(self, filepath, cid=2):
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

    def get_P_rect(self, scene_data, calibs, get_2cam_dict=True):
        # calib_file = scene_data['dir'].parent/'calib_cam_to_cam.txt'
        calib_file = scene_data['dir']/'calib.txt'
        if get_2cam_dict:
            P_rect = {}
            for cid in ['00', '01', '02', '03']:
                P_rect[cid], _ = self.read_odo_calib_file(calib_file, cid=self.cid_to_num[cid])
                if calibs['rescale']:
                    P_rect[cid] = scale_P(P_rect[cid], calibs['zoom_xy'][0], calibs['zoom_xy'][1])
            return P_rect
        else:
            P_rect, _ = self.read_odo_calib_file(calib_file, cid=self.cid_to_num[cid])
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

    def get_sift_match_idx_pair(self, des1, des2):
        matches = self.sift_matcher.knnMatch(des1, des2, k=2) # another option is https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py#L309
        # store all the good matches as per Lowe's ratio test.
        good = []
        all_m = []
        for m,n in matches:
            all_m.append(m)
            if m.distance < 0.8*n.distance:
                good.append(m)

        good_ij = [[mat.queryIdx for mat in good], [mat.trainIdx for mat in good]]
        all_ij = [[mat.queryIdx for mat in all_m], [mat.trainIdx for mat in all_m]]
        return np.asarray(all_ij).T.copy(), np.asarray(good_ij).T.copy()
