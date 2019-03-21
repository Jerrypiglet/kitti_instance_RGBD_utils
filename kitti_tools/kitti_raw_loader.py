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

from kitti_tools.utils_kitti import *
from utils_good import *

class KittiRawLoader(object):
    def __init__(self,
                 dataset_dir,
                 test_scene_file=None,
                 static_frames_file=None,
                 img_height=375,
                 img_width=1242,
                 min_speed=2.,
                 get_X=False,
                 get_pose=False,
                 get_sift=False,
                 sift_num=2000,
                 BF_matcher=False,
                 save_npy=True):
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
        self.img_height = img_height
        self.img_width = img_width
        self.cam_ids = ['02']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        self.min_speed = min_speed
        self.get_X = get_X
        self.get_pose = get_pose
        self.get_sift = get_sift
        self.save_npy = save_npy
        if self.save_npy:
            logging.info('+++ Dumping as npy')
        else:
            logging.info('+++ Dumping as h5')
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

    def load_image(self, scene_data, tgt_idx):
        img_file = scene_data['dir']/'image_{}'.format(scene_data['cid'])/'data'/scene_data['frame_id'][tgt_idx]+'.png'
        if not img_file.isfile():
            logging.warning('Image %s not found!'%img_file)
            return None, None, None
        img = scipy.misc.imread(img_file)
        if [self.img_height, self.img_width] == [img.shape[0], img.shape[1]]:
            return img, 1., 1.
        else:
            zoom_y = self.img_height/img.shape[0]
            zoom_x = self.img_width/img.shape[1]
            img = scipy.misc.imresize(img, (self.img_height, self.img_width))
            # logging.warning('[%s] Zooming the image with zoom_x=%f, zoom_y=%f.'%(img_file, zoom_x, zoom_y))
            return img, zoom_x, zoom_y

    def load_velo(self, scene_data, tgt_idx):
        velo_file = scene_data['dir']/'velodyne_points'/'data'/scene_data['frame_id'][tgt_idx]+'.bin'
        if not velo_file.isfile():
            logging.warning('Velo file %s not found!'%velo_file)
            return None
        velo = load_velo_scan(velo_file)[:, :3]
        return velo

    def read_raw_calib_file(self, filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                        data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                        pass
        return data

    def get_P_rect(self, scene_data, zoom_x=1., zoom_y=1., get_2cam_dict=True):
        calib_file = scene_data['dir'].parent/'calib_cam_to_cam.txt'

        filedata = self.read_raw_calib_file(calib_file)
        if get_2cam_dict:
            P_rect = {}
            for cid in ['02', '03']:
                P_rect[cid] = np.reshape(filedata['P_rect_' + cid], (3, 4))
            return P_rect
        else:       
            P_rect = np.reshape(filedata['P_rect_' + scene_data['cid']], (3, 4))
            P_rect[0] *= zoom_x
            P_rect[1] *= zoom_y
            return P_rect

    def get_rect_cams(self, K, P_rect_20, P_rect_30):
        Ml_gt = np.matmul(np.linalg.inv(K), P_rect_20)
        Mr_gt = np.matmul(np.linalg.inv(K), P_rect_30)
        tl_gt = Ml_gt[:, 3:4]
        Rl_gt = Ml_gt[:, :3]
        tr_gt = Mr_gt[:, 3:4]
        Rr_gt = Mr_gt[:, :3]
        Rtl_gt = np.vstack((np.hstack((Rl_gt, tl_gt)), np.array([0., 0., 0., 1.], dtype=np.float64)))
        delta_Rtlr_gt = np.matmul(np.hstack((Rr_gt, tr_gt)), np.linalg.inv(Rtl_gt))
        delta_Rlr_gt = delta_Rtlr_gt[:, :3]
        delta_tlr_gt = delta_Rtlr_gt[:, 3:4]
        calibs_rects = {'Rtl_gt': Rtl_gt, 'delta_Rtlr_gt': delta_Rtlr_gt, 'delta_Rlr_gt': delta_Rlr_gt, 'delta_tlr_gt': delta_tlr_gt}
        return calibs_rects


    def collect_scene_from_drive(self, drive_path):
        train_scenes = []
        for c in self.cam_ids:
            oxts_path = drive_path + '/oxts/data/*.txt'
            oxts = sorted(glob.glob(oxts_path))

            scene_data = {'cid': c, 'dir': Path(drive_path), 'speed': [], 'frame_id': [], 'imu_pose_matrix':[], 'rel_path': Path(drive_path).name + '_' + c}
            scene_data['P_rect_ori_dict'] = self.get_P_rect(scene_data, get_2cam_dict=True)
            scene_data['intrinsics_ori'] = scene_data['P_rect_ori_dict']['02'][:,:3]
            calibs_rects = self.get_rect_cams(scene_data['intrinsics_ori'], scene_data['P_rect_ori_dict']['02'], scene_data['P_rect_ori_dict']['03'])

            scale = None
            origin = None
            imu2velo_dict = read_calib_file(drive_path+'/../calib_imu_to_velo.txt')
            velo2cam_dict = read_calib_file(drive_path+'/../calib_velo_to_cam.txt')
            cam2cam_dict = read_calib_file(drive_path+'/../calib_cam_to_cam.txt')
            velo2cam_mat = transform_from_rot_trans(velo2cam_dict['R'], velo2cam_dict['T'])
            imu2velo_mat = transform_from_rot_trans(imu2velo_dict['R'], imu2velo_dict['T'])
            cam_2rect_mat = transform_from_rot_trans(cam2cam_dict['R_rect_00'], np.zeros(3))
            calibs = {'K': scene_data['intrinsics_ori'], 'cam_2rect': cam_2rect_mat, 'velo2cam': velo2cam_mat, \
                'im_shape': [self.img_height, self.img_width]}
            calibs.update(calibs_rects)

            scene_data['imu2cam'] = calibs['Rtl_gt'] @ cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

            logging.info('Getting imu poses...'+drive_path)
            for n, f in enumerate(oxts):
                metadata = np.genfromtxt(f)
                speed = metadata[8:11]
                scene_data['speed'].append(speed)
                scene_data['frame_id'].append('{:010d}'.format(n))
                drive = str(scene_data['dir'].name)
                # print(self.static_frames.keys())
                # print(self.static_frames[drive])
                # print(drive)
                # print(scene_data['frame_id'][-1])
                # if not((drive not in self.static_frames.keys()) or (scene_data['frame_id'][-1] not in self.static_frames[drive])):
                #     logging.warning('%s in static_frames.keys() and scene_data[%s] in static_frames[drive]'%(drive, scene_data['frame_id'][-1]))
                #     return []
                lat = metadata[0]
                if scale is None:
                    scale = np.cos(lat * np.pi / 180.)
                imu_pose_matrix = pose_from_oxts_packet(metadata[:6], scale)
                scene_data['imu_pose_matrix'].append(imu_pose_matrix.copy())

            scene_data['N_frames'] = len(scene_data['imu_pose_matrix'])
            if scene_data['N_frames'] == 0:
                logging.warning('0 oxt frames in %s. Skipped.'%drive_path)
                return []

            scene_data['sift_kp'] = []
            scene_data['sift_des'] = []

            # Check images and optionally get SIFT
            if self.get_sift:
                logging.info('Getting SIFT...'+drive_path)
            for idx in range(scene_data['N_frames']):
                img, _, _ = self.load_image(scene_data, idx)
                if img is None and idx==0:
                    logging.warning('0 images in %s. Skipped.'%drive_path)
                    return []
                if self.get_sift:
                    # logging.info('Getting sift for frame %d/%d.'%(idx, scene_data['N_frames']))
                    kp, des = self.sift.detectAndCompute(img, None) ## IMPORTANT: normalize these points
                    x_all = np.array([p.pt for p in kp])
                    scene_data['sift_kp'].append(x_all)
                    scene_data['sift_des'].append(des)
            if self.get_sift:
                assert scene_data['N_frames']==len(scene_data['sift_kp']), 'scene_data[N_frames]!=len(scene_data[sift_kp]), %d!=%d'%(scene_data['N_frames'], len(scene_data['sift_kp']))

            if self.get_X:
                logging.info('Getting X, rectifying...'+drive_path)
                # for each frame, get the visible points on front view with identity left camera, as well as indexes of points on both left/right images
                val_idxes_list = []
                X_rect_list = []
                for idx in range(scene_data['N_frames']):
                    velo = self.load_velo(scene_data, idx)
                    if velo is None:
                        break
                    velo_reproj = utils_misc.homo_np(velo)
                    val_idxes, X_rect = rectify(velo_reproj, calibs)
                    val_idxes_list.append(val_idxes)
                    X_rect_list.append(X_rect)
                if velo is None and idx==0:
                    logging.warning('0 velo in %s. Skipped.'%drive_path)
                    return []
                scene_data['val_idxes'] = val_idxes_list
                scene_data['X_rect'] = X_rect_list
                # Check number of velo frames
                assert scene_data['N_frames']==len(scene_data['X_rect']), 'scene_data[N_frames]!=len(scene_data[X_rect]), %d!=%d'%(scene_data['N_frames'], len(scene_data['X_rect']))

            # Check number of image frames
            imgs_path = scene_data['dir']/'image_{}'.format(scene_data['cid'])/'data'
            imgs = sorted(imgs_path.files('*.png'))
            assert scene_data['N_frames']==len(imgs), 'scene_data[N_frames]!=len(imgs), %d!=%d'%(scene_data['N_frames'], len(imgs))

            # print(scene_data['N_frames'], len(imgs), len(scene_data['sift_kp']), len(scene_data['X_rect']))

            train_scenes.append(scene_data)
        return train_scenes

    # def collect_scene_from_drive_old(self, drive_path):
    #     logging.info('Collecting scene_data from drive %s.'%drive_path)
    #     path = drive_path.rstrip('/')
    #     basedir = path.rsplit('/',2)[0]
    #     date = path.split('/')[-2]
    #     drive = path.split('/')[-1].split('_')[-2]

    #     logging.info('Setting up kitti_two_frame_loader.')
    #     kitti_two_frame_loader = KittiLoader(self.dataset_dir)
    #     kitti_two_frame_loader.set_drive(date, drive, drive_path)
    #     if kitti_two_frame_loader.N_frames == 0:
    #         logging.warning('0 frames in %s. Skipped.'%drive_path)
    #         return []
    #     logging.info('get_left_right_gt')
    #     kitti_two_frame_loader.get_left_right_gt()
    #     logging.info('load_cam_poses')
    #     scene_data = kitti_two_frame_loader.load_cam_poses()
    #     # self.kitti_two_frame_loader.show_demo()
    #     assert len(scene_data['imu_pose_matrix']) == kitti_two_frame_loader.N_frames, \
    #         '[Error] Unequal lengths of imu_pose_matrix:%d, N_frames:%d!'%(len(scene_data['imu_pose_matrix']), kitti_two_frame_loader.N_frames)

    #     if self.get_X:
    #         logging.info('rectify_all')
    #         val_idxes_list, X_rect_list = kitti_two_frame_loader.rectify_all(visualize=False)
    #         scene_data['val_idxes'] = val_idxes_list
    #         if len(val_idxes_list) == len(X_rect_list):
    #             scene_data['X_rect'] = X_rect_list
    #         else:
    #             logging.error('Unequal lengths of imu_pose_matrix:%d, val_idxes_list:%d, X_rect_list:%d! Not saving X_rect for %s-%s'%(len(scene_data['imu_pose_matrix']), len(val_idxes_list), len(X_rect_list), date, drive))
    #     scene_data['intrinsics'] = kitti_two_frame_loader.K.copy()
    #     scene_data['img_l'] = [im[0] for im in kitti_two_frame_loader.dataset_rgb]

    #     if self.get_sift:
    #         scene_data['sift_kp'] = []
    #         scene_data['sift_des'] = []            
    #         for idx in range(kitti_two_frame_loader.N_frames):
    #         # for idx in range(1):
    #             logging.info('Getting sift for frame %d/%d.'%(idx, kitti_two_frame_loader.N_frames))
    #             kp, des = self.sift.detectAndCompute(np.array(kitti_two_frame_loader.dataset_rgb[idx][0]), None) ## IMPORTANT: normalize these points
    #             x_all = np.array([p.pt for p in kp])
    #             scene_data['sift_kp'].append(x_all)
    #             scene_data['sift_des'].append(des)

    #     if self.get_pose:
    #         velo2cam_mat = kitti_two_frame_loader.dataset.calib.T_cam0_velo_unrect
    #         imu2velo_mat = kitti_two_frame_loader.dataset.calib.T_velo_imu
    #         cam_2rect_mat = kitti_two_frame_loader.dataset.calib.R_rect_00
    #         imu2cam = kitti_two_frame_loader.Rtl_gt @ cam_2rect_mat @ velo2cam_mat @ imu2velo_mat
    #         scene_data['imu2cam'] = imu2cam.copy()

    #     del kitti_two_frame_loader

    #     return scene_data

    def scene_data_to_samples(self, scene_data):
        def construct_sample(scene_data, i, frame_id):
            sample = {"img":self.load_image(scene_data, i)[0], "id":frame_id}
            # sample = {"img": np.array(scene_data['img_l'][i]).copy(), "id":frame_id}

            # if self.get_depth:
            #     sample['depth'] = self.generate_depth_map(scene_data, i)
            if self.get_X:
                sample['X_rect_vis'] = scene_data['X_rect'][i][:, scene_data['val_idxes'][i]].copy()
            if self.get_pose:
                sample['imu_pose_matrix'] = scene_data['imu_pose_matrix'][i].copy()
            if self.get_sift:
                sample['sift_kp'] = scene_data['sift_kp'][i].copy()
                sample['sift_des'] = scene_data['sift_des'][i].copy()
            return sample

        all_imgs = []
        if self.from_speed:
            logging.info('+++ Using speed to filter frames. Min speed %d'%self.min_speed)
            cum_speed = np.zeros(3)
            for i, speed in enumerate(scene_data['speed']):
                cum_speed = speed
                speed_mag = np.linalg.norm(cum_speed)
                if speed_mag > self.min_speed:
                    frame_id = scene_data['frame_id'][i]
                    print(speed_mag, speed)
                    # yield construct_sample(scene_data, i, frame_id)
                    all_imgs.append(construct_sample(scene_data, i, frame_id))
                    cum_speed *= 0
        else:
            drive = str(scene_data['dir'].name)
            for (i,frame_id) in enumerate(scene_data['frame_id']):
                speed = np.linalg.norm(scene_data['speed'][i])
                if (drive not in self.static_frames.keys()) or (frame_id not in self.static_frames[drive]):
                    if speed < self.min_speed:
                        logging.error('Non static frame %s-%s with small speed %f<%f'%(drive, frame_id, speed, self.min_speed))
                    else:
                        # yield construct_sample(scene_data, i, frame_id)
                        all_imgs.append(construct_sample(scene_data, i, frame_id))
                else:
                    logging.warning('[%s-%s] Skipped in scene_data_to_samples because of static. Speed %f'%(drive, frame_id, speed))
                    pass

        return all_imgs

    def dump_drive(self, args, drive_path, scene_data=None):
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

        intrinsics = scene_data['intrinsics_ori']
        dump_cam_file = dump_dir/'cam'
        np.save(dump_cam_file+'.npy', intrinsics)
        
        dump_imu2cam_file = dump_dir/'imu2cam'
        np.save(dump_imu2cam_file, scene_data['imu2cam'])

        poses_file = dump_dir/'imu_pose_matrixs'
        poses = []

        logging.info('Dumping %d samples to %s...'%(len(scene_samples), dump_dir))
        for ii, sample in enumerate(scene_samples):
            # logging.info('Dumping %d/%d.'%(ii, len(scene_samples)))
            img, frame_nb = sample["img"], sample["id"]
            dump_img_file = dump_dir/'{}.jpg'.format(frame_nb)
            scipy.misc.imsave(dump_img_file, img)
            if "imu_pose_matrix" in sample.keys():
                poses.append(sample["imu_pose_matrix"].reshape(-1).tolist())
                if len(poses) != 0:
                    # np.savetxt(poses_file, np.array(poses).reshape(-1, 16), fmt='%.20e')a
                    if self.save_npy:
                        np.save(poses_file+'.npy', np.array(poses).reshape(-1, 16))
                    else:
                        saveh5({"pose": np.array(poses).reshape(-1, 16)}, poses_file+'.h5')
            if "X_rect_vis" in sample.keys():
                dump_X_file = dump_dir/'X_{}'.format(frame_nb)
                if self.save_npy:
                    np.save(dump_X_file+'.npy', sample["X_rect_vis"])
                else:
                    saveh5({"X_rect_vis": sample["X_rect_vis"]}, dump_X_file+'.h5')
            if "sift_kp" in sample.keys():
                dump_sift_file = dump_dir/'sift_{}'.format(frame_nb)
                if self.save_npy:
                    np.save(dump_sift_file+'.npy', np.hstack((sample['sift_kp'], sample['sift_des'])))
                else:
                    saveh5({'sift_kp': sample['sift_kp'], 'sift_des': sample['sift_des']}, dump_sift_file+'.h5')

        if self.get_sift:
            delta_ijs = [1, 2, 3, 5]
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
        return np.asarray(all_ij).T.copy(), np.asarray(good_ij).T.copy()
