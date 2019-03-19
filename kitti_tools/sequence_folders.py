import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import cv2
from utils_good import *

# import os,sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(ROOT_DIR)
# from kitti_tools.utils_opencv import *

def load_as_float(path):
    return np.array(imread(path)).astype(np.float32)

def load_as_array(path):
    return np.load(path)

class SequenceLoader(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=2, delta_ij=1, 
                 get_X=False,
                 get_pose=False,
                 get_sift=False, 
                 transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.sequence_length = sequence_length
        self.delta_ij = delta_ij
        self.get_X = get_X
        self.get_pose = get_pose
        self.get_sift = get_sift
        self.bf = cv2.BFMatcher()
        self.crawl_folders(self.sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        # demi_length = (sequence_length-1)//2
        # demi_length = sequence_length-1
        # shifts = list(range(-demi_length, demi_length + 1))
        # shifts.pop(demi_length)
        max_idx = (sequence_length-1)*self.delta_ij

        for scene in self.scenes:
            # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            intrinsics = load_as_array(scene/'cam.npy').astype(np.float32).reshape((3, 3))
            # imu_pose_matrixs = np.genfromtxt(scene/'imu_pose_matrixs.txt').astype(np.float64).reshape(-1, 4, 4)
            # imu_pose_matrixs = load_as_array(scene/'imu_pose_matrixs.npy').astype(np.float64).reshape(-1, 4, 4)
            imu_pose_matrixs = loadh5(scene/'imu_pose_matrixs.h5')['pose'].astype(np.float64).reshape(-1, 4, 4)
            self.imu2cam = load_as_array(scene/'imu2cam.npy')
            imgs = sorted(scene.files('*.jpg'))
            full_length = len(imgs)
            X_files = sorted(scene.files('*_X.h5'))
            sift_files = sorted(scene.files('*_sift.h5'))

            if full_length <= max_idx:
                logging.warning('Number of images in scene %s smaller than the seq length required!'%scene)
                print(full_length, [idx*self.delta_ij for idx in range(sequence_length)])
                continue
            if not(imu_pose_matrixs.shape[0]==len(imgs)==len(X_files)==len(sift_files)):
                logging.error('Unequal number of files in scene %s! imu_pose_matrixs.shape[0] %d, len(imgs) %d, len(X_files) %d, len(sift_files)%d'%\
                    (scene, imu_pose_matrixs.shape[0], len(imgs), len(X_files), len(sift_files)))

            # for i in range(demi_length, len(imgs)-demi_length):
            for i in range(full_length - max_idx):
                sample = {'intrinsics': intrinsics, 'imgs': [imgs[i]], 'scene_name': scene.name, 'frame_ids': [i]}
                if self.get_pose:
                    sample['scene_poses'] = [np.eye(3, dtype=np.float32)]
                    sample['imu_pose_matrixs'] = [imu_pose_matrixs[i]]
                if self.get_X:
                    sample['X_files'] = [X_files[i]]
                if self.get_sift:
                    sample.update({'sift_files': [sift_files[i]], 'scene_name': scene.name})

                for k in range(1, sequence_length):
                    j = k   * self.delta_ij
                    sample['imgs'].append(imgs[i+j])
                    if self.get_pose:
                        sample['imu_pose_matrixs'].append(imu_pose_matrixs[i+j])
                        sample['scene_poses'].append(self.imu2cam @ np.linalg.inv(imu_pose_matrixs[i+j]) @ imu_pose_matrixs[i] @ np.linalg.inv(self.imu2cam))
                        # if i==166:
                        #     print(i, scene, '\n', imu_pose_matrixs[i])
                        #     print(i+j, scene, '\n', imu_pose_matrixs[i+j])
                        #     print(self.imu2cam)
                        #     test = self.imu2cam @ np.linalg.inv(imu_pose_matrixs[i]) @ imu_pose_matrixs[i+j] @ np.linalg.inv(self.imu2cam)
                        #     print(test)
                    if self.get_X:
                        sample['X_files'].append(X_files[i+j]) # [3, N]
                    if self.get_sift:
                        sample['sift_files'].append(sift_files[i+j]) # [N, 256+2]
                    sample['frame_ids'].append(i+j)
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        # tgt_img = load_as_float(sample['tgt'])
        # ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        imgs = [load_as_float(img) for img in sample['imgs']]
        # print(imgs[0])

        # if self.transform is not None:
        #     imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
        #     tgt_img = imgs[0]
        #     ref_imgs = imgs[1:]
        # else:

        intrinsics = sample['intrinsics']
        # imu_pose_matrixs = sample['imu_pose_matrixs']
        scene_name = sample['scene_name']
        frame_ids = sample['frame_ids']

        # Xs = [load_as_array(X_file) for X_file in sample['X_files']] if self.get_X else [-1]*self.sequence_length
        Xs = [loadh5(X_file)['X_rect_vis'] for X_file in sample['X_files']] if self.get_X else [-1]*self.sequence_length

        if self.get_sift:
            # sift_arrays = [load_as_array(sift_file) for sift_file in sample['sift_files']]
            # sift_kps = [sift_array[:, :2] for sift_array in sift_arrays]
            # sift_deses = [sift_array[:, 2:] for sift_array in sift_arrays]
            sift_arrays = [loadh5(sift_file) for sift_file in sample['sift_files']]
            sift_kps = [sift_array['sift_kp'] for sift_array in sift_arrays]
            sift_deses = [sift_array['sift_des'] for sift_array in sift_arrays]
            

            ## Match on the fly: too slow (~4fps)
            # if self.sequence_length==2:
            #     # BFMatcher with default params
            #     bf = cv2.BFMatcher(normType=cv2.NORM_L2)
            #     matches = bf.knnMatch(np.asarray(sift_deses[0], np.float32), np.asarray(sift_deses[1], np.float32), k=2)
            #     # Apply ratio test
            #     good = []
            #     for m,n in matches:
            #         if m.distance < 0.75*n.distance:
            #             good.append([m])
            #     x1 = x1_all[[mat.queryIdx for mat in good], :]
            #     x2 = x2_all[[mat.trainIdx for mat in good], :]
            #     print('--', x1.shape, x2.shape)
        else:
            sift_kps = [-1]*self.sequence_length
            sift_deses = [-1]*self.sequence_length
        scene_poses = sample['scene_poses'] if self.get_pose else [-1]*self.sequence_length

        # print(Xs[0].shape)

        # print(frame_ids)
        # return imgs, intrinsics, imu_pose_matrixs, scene_name, frame_ids
        return imgs, intrinsics, scene_name, frame_ids, Xs, sift_kps, sift_deses, scene_poses

    def __len__(self):
        return len(self.samples)
