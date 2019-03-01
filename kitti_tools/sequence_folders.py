import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random


def load_as_float(path):
    return np.array(imread(path)).astype(np.float32)

def load_as_array(path):
    return np.load(path)

class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=2, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        # demi_length = (sequence_length-1)//2
        demi_length = sequence_length-1
        # shifts = list(range(-demi_length, demi_length + 1))
        # shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imu_pose_matrixs = np.genfromtxt(scene/'imu_pose_matrixs.txt').astype(np.float64).reshape(-1, 4, 4)
            imgs = sorted(scene.files('*.jpg'))
            X_files = sorted(scene.files('*.npy'))
            if len(imgs) < sequence_length:
                continue
            # for i in range(demi_length, len(imgs)-demi_length):
            for i in range(0, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'imu_pose_matrixs': [imu_pose_matrixs[i]], 'imgs': [imgs[i]], 'Xs': [load_as_array(X_files[i])], 'scene_name': scene.name, 'frame_ids': [i]}
                # for j in shifts:
                for j in range(1, demi_length+1):
                    sample['imgs'].append(imgs[i+j])
                    sample['imu_pose_matrixs'].append(imu_pose_matrixs[i+j])
                    sample['Xs'].append(load_as_array(X_files[i])) # [3, N]
                    sample['frame_ids'].append(i+j)
                sequence_set.append(sample)
                # print(sample)
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

        intrinsics = np.copy(sample['intrinsics'])
        imu_pose_matrixs = sample['imu_pose_matrixs']
        scene_name = sample['scene_name']
        frame_ids = sample['frame_ids']
        # print(frame_ids)
        # return imgs, intrinsics, imu_pose_matrixs, scene_name, frame_ids
        return imgs, intrinsics, imu_pose_matrixs, scene_name, frame_ids

    def __len__(self):
        return len(self.samples)
