# KITTI data dump and load demo
This dump(er) and loader deal with images, lidar points and SIFT for each frame, camera pose and SIFT correspondences between frames.
## Requirements
There are two repos:
[1] [Branch: **deepF**) This repo (https://github.com/eric-yyjau/deepSfm/tree/deep_F) for dumping and loading dataset.
Install dependeicies by
> pip install -r requirements.txt

[2] Another repo (https://github.com/Jerrypiglet/kitti_instance_RGBD_utils) for some tool functions. Add the PATH to this repo to envs by (change to your path):
> export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'

## Datasets
The [KITTI Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). It is already available at 
> descartes.ucsd.edu:/data/KITTI/odometry

## Peek into the dataset and simple single-frame loader
**kitti_instance_RGBD_utils/KITTI_5_RANSAC_sample_twoFrame_odo.ipynb**
This is a single frame loader from the **original** Dataset without the need to dump into format of sequence. You can 

- Visualize for single frame in section **# Verify my rectification**, 
- For two frames in section **# Get ij**, with two images overlaid with lidar points, and output of the relative **scene** pose (inverse of the actual camera motion).
- In **## Test OpenCV-5** you can visualize the SIFT keypoints and matches, and the results from OpenCV 5 point algorithm.

## Dump Data for sequential loading (e.g. training)
<!--### KITTI RAW dataset
> python dump_img_raw.py --dump --dataset_dir /data/kitti/raw --with_pose --with_X --with_sift --static_frames_file /home/ruizhu/Documents/Projects/SfmLearner-Pytorch/data/static_frames.txt --test_scene_file /home/ruizhu/Documents/Projects/SfmLearner-Pytorch/data/test_scenes.txt  --dump_root /home/ruizhu/Documents/Datasets/kitti/kitti_dump/corr_dump_siftIdx_npy_speed05_delta1235 --num_threads=1

Set ``--with_pose`` ``--with_X`` ``--with_sift`` to decide whether to dump pose files, rectified lidar points, and SIFT kps/des and corres.

Set the ``--static_frames_file`` and ``--test_scene_file`` to where your static frames file and test scene file is. You can acquire them from https://github.com/ClementPinard/SfmLearner-Pytorch.

By default, we use frames excluded from the ``static_frames_file`` and also with a speed of no more than 0.5m/s. Also we dump correspondences for each frame pairs of 1, 2, 3, or 5 frames away (e.g. frame i and frame i+{1, 2, 3, 5}) so that you can set ``--delta_ij`` to 1, 2, 3, 5 in sequential reading.-->

### KITTI Odometry dataset
**``WE ARE NOT FILTERING STATIC FRAMES FOR THE ODO DATASET!``**
(In **kitti_instance_RGBD_utils/kitti_tools/**:)
> python dump_img_odo.py --dump --dataset_dir /data/kitti/odometry --with_pose --with_X --with_sift --dump_root /home/ruizhu/Documents/Datasets/kitti/kitti_dump/odo_corr_dump_siftIdx_npy_delta1235810_full

Set ``--with_pose`` ``--with_X`` ``--with_sift`` to decide whether to dump pose files, rectified lidar points, and SIFT kps/des and corres.

We dump correspondences for each frame pairs of 1, 2, 3, 5, 8 or 10 frames away (e.g. frame i and frame i+{1, 2, 3, 5, 8, 10}) so that you can set ``--delta_ij`` to 1, 2, 3, 5, 8, 10 in sequential reading.

## A sequence loader
**kitti_instance_RGBD_utils/kitti_tools/kitti_seq_reader.ipynb**
Run:
- # [1] Necessary imports
- # [3] Sequence read - Odo KITTI

The first four sections (until ## Get two frames) lets you create a sequence loader (of length 2) and visualize basic info of images, overlaid with reprojected lidar points. 


<!--## Loader
Run the sections [1] and [3] in `kitti_seq_reader.ipynb` for a demo of sequential read. It will throw warnings if certain files are not found. Also you can run the second box in [3] for a visualization of the image, lidar points, and SIFT matches.
-->
