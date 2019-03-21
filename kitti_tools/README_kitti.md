# KITTI data dump and load demo
## Requirements
Install by
> pip install -r requirements.txt

## Dump Data
> python dump_img.py --dump --with_pose --with_X --with_sift --static_frames_file /home/ruizhu/Documents/Projects/SfmLearner-Pytorch/data/static_frames.txt --test_scene_file /home/ruizhu/Documents/Projects/SfmLearner-Pytorch/data/test_scenes.txt --dataset_dir /data/kitti/raw --dump_root /home/ruizhu/Documents/Datasets/kitti/kitti_dump/corr_dump_siftIdx_npy_speed05_delta1235 --num_threads=1

Set ``--with_pose`` ``--with_X`` ``--with_sift`` to decide whether to dump pose files, rectified lidar points, and SIFT kps/des and corres.

Set the ``--static_frames_file`` and ``--test_scene_file`` to where your static frames file and test scene file is. You can acquire them from https://github.com/ClementPinard/SfmLearner-Pytorch.

By default, we use frames excluded from the ``static_frames_file`` and also with a speed of no more than 0.5m/s. Also we dump correspondences for each frame pairs of 1, 2, 3, or 5 frames away (e.g. frame i and frame i+{1, 2, 3, 5}) so that you can set ``--delta_ij`` to 1, 2, 3, 5 in sequential reading.

## Loader
Run the sections [1] and [3] in `kitti_seq_reader.ipynb` for a demo of sequential read. It will throw warnings if certain files are not found. Also you can run the second box in [3] for a visualization of the image, lidar points, and SIFT matches.

