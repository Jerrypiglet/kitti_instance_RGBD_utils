import sys
# sys.path[0] = '/home/rzhu/.conda/envs/mayavi/lib/python2.7/site-packages'
# sys.path.append('/home/rzhu/Documents/kitti-lidar-utils')
# print sys.path
import itertools
import pdb
import pykitti  # install using pip install pykitti
import os
import numpy as np
# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from mayavi import mlab
# mlab.init_notebook() # http://docs.enthought.com/mayavi/mayavi/tips.html
mlab.options.offscreen = True
from imayavi import *
import time
from source.utils import load_tracklets_for_frames, point_inside, in_hull
from source import parseTrackletXML as xmlParser
import argparse
from matplotlib import cm
from cluster_pcs.filters import *
from math import atan2, degrees
#import pcl

parser=argparse.ArgumentParser()
# parser.add_argument('--fdir',type=str,help='dir of format base/data/drive',default='/data/KITTI/2011_09_26/2011_09_26_drive_0005_sync')
parser.add_argument('--outdir',type=str,help='output dir',default='./output_reproj')
args = parser.parse_args()
# args = parser.parse_args(['--fdir', '/home/rzhu/Documents/kitti_dataset/raw/2011_09_26/2011_09_26_drive_0005_sync/', '--outdir', './output/'])

# Raw Data directory information
# path = args.fdir.rstrip('/')
basedir = '../kitti_dataset/raw'

# Prepare for drawings
fig_scale = 300
fig_ratio = [4, 3]
fig = mlab.figure(bgcolor=(0, 0, 0), size=(fig_ratio[0]*fig_scale, fig_ratio[1]*fig_scale))

plt_3d = plt.figure(1, figsize=(fig_ratio[0], fig_ratio[1]), dpi=fig_scale)
ax_3d = plt_3d.add_axes([0, 0, 1, 1])
ax_3d.axis('off')
plt.show(block=False)

# plt_rgb, ax_rgb = plt.subplots(2, 2, figsize=(24, 8))
# plt.show(block=False)

plt_proj, ax_proj = plt.subplots(2, 1, figsize=(20, 10))
plt.show(block=False)

cam=2
from depth_evaluation_utils import *
date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
for date in date_list:
    # calib_dir = '/home/rzhu/Documents/kitti_dataset/raw/%s/'%date
    # cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    # velo2cam = read_calib_file(calib_dir + 'calib_velo_to_cam.txt')
    # velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis])) # == dataset.calib.T_cam0_velo_unrect
    # velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    # # compute [projection matrix] velodyne->image plane
    # R_cam2rect = np.eye(4)
    # R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    # P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    # P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam) # 4*3

    P_rect = dataset.calib.P_rect_20 # Documentation see https://github.com/Jerrypiglet/pykitti/blob/master/pykitti/raw.py
    velo2cam = dataset.calib.T_cam2_velo
    P_velo2im = np.dot(P_rect, velo2cam) # 4*3

    drive_list = os.listdir(calib_dir)
    # print drive_list
    for drive_full_name in drive_list:
        drive = drive_full_name.split('_')[-2]
        print drive
        outdir = '%s/%s_%s' % (args.outdir,date,drive)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            print('---- Output Folder Made:', outdir)
        else:
            print('---- Output Folder Exists:', outdir)
        dataset = pykitti.raw(basedir, date, drive)
        tracklet_name = '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir,date, date, drive)
        if not(os.path.isfile(tracklet_name)):
            print('!!!! Tracklet Not Found at:', tracklet_name)
            continue
        tracklet_rects, tracklet_types, tracklet_ids = load_tracklets_for_frames(len(list(dataset.velo)), tracklet_name)
        dataset_gray = list(dataset.gray)
        dataset_rgb = list(dataset.rgb) 

        # view point
        # v1 = next(iter(itertools.islice(dataset.oxts, 0, None))).T_w_imu.dot([0,0,0,1])
        # v2 = next(iter(itertools.islice(dataset.oxts, 1, None))).T_w_imu.dot([0,0,0,1])
        # vec = (v1 - v2)[:2]
        # deg = degrees(atan2(vec[1],vec[0]))
        # config=(deg, 70.545295075710314, 56.913794133592624,[ 0.,   1.,   1.])

        cen=np.zeros((0,4))
        for frame_idx, velo in enumerate(dataset.velo):

            ## --- Plot 1: 3D
            mlab.clf()
            cen = np.vstack((cen, [0,0,0,1]))
            mlab.points3d(cen[:,0],cen[:,1],cen[:,2],color=(1,0,0),scale_factor=0.5)
            
            # Ground only
        #     velo = filter_ground(velo,cen,th=1000)
            # Front only
            velo = velo[velo[:, 0] >= 0, :]

            # draw annotated objects
            filled_idx = np.zeros((velo.shape[0],),dtype=bool)
            for j,box in enumerate(tracklet_rects[frame_idx]):
                draw_class.draw_box(box, tracklet_ids[frame_idx][j])
                idx = in_hull(velo[:,:3],box[:3,:].T)
                draw_class.draw_cluster(velo[idx,:], tracklet_ids[frame_idx][j])
                filled_idx |= idx
            
            # print other points
            draw_class.draw_cluster(velo[~filled_idx,:])

            mlab.view(azimuth=180, elevation=70, distance=50, focalpoint=[20., 0., 0.]) # tracking-view
            # mlab.view(azimuth=180, elevation=0, distance=70, roll=90, focalpoint=[0., 0., 0.]) # over-view
            img_3d = imayavi_return_inline(fig=fig)
            ax_3d.imshow(img_3d)
            plt_3d.canvas.draw()
            plt_3d.savefig('%s/3d_%03d.png'%(outdir,frame_idx))
            # plt.show(block=False)
            
            ## --- Plot 2: RGB
            # ax_rgb[0, 0].imshow(dataset_gray[frame_idx][0], cmap='gray')
            # ax_rgb[0, 0].set_title('Left Gray Image (cam0)')
            # ax_rgb[0, 1].imshow(dataset_gray[frame_idx][1], cmap='gray')
            # ax_rgb[0, 1].set_title('Right Gray Image (cam1)')
            # ax_rgb[1, 0].imshow(dataset_rgb[frame_idx][0])
            # ax_rgb[1, 0].set_title('Left RGB Image (cam2)')
            # ax_rgb[1, 1].imshow(dataset_rgb[frame_idx][1])
            # ax_rgb[1, 1].set_title('Right RGB Image (cam3)')
            # plt_rgb.canvas.draw()

            ## --- Plot 3: Projection
            vel_depth = True
            im_shape = [375, 1242]
            # project the points to the camera
            # Filter points inside bbox
            velo = velo[filled_idx,:]
            velo_pts_im = np.dot(P_velo2im, velo.T).T # [*, 3]
            velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,2][..., np.newaxis]

            if vel_depth: # Use velo first dimmension as depth
                velo_pts_im[:, 2] = velo[:, 0]

            # check if in bounds
            # use minus 1 to get the exact same value as KITTI matlab code
            velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
            velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
            val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
            val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
            velo_pts_im = velo_pts_im[val_inds, :]

            # project to image
            depth = np.zeros((im_shape))
            depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

            # find the duplicate points and choose the closest depth
            inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
            dupe_inds = [item for item, count in Counter(inds).iteritems() if count > 1]
            for dd in dupe_inds:
                pts = np.where(inds==dd)[0]
                x_loc = int(velo_pts_im[pts[0], 0])
                y_loc = int(velo_pts_im[pts[0], 1])
                depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
            depth[depth<0] = np.nan
            ax_proj[0].clear()
            ax_proj[0].imshow(dataset_rgb[frame_idx][0])
            ax_proj[0].scatter(velo_pts_im[:, 0].astype(np.int), velo_pts_im[:, 1].astype(np.int), s=5, c=velo_pts_im[:, 2])
            ax_proj[0].set_xlim([0, im_shape[1]-1])
            ax_proj[0].set_ylim([0, im_shape[0]-1])
            ax_proj[0].invert_yaxis()
            # ax_proj[0].set_title('Left RGB Image (cam2)')\
            ax_proj[0].axis('off')
            ax_proj[1].imshow(depth)
            ax_proj[1].axis('off')
            plt_proj.canvas.draw()
            plt_proj.savefig('%s/proj_%03d.png'%(outdir,frame_idx))
        break

