# Written by Gengshan Yang

import itertools
import pdb
import pykitti  # install using pip install pykitti
import os
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
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
parser.add_argument('--fdir',type=str,help='dir of format base/data/drive',default='/data/KITTI/2011_09_26/2011_09_26_drive_0005_sync')
parser.add_argument('--outdir',type=str,help='output dir',default='/data/output')
args = parser.parse_args()

# Raw Data directory information
path = args.fdir.rstrip('/')
basedir = path.rsplit('/',2)[0]
date = path.split('/')[-2]
drive = path.split('/')[-1].split('_')[-2]
outdir = '%s/%s_%s' % (args.outdir,date,drive)

dataset = pykitti.raw(basedir, date, drive)
tracklet_rects, tracklet_types, tracklet_ids = load_tracklets_for_frames(len(list(dataset.velo)),\
               '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir,date, date, drive))
dataset_gray = list(dataset.gray)
dataset_rgb = list(dataset.rgb) 

# view point
v1 = next(iter(itertools.islice(dataset.oxts, 0, None))).T_w_imu.dot([0,0,0,1])
v2 = next(iter(itertools.islice(dataset.oxts, 1, None))).T_w_imu.dot([0,0,0,1])
vec = (v1 - v2)[:2]
deg = degrees(atan2(vec[1],vec[0]))
config=(deg, 70.545295075710314, 56.913794133592624,[ 0.,   1.,   1.])

# begin drawing
if not os.path.exists(outdir):
    os.makedirs(outdir)
    print('======Made:', outdir)
else:
	print('======Exists:', outdir)


cen=np.zeros((0,4))
#prev_velo = None
#reg = pcl.IterativeClosestPointNonLinear()
fig_scale = 300
fig_ratio = [4, 3]
plt_fig = plt.figure(1, figsize=(fig_ratio[0], fig_ratio[1]), dpi=fig_scale)
ax_fig = plt_fig.add_axes([0, 0, 1, 1])
ax_fig.axis('off')
plt.show(block=False)
fig = mlab.figure(bgcolor=(0, 0, 0), size=(fig_ratio[0]*fig_scale, fig_ratio[1]*fig_scale))
for i,velo in enumerate(dataset.velo):
    mlab.clf()
    cen = np.vstack((cen, [0,0,0,1]))
    mlab.points3d(cen[:,0],cen[:,1],cen[:,2],color=(1,0,0),scale_factor=0.5)

    velo = filter_ground(velo,cen,th=1000)

    # draw annotated objects
    filled_idx = np.zeros((velo.shape[0],),dtype=bool)
    for j,box in enumerate(tracklet_rects[i]):
        draw_class.draw_box(box, tracklet_ids[i][j])
        idx = in_hull(velo[:,:3],box[:3,:].T)
        draw_class.draw_cluster(velo[idx,:], tracklet_ids[i][j])
        filled_idx |= idx
    
    # print other points
    draw_class.draw_cluster(velo[~filled_idx,:])

    mlab.view(azimuth=180, elevation=0, distance=30, focalpoint=[0., 0., 0.]) # tracking-view
    # mlab.view(azimuth=180, elevation=0, distance=70, roll=90, focalpoint=[0., 0., 0.]) # over-view

    plt_img = imayavi_return_inline(fig=fig)
    plt.imshow(plt_img)
    plt_fig.canvas.draw()

    mlab.savefig('./output/%s_%s/test_%03d.png'%(date,drive,i))
    print('./output/%s_%s/test_%03d.png'%(date,drive,i), plt_img.shape)


