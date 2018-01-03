import itertools
import pdb
import pykitti  # install using pip install pykitti
import os
import numpy as np
from mayavi import mlab
import time
from source.utils import load_tracklets_for_frames, point_inside, in_hull, draw
from source import parseTrackletXML as xmlParser
import argparse
from matplotlib import cm
from math import atan2, degrees

parser=argparse.ArgumentParser()
parser.add_argument('--fdir',type=str,help='dir of format base/data/drive',default='/data/KITTI/2011_09_26/2011_09_26_drive_0005_sync')
parser.add_argument('--outdir',type=str,help='output dir',default='/data/output')
args = parser.parse_args()


# color map
color_num = 20
cmap = cm.get_cmap('tab20')
colors = cmap(range(20))[:,:3]

# Raw Data directory information
path = args.fdir.rstrip('/')
basedir = path.rsplit('/',2)[0]
date = path.split('/')[-2]
drive = path.split('/')[-1].split('_')[-2]
outdir = '%s/%s_%s' % (args.outdir,date,drive)

dataset = pykitti.raw(basedir, date, drive)
tracklet_rects, tracklet_types, tracklet_ids = load_tracklets_for_frames(len(list(dataset.velo)),\
               '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir,date, date, drive))

# view point
v1 = next(iter(itertools.islice(dataset.oxts, 0, None))).T_w_imu.dot([0,0,0,1])
v2 = next(iter(itertools.islice(dataset.oxts, 1, None))).T_w_imu.dot([0,0,0,1])
vec = (v1 - v2)[:2]
deg = degrees(atan2(vec[1],vec[0]))
config=(deg, 70.545295075710314, 56.913794133592624,[ 0.,   1.,   1.])

# begin drawing
if not os.path.exists(outdir):
    os.makedirs(outdir)

ori=np.zeros((0,4))
fig = mlab.figure(bgcolor=(0, 0, 0), size=(1080, 720))
for i,velo in enumerate(dataset.velo):
    if i%30!=0:
        continue
    # mlab.clf()
    oxts_pose = next(iter(itertools.islice(dataset.oxts, i, None))).T_w_imu
    oxts_pose = oxts_pose.dot( np.linalg.inv(dataset.calib.T_velo_imu) )
    velo = np.hstack( (velo[:,:3],np.ones((velo.shape[0],1))) )
    velo = np.asarray([oxts_pose.dot(point_imu) for point_imu in velo])
    ori = np.vstack((ori, oxts_pose.dot([0,0,0,1])))
    mlab.points3d(ori[:,0],ori[:,1],ori[:,2],color=(1,0,0),scale_factor=0.5)
    filled_idx = np.zeros((velo.shape[0],),dtype=bool)
    
    # print bbox objects
    for j,box in enumerate(tracklet_rects[i]):
        #if not tracklet_ids[i][j] == 0:
        #    continue
        box = np.vstack((box, np.ones((1,8)))).T
        box = np.asarray([oxts_pose.dot(b) for b in box]).T
        col = tuple(colors[tracklet_ids[i][j]%color_num])
        for c in draw.connections:
            mlab.plot3d(box[0,c],box[1,c],box[2,c],color=col)
        idx = in_hull(velo[:,:3],box[:3,:].T)
        mlab.points3d(
            velo[idx, 0],   # x
            velo[idx, 1],   # y
            velo[idx, 2],   # z
            mode="point", # How to render each point {'point', 'sphere' , 'cube' }
            color=col,     # Used a fixed (r,g,b) color instead of colormap
            scale_factor=100,
            line_width=10,
        )
        filled_idx |= idx
    
    # print other points
    mlab.points3d(
           velo[~filled_idx, 0],   # x
           velo[~filled_idx, 1],   # y
           velo[~filled_idx, 2],   # z
           mode="point", # How to render each point {'point', 'sphere' 
           color=(1,1,1),     # Used a fixed (r,g,b)
           scale_factor=100,     # scale of the points
           line_width=10,        # Scale of the line, if any
    )
    mlab.view(*config)
    # mlab.savefig('./output/%s_%s/test_%03d.png'%(date,drive,i))
mlab.show()
