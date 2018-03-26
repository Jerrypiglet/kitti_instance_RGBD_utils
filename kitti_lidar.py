# Written by Gengshan Yang

import itertools
import pdb
import pykitti  # install using pip install pykitti
import os
import numpy as np
from mayavi import mlab
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
fig = mlab.figure(bgcolor=(0, 0, 0), size=(1080, 720))
for i,velo in enumerate(dataset.velo):
    oxts_pose = next(iter(itertools.islice(dataset.oxts, i, None))).T_w_imu
    oxts_pose = oxts_pose.dot( np.linalg.inv(dataset.calib.T_velo_imu) )

    velo = np.hstack( (velo[:,:3],np.ones((velo.shape[0],1))) ).dot(oxts_pose.T)

    cen = np.vstack((cen, oxts_pose.dot([0,0,0,1])))
    mlab.points3d(cen[:,0],cen[:,1],cen[:,2],color=(1,0,0),scale_factor=0.5)

    

    ## registration
    #pcl_velo = pcl.PointCloud(velo[:,:3].astype(np.float32))
    #if not prev_velo == None:
    #    converged,transf,estimate,fitness = reg.icp_nl(prev_velo, pcl_velo)
    #    pcl_velo = estimate
    #prev_velo = pcl_velo
    #velo = pcl_velo.to_array()

    velo = filter_ground(velo,cen,th=1000)

    # draw annotated objects
    filled_idx = np.zeros((velo.shape[0],),dtype=bool)
    for j,box in enumerate(tracklet_rects[i]):
        box = oxts_pose.dot( np.vstack((box, np.ones((1,8)) )) )  # register boxes
        draw_class.draw_box(box, tracklet_ids[i][j])
        idx = in_hull(velo[:,:3],box[:3,:].T)
        draw_class.draw_cluster(velo[idx,:], tracklet_ids[i][j])
        filled_idx |= idx
    
    # print other points
    draw_class.draw_cluster(velo[~filled_idx,:])

    mlab.view(*config)
    mlab.savefig('./output/%s_%s/test_%03d.png'%(date,drive,i))
    print('./output/%s_%s/test_%03d.png'%(date,drive,i))
    mlab.clf()
mlab.show()
