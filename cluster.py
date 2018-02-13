import itertools
import pcl
import pdb
import pykitti  # install using pip install pykitti
import os
import numpy as np
from mayavi import mlab
import time
from source.utils import load_tracklets_for_frames, in_hull
from source import parseTrackletXML as xmlParser
import argparse
from math import atan2, degrees
from cluster_pcs.filters import *
from cluster_pcs.cluster import cluster_manager

parser=argparse.ArgumentParser()
parser.add_argument('--fdir',type=str,help='dir of format base/data/drive',default='/data/KITTI/2011_09_26/2011_09_26_drive_0005_sync')
parser.add_argument('--outdir',type=str,help='output dir',default='/data/output')
parser.add_argument('--debug', help='', action='store_true')
args = parser.parse_args()


# color map
n_clusters = 20

# Raw Data directory information
path = args.fdir.rstrip('/')
basedir = path.rsplit('/',2)[0]
date = path.split('/')[-2]
drive = path.split('/')[-1].split('_')[-2]
outdir = '%s/%s_%s' % (args.outdir,date,drive)

dataset = pykitti.raw(basedir, date, drive)
tracklet_rects, tracklet_types, tracklet_ids = load_tracklets_for_frames(len(list(dataset.velo)),\
               '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir,date, date, drive))
eval_dict = {i:{} for i in np.unique([j for i in tracklet_ids.values() for j in i]) }


# view point
v1 = next(iter(itertools.islice(dataset.oxts, 0, None))).T_w_imu.dot([0,0,0,1])
v2 = next(iter(itertools.islice(dataset.oxts, 1, None))).T_w_imu.dot([0,0,0,1])
vec = (v1 - v2)[:2]
deg = degrees(atan2(vec[1],vec[0]))
config=(deg, 70.545295075710314, 56.913794133592624,[ 0.,   1.,   1.])

# begin drawing
if not os.path.exists(outdir):
    os.makedirs(outdir)

cen=np.zeros((0,4))
if args.debug:
    manager = cluster_manager(debug=args.debug)

for i,velo in enumerate(dataset.velo):
    if len(tracklet_rects[i])==0: continue
    #if i%30!=0:
    #    continue
    oxts_pose = next(iter(itertools.islice(dataset.oxts, i, None))).T_w_imu
    oxts_pose = oxts_pose.dot( np.linalg.inv(dataset.calib.T_velo_imu) )  # world-velo

    # register points to world coord
    velo = np.hstack( (velo[:,:3],np.ones((velo.shape[0],1))) ).dot(oxts_pose.T)

    # vox
    velo = pcl.PointCloud(velo[:,:3].astype(np.float32))
    sor = velo.make_voxel_grid_filter(); sor.set_leaf_size(0.1, 0.1, 0.1)
    velo = sor.filter().to_array()

    # register robot center
    cen = np.vstack((cen, oxts_pose.dot([0,0,0,1])))

    velo = filter_ground(velo,cen,th=15)

    labels = cluster_points(velo, n_clusters=n_clusters)
    
    if args.debug:
        manager.update(velo, labels)
    # evaluation
    for j,box in enumerate(tracklet_rects[i]):
        box = oxts_pose.dot( np.vstack((box, np.ones((1,8)) )) )  # register boxes
        idx = in_hull(velo[:,:3],box[:3,:].T)
        if sum(idx) == 0:
            continue
        lidx = np.unique(labels)
        intersect = [len(np.intersect1d(np.where(idx)[0], np.where(labels==k)))\
              for k in lidx]
        cid = np.argmax(intersect)
        metric_u = float(intersect[cid])/(np.sum(labels==lidx[cid]))
        metric_o = float(intersect[cid])/(np.sum(idx)) if np.sum(idx) else 0
        eval_dict[tracklet_ids[i][j]].update({i:(metric_u, metric_o)})
        #if metric_o < 1 and metric_o > 0.95:
        #    manager = cluster_manager(debug=True)
        #    manager.update(velo, labels)
        print 'most gt-%d poinst in cluster %d, u=%.2f, o=%.2f' % (tracklet_ids[i][j], lidx[cid], metric_u, metric_o)
        if args.debug:
            draw_class.draw_box(box, tracklet_ids[i][j], handle=manager.det_plot)
            # draw_class.draw_cluster(velo[idx,:], tracklet_ids[i][j])

    if args.debug:
        mlab.points3d(cen[:,0],cen[:,1],cen[:,2],color=(1,0,0),scale_factor=0.5,figure=manager.main_plot)
        mlab.view(*config,figure=manager.main_plot)
        mlab.text3d(0,0,-14,'frame %d'%i,figure=manager.main_plot,scale=0.6)
        mlab.savefig('./output/%s_%s/test_%03d.png'%(date,drive,i), figure=manager.det_plot)
    print 'frame-%d' % i


eval_res = np.asarray([j for i in eval_dict.values() for j in i.values()])
np.save('./output/%s_%s.npy'%(date,drive),eval_res)
np.sum(eval_res[:,0] < 0.5)/float(len(eval_res))
np.sum(eval_res[:,1] < 1)/float(len(eval_res))
