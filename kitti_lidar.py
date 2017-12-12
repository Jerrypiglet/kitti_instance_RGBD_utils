"""
VISUALISE THE LIDAR DATA FROM THE KITTI DATASET

Based on the sample code from
    https://github.com/utiasSTARS/pykitti/blob/master/demos/demo_raw.py
And:
    http://stackoverflow.com/a/37863912

Contains two methods of visualizing lidar data interactively.
 - Matplotlib - very slow, and likely to crash, so only 1 out of every 100
                points are plotted.
              - Also, data looks VERY distorted due to auto scaling along
                each axis. (this could potentially be edited)
 - Mayavi     - Much faster, and looks nicer.
              - Preserves actual scale along each axes so items look
                recognizable
"""
import itertools
import pdb
import pykitti  # install using pip install pykitti
import os
import numpy as np
from mayavi import mlab
import time

class constant_camera_view(object):
    def __init__(self):
        pass

    def __enter__(self):
        self.orig_no_render = mlab.gcf().scene.disable_render
        if not self.orig_no_render:
            mlab.gcf().scene.disable_render = True
        cc = mlab.gcf().scene.camera
        self.orig_pos = cc.position
        self.orig_fp = cc.focal_point
        self.orig_view_angle = cc.view_angle
        self.orig_view_up = cc.view_up
        self.orig_clipping_range = cc.clipping_range

    def __exit__(self, t, val, trace):
        cc = mlab.gcf().scene.camera
        cc.position = self.orig_pos
        cc.focal_point = self.orig_fp
        cc.view_angle =  self.orig_view_angle 
        cc.view_up = self.orig_view_up
        cc.clipping_range = self.orig_clipping_range

        if not self.orig_no_render:
            mlab.gcf().scene.disable_render = False
        if t != None:
            print t, val, trace
            ipdb.post_mortem(trace)

from source import parseTrackletXML as xmlParser

def load_tracklets_for_frames(n_frames, xml_path):
    """
    Loads dataset labels also referred to as tracklets, saving them individually for each frame.

    Parameters
    ----------
    n_frames    : Number of frames in the dataset.
    xml_path    : Path to the tracklets XML.

    Returns
    -------
    Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
    contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
    types as strings.
    """
    tracklets = xmlParser.parseXML(xml_path)

    frame_tracklets = {}
    frame_tracklets_types = {}
    frame_tracklets_id = {}
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []
        frame_tracklets_id[i] = []
    # loop over tracklets
    for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        # import pdb; pdb.set_trace()
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet.__iter__():
            # determine if object is in the image; otherwise continue
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            frame_tracklets[absoluteFrameNumber] += [cornerPosInVelo]
            frame_tracklets_types[absoluteFrameNumber] += [tracklet.objectType]
            frame_tracklets_id[absoluteFrameNumber] += [i]

    return (frame_tracklets, frame_tracklets_types, frame_tracklets_id)


def draw_box(vertices, color):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    v = vertices[[0,1,2], :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for c in connections:
        mlab.plot3d(v[0,c],v[1,c],v[2,c],color=color)

def point_inside(rectangle, point):
    firstcorner, secondcorner = rectangle
    xmin, xmax = firstcorner[0]-1, secondcorner[0]+1
    yield xmin < point[0] < xmax
    ymin, ymax = firstcorner[1]-1, secondcorner[1]+1
    yield ymin < point[1] < ymax
    zmin, zmax = firstcorner[2]-1, secondcorner[2]+1
    yield zmin < point[2] < zmax


def in_hull(p,hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull=Delaunay(hull)

    return hull.find_simplex(p)>=0


# Raw Data directory information
basedir = '/data/KITTI/'
date = '2011_09_26'
drive = '0020'

dataset = pykitti.raw(basedir, date, drive)
tracklet_rects, tracklet_types, tracklet_ids = load_tracklets_for_frames(len(list(dataset.velo)),\
               '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir,date, date, drive))

config = (-160.72255236439889, 73.322705349553033, 47.036193498835345, [ 10.,   1.,   1.])
color_num = 32
colors = np.asarray(np.random.rand(color_num,3))
fig = mlab.figure(bgcolor=(0, 0, 0), size=(1080, 720))
#with constant_camera_view():
directory = './output/%s_%s' % (date,drive)
if not os.path.exists(directory):
    os.makedirs(directory)
for i,velo in enumerate(dataset.velo):
    #pdb.set_trace()
    mlab.clf()
    # mlab.axes(extent=[-20,80,-20,20,-3,10])
    #if i == 0:
    #    v = mlab.view()
    #else:
    #    mlab.view(*v)
    #pdb.set_trace()
    filled_idx = np.zeros((velo.shape[0],),dtype=bool)
    
    tracklet_types
    for j,box in enumerate(tracklet_rects[i]):
        col = tuple(colors[tracklet_ids[i][j]%color_num])
        draw_box(box, col)
        idx = in_hull(velo[:,:3],box.T)
        mlab.points3d(
            velo[idx, 0],   # x
            velo[idx, 1],   # y
            velo[idx, 2],   # z
            #i*np.ones(sum(idx),),
            #velo[idx, 3],   # Height data used for shading
            mode="point", # How to render each point {'point', 'sphere' , 'cube' }
            colormap='spectral',  # 'bone', 'copper',
            color=col,     # Used a fixed (r,g,b) color instead of colormap
            #color=(0,1,0),     # Used a fixed (r,g,b) color instead of colormap
            scale_factor=100,     # scale of the points
            line_width=10,        # Scale of the line, if any
            #figure=fig,
            #extent = [-20,80,-20,20,-3,10]
        )
        filled_idx |= idx
        
    mlab.points3d(
           velo[~filled_idx, 0],   # x
           velo[~filled_idx, 1],   # y
           velo[~filled_idx, 2],   # z
           #i*np.ones(sum(idx),),
           #velo[idx, 3],   # Height data used for shading
           mode="point", # How to render each point {'point', 'sphere' 
           colormap='spectral',  # 'bone', 'copper',
           color=(1,1,1),     # Used a fixed (r,g,b)
           #color=(0,1,0),     # Used a fixed (r,g,b) color instead of 
           scale_factor=100,     # scale of the points
           line_width=10,        # Scale of the line, if any
           #figure=fig,
           #extent = [-20,80,-20,20,-3,10]
    )
        

    mlab.view(*config)
    
    # velo[:, 3], # reflectance values
    mlab.savefig('./output/%s_%s/test_%03d.png'%(date,drive,i))
mlab.show(stop=True)
