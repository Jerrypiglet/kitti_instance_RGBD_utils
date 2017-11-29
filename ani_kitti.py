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
#import wx
import time

def animate_sleep(x):
    n_steps = int(x / 0.01)
    for i in range(n_steps):
        time.sleep(0.01)
        wx.Yield()

# Raw Data directory information
basedir = '/home/gengshan/wnov/kitti/'
date = '2011_09_26'
drive = '0048'

# Optionally, specify the frame range to load
# since we are only visualizing one frame, we will restrict what we load
# Set to None to use all the data
# frame_range = range(10, 11, 1)

# Load the data
# dataset = pykitti.raw(basedir, date, drive, frames=frame_range)
dataset = pykitti.raw(basedir, date, drive)

# Plot only the ith frame (out of what has been loaded)
velo = next(itertools.islice(dataset.velo,0,1))

fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
plt = mlab.points3d(
    velo[:, 0],   # x
    velo[:, 1],   # y
    velo[:, 2],   # z
    velo[:, 2],   # Height data used for shading
    mode="point", # How to render each point {'point', 'sphere' , 'cube' }
    colormap='spectral',  # 'bone', 'copper',
    #color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
    scale_factor=100,     # scale of the points
    line_width=10,        # Scale of the line, if any
    figure=fig,
)

pdb.set_trace()
#for velo in dataset.velo:
#    # animate_sleep(1)
#    # time.sleep(1)
#    mlab.clf()
#    mlab.points3d(
#        velo[:, 0],   # x
#        velo[:, 1],   # y
#        velo[:, 2],   # z
#        velo[:, 2],   # Height data used for shading
#        mode="point", # How to render each point {'point', 'sphere' , 'cube' }
#        colormap='spectral',  # 'bone', 'copper',
#        #color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
#        scale_factor=100,     # scale of the points
#        line_width=10,        # Scale of the line, if any
#        figure=fig,
#    )
#    # velo[:, 3], # reflectance values
## mlab.show(stop=True)
    
@mlab.animate(delay=100)
def anim():
    f = mlab.gcf()
    for velo in dataset.velo:
        print('Updating scene...')
        #mlab.clf()
        #plt = mlab.points3d(
        #velo[:, 0],   # x
        #velo[:, 1],   # y
        #velo[:, 2],   # z
        #velo[:, 2],   # Height data used for shading
        #mode="point", # How to render each point {'point', 'sphere' , 'cube' }
        #colormap='spectral',  # 'bone', 'copper',
        ##color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
        #scale_factor=100,     # scale of the points
        #line_width=10,        # Scale of the line, if any
        #figure=fig,
        #)       
        
        plt.mlab_source.reset(x=velo[:, 0], y=velo[:, 1], z=velo[:, 2])
        yield


anim()
