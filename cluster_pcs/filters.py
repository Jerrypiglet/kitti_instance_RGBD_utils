from mayavi import mlab
from numpy.linalg import norm
import pdb
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN
from tvtk.tools import visual
from traits.api import on_trait_change
import numpy as np


def filter_ground(velo, cen, th=20):
    idxs = velo[:,2] > (cen[-1][2] - 1.2)
    idxs &= norm(velo[:,:2] - cen[-1,:2], axis=1) < th
    # draw_cluster(velo[idxs], (1,1,1))
    return velo[idxs]


def cluster_points(X, n_clusters=2):
    X = X[:,:3]
    # y_pred = KMeans(n_clusters=n_clusters).fit_predict(X)

    #bandwidth = estimate_bandwidth(X, quantile=0.05)
    #ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #ms.fit(X)
    #y_pred = ms.labels_

    db = DBSCAN(eps=0.8).fit(X)
    y_pred = db.labels_
    return y_pred

class draw_class(object):
    from matplotlib import cm
    color_num = 20
    cmap = cm.get_cmap('tab20')
    colors = cmap(range(20))[:,:3]

    @staticmethod
    @on_trait_change('n_meridional,n_longitudinal,scene.activated')
    def draw_cluster(trk,handle):
        velo = trk.points
        col = tuple(draw_class.colors[trk.id%draw_class.color_num])
            
        mlab.points3d(
            velo[:, 0],   # x
            velo[:, 1],   # y
            velo[:, 2],   # z
            mode="point", # How to render each point {'point', 'sphere' , 'cube' }
            color=col,     # Used a fixed (r,g,b) color instead of colormap
            scale_factor=100,
            line_width=10,
            figure=handle
        )

        
        draw_class.draw_arrow(trk,handle)


    @staticmethod
    def draw_point(trk,handle):
        p = trk.kf.x[:3].flatten()
        col = tuple(draw_class.colors[trk.id%draw_class.color_num])
        mlab.points3d(p[0],p[1],p[2]+2,color=col,scale_factor=0.3,figure=handle)


    @staticmethod
    def draw_label(trk,handle):
        p = trk.center
        mlab.text3d(p[0],p[1],p[2]+2,str(trk.id),scale=0.5,figure=handle)

    
    @staticmethod
    def draw_arrow(trk, handle):
        add = trk.kf.x[3:6].flatten()
        if sum(abs(add)) == 0:
            add += 0.001
        col = tuple(draw_class.colors[trk.id%draw_class.color_num])

        x,y,z = trk.center; u,v,w = add
        mlab.quiver3d(x,y,z+2,u,v,w,color=col,line_width=3., scale_factor=1., scale_mode='vector',figure=handle)
