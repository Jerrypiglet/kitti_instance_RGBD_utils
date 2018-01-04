from mayavi import mlab
from numpy.linalg import norm
import pdb
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN


def filter_ground(velo, cen, th=20):
    idxs = velo[:,2] > (cen[-1][2] - 1.6)
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

def draw_cluster(velo, col):
    mlab.points3d(
        velo[:, 0],   # x
        velo[:, 1],   # y
        velo[:, 2],   # z
        mode="point", # How to render each point {'point', 'sphere' , 'cube' }
        color=col,     # Used a fixed (r,g,b) color instead of colormap
        scale_factor=100,
        line_width=10,
    )
