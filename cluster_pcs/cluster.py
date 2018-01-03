import numpy as np
import pdb

class cluster_manager(object):
    def __init__(self):
        self.count = 0
        self.cluster = []
            
    def update(self,points, idxs):
        in_clusters = []
        for i in np.unique(idxs):
            in_clusters.append(cluster(points[idxs==i],self.count))
            self.count += 1
       
         
        # for i in self.cluster:
            # match
            

class cluster(object):
    def __init__(self, points,cid):
        self.points = points
        self.id = cid
