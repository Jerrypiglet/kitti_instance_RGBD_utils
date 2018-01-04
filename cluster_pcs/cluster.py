import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from numpy.linalg import norm
import pdb

class cluster_manager(object):
    def __init__(self):
        self.count = 0
        self.trks = []
        
    def build_dets(self, points, idxs):
        dets = []
        for i in np.unique(idxs):
            dets.append(cluster(points[idxs==i],self.count))
            self.count += 1
        return dets

    def associate_dets(self, dets):
        if(len(self.trks)==0):
            return np.empty((0,2),dtype=int), np.arange(len(dets)), np.empty((0,1),dtype=int)
        iou_matrix = np.zeros((len(dets),len(self.trks)),dtype=np.float32)
        for d,det in enumerate(dets):
            for t,trk in enumerate(self.trks):
                iou_matrix[d,t] = norm(det.center - trk.center)

        matched_indices = linear_assignment(iou_matrix)

        unmatched_detections = []
        for d,det in enumerate(dets):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t,trk in enumerate(self.trks):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if 0:# (iou_matrix[m[0],m[1]]>1000):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    
    def update(self,points, idxs):
        dets = self.build_dets(points, idxs)
        matched, unmatched_dets, unmatched_trks = self.associate_dets(dets)
        for t,trk in enumerate(self.trks):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:,1]==t)[0],0][0]
                trk.update(dets[d])
        for i in unmatched_dets:
            self.trks.append(dets[i])

       
         
        # for i in self.cluster:
            # match
            

class cluster(object):
    def __init__(self, points, cid):
        self.points = points[:,:3]
        self.id = cid
        self.build_features()

    def build_features(self):
        self.center = np.mean(self.points,0)

    def update(self,det):
        self.points = np.concatenate((self.points, det.points))
        self.build_features()
