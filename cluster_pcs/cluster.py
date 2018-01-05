import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from numpy.linalg import norm
from filterpy.kalman import KalmanFilter
from mayavi import mlab
from cluster_pcs.filters import draw_class
import pdb
from scipy.spatial.distance import cdist

class cluster_manager(object):
    def __init__(self):
        self.count = 0
        self.trks = []
        self.main_plot = mlab.figure(bgcolor=(0, 0, 0), size=(1080, 720))
        #self.det_plot = mlab.figure(bgcolor=(0, 0, 0), size=(1080, 720))
        
    def build_dets(self, points, idxs):
        dets = []
        #mlab.clf(self.det_plot)
        for i in np.unique(idxs):
            if i == -1:
                continue
            new_clus = cluster(points[idxs==i],self.count)
            #if new_clus.size < 20:
            #    continue
            #draw_class.draw_cluster(new_clus,self.det_plot)
            #draw_class.draw_label(new_clus,self.det_plot)
            dets.append(new_clus)
            self.count += 1
        return dets

    def associate_dets(self, dets):
        if(len(self.trks)==0):
            return np.empty((0,2),dtype=int), np.arange(len(dets)), np.empty((0,1),dtype=int)
        iou_matrix = np.zeros((len(dets),len(self.trks)),dtype=np.float32)
        for d,det in enumerate(dets):
            for t,trk in enumerate(self.trks):
                iou_matrix[d,t] = norm(det.center - trk.kf.x[:3].flatten())

        det_table = [i.id for i in dets]
        trk_table = [i.id for i in self.trks]
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
            if (iou_matrix[m[0],m[1]]>5):  # cannot be larger than 5 meters
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    
    def predict(self):
        for trk in self.trks:
            trk.kf.predict()
            draw_class.draw_point(trk,self.main_plot)

    def update(self,points, idxs):
        dets = self.build_dets(points, idxs)
        self.predict()
        matched, unmatched_dets, unmatched_trks = self.associate_dets(dets)
        for t,trk in enumerate(self.trks):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:,1]==t)[0],0][0]
                trk.update(dets[d])
        for i in unmatched_dets:
            self.trks.append(dets[i])

        mlab.clf(self.main_plot)
        for i,trk in enumerate(self.trks):
            draw_class.draw_cluster(trk,self.main_plot)
            draw_class.draw_label(trk,self.main_plot)
            

class cluster(object):
    def __init__(self, points, cid):
        self.points = points[:,:3]
        self.id = cid
        self.build_features(init=True)
        self.init_kf()


    def init_kf(self,momt=0.1):
        # constant accel model
        self.kf = KalmanFilter(dim_x=6 , dim_z=3)
        self.kf.F = np.array([[1,0,0,momt,0,0],[0,1,0,0,momt,0],[0,0,1,0,0,momt],\
                              [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
        #self.kf = KalmanFilter(dim_x=9 , dim_z=3)
        #self.kf.F = np.array([[1,0,0,momt,0,0,0.5*momt*momt,0,0],[0,1,0,0,momt,0,0,0.5*momt*momt,0],[0,0,1,0,0,momt,0,0,0.5*momt*momt],\
        #                      [0,0,0,1,0,0,momt,0,0],[0,0,0,0,1,0,0,momt,0],[0,0,0,0,0,1,0,0,momt],\
        #                      [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1]])
        #self.kf.H = np.array([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]])
        self.kf.x[:3] = np.expand_dims(self.center,1)
        
        

    def build_features(self,init=False):
        self.center = np.mean(self.points,0)
        self.size = len(self.points)
        self.std = np.mean(cdist(self.points,np.expand_dims(self.center,0),'euclidean'))
        if init:
            self.vel = 0
        else:
            self.vel = norm(self.kf.x[3:5].flatten())
            #print('cluster%d vel=%f'%(self.id,self.vel))
        # print('cluster %d: size=%d, std=%f'%(self.id, self.size, self.std))

    def update(self,det):
        if self.vel > 0.5:
            self.points = det.points
        else:
            self.points = np.concatenate((self.points, det.points))
            
        self.kf.update(np.expand_dims(np.mean(self.points,0),1))
        self.build_features()
