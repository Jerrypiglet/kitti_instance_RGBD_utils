import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from numpy.linalg import norm
from filterpy.kalman import KalmanFilter
from mayavi import mlab
from cluster_pcs.filters import draw_class
import pdb
import pcl
from scipy.spatial.distance import cdist

class cluster_manager(object):
    def __init__(self, debug=False):
        self.count = 0
        self.trks = []
        self.debug = debug
        if self.debug:
            self.main_plot = mlab.figure(bgcolor=(0, 0, 0), size=(1080, 720))
            self.det_plot = mlab.figure(bgcolor=(0, 0, 0), size=(1080, 720))
        
    def build_dets(self, points, idxs):
        dets = []
        if self.debug: mlab.clf(self.det_plot)
        for i in np.unique(idxs):
            if i == -1:
                continue
            if np.sum(idxs==i)<50:
                continue
            new_clus = cluster(points[idxs==i], i)
            if self.debug:
                draw_class.draw_cluster(new_clus.points.to_array(),new_clus.id,self.det_plot)
                draw_class.draw_label(new_clus,self.det_plot)
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

        #det_table = [i.id for i in dets]
        #trk_table = [i.id for i in self.trks]
        
        det_t_r = np.where(np.sum(iou_matrix<2,1))[0]
        trk_t_r = np.where(np.sum(iou_matrix<2,0))[0]
        matched_indices = linear_assignment(iou_matrix[det_t_r][:,trk_t_r])
        matched_indices[:,0] = [det_t_r[i] for i in matched_indices[:,0]]
        matched_indices[:,1] = [trk_t_r[i] for i in matched_indices[:,1]]

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
            if (iou_matrix[m[0],m[1]]>2):  # cannot be larger than 2 meters
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
            if self.debug: draw_class.draw_point(trk,self.main_plot)

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

        if self.debug: 
            mlab.clf(self.main_plot)
            for i,trk in enumerate(self.trks):
                draw_class.draw_cluster(trk.points.to_array(),trk.id,self.main_plot)
                draw_class.draw_arrow(trk,self.main_plot)
                draw_class.draw_label(trk,self.main_plot)
            

class cluster(object):
    def __init__(self, points, cid):
        self.points = pcl.PointCloud(points[:,:3].astype(np.float32))
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
        self.center = np.mean(self.points.to_array(),0)
        if init:
            self.vel = 0
        else:
            self.vel = norm(self.kf.x[3:5].flatten())
            #print('cluster%d vel=%f'%(self.id,self.vel))
        # print('cluster %d: size=%d, std=%f'%(self.id, self.size, self.std))

    def update(self,det):
        if self.vel > 0.2:
            self.points = det.points
        else:
            self.points = np.concatenate((self.points.to_array(), det.points.to_array()))
            self.points = pcl.PointCloud(self.points[:,:3].astype(np.float32))
            # registration
            #self.points = pcl.PointCloud(self.points[:,:3].astype(np.float32))
            #det.points = pcl.PointCloud(det.points[:,:3].astype(np.float32))
            #icp = det.points.make_IterativeClosestPoint()
            #converged, transf, estimate, fitness = icp.icp(det.points, self.points)
            #self.points = np.vstack((estimate.to_array(), self.points.to_array()))
    

        # voxelize
        sor = self.points.make_voxel_grid_filter(); sor.set_leaf_size(0.1, 0.1, 0.1)
        self.points = sor.filter()

        self.kf.update(np.expand_dims(np.mean(self.points.to_array(),0),1))
        self.build_features()
