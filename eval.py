import os
import pdb
import cv2
import numpy as np

path_fold = 'output'
with open('./trash/filelist2') as f:
    set_fold = [i.split('/')[-1].strip() for i in f.readlines()]

pdb.set_trace()
res_all = np.zeros((0,2))
for fold in set_fold:
    path = '%s/%s.npy' % (path_fold,fold)
    res_all = np.vstack( (res_all, np.load(path) ))
    eval_res = np.load(path)
    u= np.sum(eval_res[:,0] < 0.5)/float(len(eval_res))
    o= np.sum(eval_res[:,1] < 1)/float(len(eval_res))
    print '%s:%f/%f' % (fold,u,o)


u= np.sum(res_all[:,0] < 0.5)/float(len(res_all))
o= np.sum(res_all[:,1] < 1)/float(len(res_all))
print 'all:%f/%f'%(u,o)
