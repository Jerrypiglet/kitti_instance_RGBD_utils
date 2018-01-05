import os
import pdb
from moviepy.editor import ImageSequenceClip
import cv2


#with open('./filelist2') as f:
#    set_fold = [i.split('/')[-1].strip() for i in f.readlines()]
set_fold = ['2011_09_26_0001','2011_09_26_0002','2011_09_26_0005']

path_fold = '/media/gengshay/My Passport/cluster_output'

for fold in set_fold:
    path = '%s/%s/' % (path_fold,fold)
    frames = []
    files = sorted(os.listdir(path))
    for f in files:
        frames.append( cv2.imread(path+f) )

    clip = ImageSequenceClip(frames, fps=5)
    clip.write_gif('%s/%s.gif'%(path_fold,fold), fps=5)
