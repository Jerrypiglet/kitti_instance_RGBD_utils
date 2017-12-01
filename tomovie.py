import os
import pdb
from moviepy.editor import ImageSequenceClip
import cv2

set_fold = ['2011_09_26_0005','2011_09_26_0020','2011_09_26_0048','2011_09_26_0052']
for fold in set_fold:
    path = '/home/gengshan/wnov/output/%s/' % fold
    frames = []
    files = sorted(os.listdir(path))
    for f in files:
        frames.append( cv2.imread(path+f) )

    clip = ImageSequenceClip(frames, fps=5)
    clip.write_gif('%s.gif'%fold, fps=5)
