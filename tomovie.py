import os
import pdb
from moviepy.editor import ImageSequenceClip
import cv2


with open('./filelist2') as f:
    set_fold = [i.split('/')[-1].strip() for i in f.readlines()]
for fold in set_fold:
    path = '/data/output/%s/' % fold
    frames = []
    files = sorted(os.listdir(path))
    for f in files:
        frames.append( cv2.imread(path+f) )

    clip = ImageSequenceClip(frames, fps=5)
    clip.write_gif('/data/output/%s.gif'%fold, fps=5)
