import time
import numpy as np
from mayavi import mlab
import wx

V = np.random.randn(20, 20, 20)
f = mlab.figure()
s = mlab.contour3d(V, contours=[0])

def animate_sleep(x):
    n_steps = int(x / 0.01)
    for i in range(n_steps):
        time.sleep(0.01)
        wx.Yield()

for i in range(5):

    animate_sleep(1)

    V = np.random.randn(20, 20, 20)

    # Update the plot with the new information
    s.mlab_source.set(scalars=V)
