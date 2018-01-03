import numpy
from mayavi.mlab import *

def test_points3d():
    t = numpy.linspace(0, 4 * numpy.pi, 20)
    cos = numpy.cos
    sin = numpy.sin

    x = sin(2 * t)
    y = cos(t)
    z = cos(2 * t)
    s = 2 + sin(t)

    return points3d(x, y, z, s, colormap="copper", scale_factor=.25)

import pdb
pdb.set_trace()
test_points3d()
