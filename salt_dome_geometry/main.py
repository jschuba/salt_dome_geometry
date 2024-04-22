from multiprocessing import dummy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm


import Geometry3D as geo

geo.set_eps(1e-6)

def make_cylinder_dome(center_base: geo.Point, radius: float, height: float,
                       angle_degrees: float, inner_offset: float,
                       top_cutoff:float = -500, bottom_cutoff: float = -30000
                       ) -> (geo.Cylinder, geo.Cylinder):

    height_unit_vector = geo.Vector(np.cos(np.deg2rad(angle_degrees)), 0,  np.sin(np.deg2rad(angle_degrees)))

    dome = geo.Cylinder(center_base, radius, height*height_unit_vector, n=100)
    dome_inner = geo.Cylinder(center_base, radius-inner_offset, height*height_unit_vector, n=100)

    mask = geo.Parallelepiped(geo.Point(0, 0, top_cutoff), height*2*geo.x_unit_vector(), height*2*geo.y_unit_vector(), (bottom_cutoff-top_cutoff)*geo.z_unit_vector())
    mask.move(-height*2 / 2 * geo.x_unit_vector())
    mask.move(-height*2 / 2 * geo.y_unit_vector())

    dome_cut = dome.intersection(mask)
    dome_inner_cut = dome_inner.intersection(mask)

    return dome, dome_inner, mask, dome_cut, dome_inner_cut




dome, dome_inner, mask, dome_cut, dome_inner_cut = make_cylinder_dome(geo.Point(0, 0, 5000), radius=5000, height=-60000, angle_degrees=50, inner_offset=400)

r2 = geo.Renderer()
r2.add((dome,'b',1))
r2.add((dome_inner,'r',1))
r2.add((mask,'y',1))
r2.add((dome_cut,'k',1))
r2.add((dome_inner_cut,'k',1))
r2.show()


n = len(dome.convex_polygons)
color = iter(cm.rainbow(np.linspace(0, 1, n)))

r3 = geo.Renderer()
for pt in dome.convex_polygons[::5]:
    c = tuple(next(color))
    r3.add((pt,c,1))
r3.show()


