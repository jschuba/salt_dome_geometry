from multiprocessing import dummy

import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm

import Geometry3D as geo

from salt_dome_geometry import dome_settings

import cvxpy as cp

from salt_dome_geometry.main import *

geo.set_eps(1e-6)
dome_radius= 2500
dome_angle=70

print(f"Creating Domes and Caverns: {dome_radius=}, {dome_angle=}")
dome, dome_inner = make_cylinder_dome(geo.Point(0, 0, 5000), radius=dome_radius, height=-60000,
                                      angle_degrees=dome_angle,
                                      inner_offset=dome_settings.edge_of_salt_to_cavern)
dome_mask = make_top_bottom_mask(top_cutoff_depth=dome_settings.top_of_salt,
                                 bottom_cutoff_depth=dome_settings.bottom_of_salt, horizontal_size=60000,
                                 center_point=dome.center_point)

dome_cut = mask_shape(dome, dome_mask)

cavern_mask = make_top_bottom_mask(top_cutoff_depth=dome_settings.top_allowed_cavern,
                                   bottom_cutoff_depth=dome_settings.bottom_allowed_cavern, horizontal_size=60000,
                                   center_point=dome.center_point)
dome_inner_cut = mask_shape(dome_inner, cavern_mask)

projection = project_on_xy(dome_inner_cut)

best = 0
y_offset = 0
x_offset = 0
print(f"Gridding Caverns: {dome_radius=}, {dome_angle=}")
print(f"Gridding Caverns: {y_offset=}, {x_offset=} ")

hex_grid = make_hex_grid_of_circles_inside_polygon(projection, circle_radius=dome_settings.salt_cavern_diameter / 2,
                                                   spacing=dome_settings.inter_cavern_spacing,
                                                   y_offset=y_offset,
                                                   x_offset=x_offset)


fig, ax = plt.subplots()
ax.plot([pt.x for pt in projection.points], [pt.y for pt in projection.points], color='black')
for circ in hex_grid:
    ax.plot([pt.x for pt in circ.points], [pt.y for pt in circ.points], color='red')
fig.suptitle(f'Caverns: {len(hex_grid)}, {y_offset=}, {x_offset=} ')
plt.show()


best = len(hex_grid)
best_hex_grid = hex_grid

for y_offset in range(0, dome_settings.inter_cavern_spacing + dome_settings.salt_cavern_diameter, 100):
    for x_offset in range(0, dome_settings.inter_cavern_spacing + dome_settings.salt_cavern_diameter, 100):


        hex_grid = make_hex_grid_of_circles_inside_polygon(projection,
                                                           circle_radius=dome_settings.salt_cavern_diameter / 2,
                                                           spacing=dome_settings.inter_cavern_spacing,
                                                           y_offset=y_offset,
                                                           x_offset=x_offset)
        print(f"Gridding Caverns: {y_offset=}, {x_offset=}, {len(hex_grid)}")
        if len(hex_grid) >= best:
            fig, ax = plt.subplots()
            ax.plot([pt.x for pt in projection.points], [pt.y for pt in projection.points], color='black')
            for circ in hex_grid:
                ax.plot([pt.x for pt in circ.points], [pt.y for pt in circ.points], color='red')
            fig.suptitle(f'Caverns: {len(hex_grid)}, {y_offset=}, {x_offset=} ')
            plt.show()
            if len(hex_grid) > best:
                best = len(hex_grid)
                best_hex_grid = hex_grid
