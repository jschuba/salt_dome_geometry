
from multiprocessing import dummy

import math

import numpy as np
import matplotlib.pyplot as plt
import shapely
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm
import matplotlib.colors
import matplotlib.cm
import datetime

import pandas as pd

import Geometry3D as geo
import geopandas as gpd
from scipy.spatial import ConvexHull

from salt_dome_geometry import dome_settings
from salt_dome_geometry.main import *

geo.set_eps(1e-6)



dome_cavern_intersections = pd.read_pickle("real_domes_and_caverns_250209_210243.pkl")


def plot_3d(df, show=True, save_as=None, figsize=(10, 10)):
    # Plot each contour in a different color, and prove that the linestrings are starting in the right places
    cmap = matplotlib.colormaps.get_cmap('summer')
    depth_col = 'struct_ft'
    color_list = cmap((df[depth_col] - df[depth_col].min()) /
                      (df[depth_col].max() - df[depth_col].min()))[::-1]
    for i in range(len(color_list)):
        color_list[i][-1] = 0.5  # Set the color transparency

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal', adjustable='box')
    for i, (row_index, row) in enumerate(df.iterrows()):
        ls = row.geometry
        if not ls:
            continue
        depth = row[depth_col]
        # Plot the points of the contour
        ax.scatter(ls.xy[0], ls.xy[1], [depth for _ in ls.xy[0]],
                   color=color_list[i],
                   label=depth)
        # # Plot the contour as a line
        ax.plot(list(ls.xy[0]) + [ls.xy[0][0]], list(ls.xy[1]) + [ls.xy[1][0]],
                [depth for _ in list(ls.xy[1]) + [ls.xy[1][0]]],
                color=color_list[i],
                label=depth)

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(row['domeName'])
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    return fig, ax


for name, grp in dome_cavern_intersections.groupby('domeName'):
    print(name)
    df = grp[grp['cavern_number'] == 0]
    fig, ax = plot_3d(df, show=False)

    cmap = matplotlib.colormaps.get_cmap('tab10')
    color_list = cmap(np.linspace(0, 1, grp['cavern_number'].max() + 1))[::-1]
    for i in range(len(color_list)):
        color_list[i][-1] = 0.5  # Set the color transparency

    for i, (cavern, cavern_grp) in enumerate(grp.groupby('cavern_number')):
        cavern_grp = cavern_grp[cavern_grp['take'] > 0]
        for row_i, (row_index, row) in enumerate(cavern_grp.iterrows()):
            ls = row['intersection'].boundary
            if not ls:
                continue
            depth = row['struct_ft']

            ax.scatter(ls.xy[0], ls.xy[1], [depth for _ in ls.xy[0]],
                       color=color_list[i],
                       label=depth)
            # # Plot the contour as a line
            ax.plot(list(ls.xy[0]) + [ls.xy[0][0]], list(ls.xy[1]) + [ls.xy[1][0]],
                    [depth for _ in list(ls.xy[1]) + [ls.xy[1][0]]],
                    color=color_list[i],
                    label=depth)


    # plt.show()
    plt.savefig(f"{name} 3d with caverns.png")
