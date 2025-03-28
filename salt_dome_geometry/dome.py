import datetime
import math

import numpy as np

import pandas as pd
import geopandas as gpd
import shapely

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm
import matplotlib.colors
import matplotlib.cm

from scipy.spatial import ConvexHull

from salt_dome_geometry import dome_settings
from salt_dome_geometry.shapely_helpers import *


class SaltDome:

    def __init__(self, dome_name: str, contours: gpd.GeoDataFrame | pd.DataFrame, depth_col: str, geometry_col: str, ):
        self.dome_name = dome_name

        self.contours = gpd.GeoDataFrame(contours, geometry=contours[geometry_col])

        self.depth_col = depth_col
        self.geometry_col = geometry_col

        self.original_contours = self.contours.copy()

    def move_to_origin(self):
        x_avg = self.contours.geometry.get_coordinates()['x'].mean()
        y_avg = self.contours.geometry.get_coordinates()['y'].mean()
        self.contours.geometry = self.contours.geometry.translate(-x_avg, -y_avg)

    def resample_contours(self, n=100):
        self.contours.geometry = self.contours.geometry.apply(resample_linestring, n=n)

    def resample_depths(self, interval=400, force_keep_original=False):
        df = self.original_contours.copy()
        df.geometry = df.geometry.apply(resample_linestring, n=100)
        depth_col = self.depth_col
        geo_col = self.geometry_col

        max_depth, min_depth = df[depth_col].max(), df[depth_col].min()
        new_depths = np.arange(min_depth, max_depth + interval, interval)

        new_contours = []
        for depth in new_depths:
            if depth > max_depth:
                break
            if depth in df[depth_col].unique():
                new_row = df.loc[df[depth_col] == depth, [depth_col, geo_col]].copy()
                new_row['interpolated'] = False
                new_contours.append(new_row)
                continue
            up_row_idx = df[df[depth_col] > depth][depth_col].idxmin()
            down_row_idx = df[df[depth_col] < depth][depth_col].idxmax()
            up_depth, down_depth = df[depth_col][up_row_idx], df[depth_col][down_row_idx]
            up_geo, down_geo = df[geo_col][up_row_idx], df[geo_col][down_row_idx]
            crs = df.geometry.crs
            percent = (depth - down_depth) / (up_depth - down_depth)
            new_geo = interpolate_between_linestrings(down_geo, up_geo, percent)
            new_row = [{depth_col: depth, geo_col: new_geo, 'interpolated': True}]
            new_contours.append(gpd.GeoDataFrame(new_row, crs=crs, geometry=geo_col))
        new_contours = pd.concat(new_contours)

        if force_keep_original:
            old_contours = df[[depth_col, geo_col]].copy()
            old_contours['interpolated'] = False
            new_contours = pd.concat([old_contours, new_contours])

        new_contours = (new_contours.drop_duplicates(subset=[depth_col], keep='first')
                        .sort_values(depth_col)
                        .reset_index(drop=True))
        new_contours['interpolated'] = new_contours['interpolated'].fillna(False)
        self.contours = new_contours

    def plot_contours(self, show=True, save_as=None):
        df = self.contours
        # Plot each contour in a different color, and prove that the linestrings are starting in the right places
        cmap = matplotlib.colormaps.get_cmap('summer')
        color_list = cmap((df[self.depth_col] - df[self.depth_col].min()) /
                          (df[self.depth_col].max() - df[self.depth_col].min()))[::-1]
        for i in range(len(color_list)):
            color_list[i][-1] = 0.5  # Set the color transparency

        fig, ax = plt.subplots(figsize=(20, 20))
        for i, (row_index, row) in enumerate(df.iterrows()):
            ls = row.geometry
            # Plot the points of the contour
            plt.scatter(ls.xy[0], ls.xy[1],
                        color=color_list[i],
                        label=row[self.depth_col],
                        s=40)
            # Plot the contour as a line
            ax.plot(list(ls.xy[0]) + [ls.xy[0][0]], list(ls.xy[1]) + [ls.xy[1][0]],
                    color=color_list[i],
                    label=row[self.depth_col])
            # Plot the first point in the contour in black
            plt.scatter(ls.xy[0][0], ls.xy[1][0], c='k')
        ax = plt.gca()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(self.dome_name)
        plt.axis('scaled')  # same x and y scaling
        plt.tight_layout()
        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()
        return fig, ax

    def plot_3d(self, show=True, save_as=None, figsize=(10, 10)):
        df = self.contours
        # Plot each contour in a different color, and prove that the linestrings are starting in the right places
        cmap = matplotlib.colormaps.get_cmap('summer')
        color_list = cmap((df[self.depth_col] - df[self.depth_col].min()) /
                          (df[self.depth_col].max() - df[self.depth_col].min()))[::-1]
        for i in range(len(color_list)):
            color_list[i][-1] = 0.5  # Set the color transparency

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('equal', adjustable='box')
        for i, (row_index, row) in enumerate(df.iterrows()):
            ls = row.geometry
            depth = row[self.depth_col]
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
        plt.title(self.dome_name)
        plt.tight_layout()
        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()
        return fig, ax

    def __repr__(self):
        return f'SaltDome(dome_name={self.dome_name})'


if __name__ == '__main__':

    all_domes = gpd.read_file(
        r"C:\Users\jschu\PycharmProjects\salt_dome_geometry\salt_dome_geometry\dataverse_files\gis\txSaltDiapStruct_v01.shp")

    # Get only domes of interest
    dome_names = ['mount_sylvan', 'bethel', 'hainesville', 'steen', 'boggy_creek']
    dome_names = list(all_domes['domeName'].unique())

    domes = []
    for dome_name in dome_names:
        this_dome_df = all_domes[all_domes['domeName'] == dome_name].copy()
        dome = SaltDome(dome_name, this_dome_df, depth_col='struct_ft', geometry_col='geometry')
        dome.move_to_origin()
        # dome.plot_contours()
        # dome.plot_3d(save_as=f"{dome_name} 3d.png")
        interval = 500
        dome.resample_depths(interval=interval, force_keep_original=False)
        dome.move_to_origin()
        # dome.plot_contours()
        fig, ax = dome.plot_3d(show=False, save_as=f"{dome_name} 3d {interval=}.png")
        plt.show()

        domes.append(dome)
