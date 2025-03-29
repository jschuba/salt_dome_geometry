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
import salt_dome_geometry.util as util


class Cavern:

    def __init__(self, center: shapely.Point | tuple[float, float], radius, top_depth, bottom_depth, contour_interval=100):
        center = shapely.Point(center)
        if not top_depth > bottom_depth:
            raise ValueError('Top depth must be greater than bottom depth')
        self.center = center
        self.radius = radius
        self.top_depth = top_depth
        self.bottom_depth = bottom_depth

        self.depth_col = 'depth'

        circle = make_circle(self.center, self.radius, n=10)
        contours = []
        contour_depths = set(np.arange(top_depth, bottom_depth, -contour_interval))
        contour_depths.add(bottom_depth)
        contour_depths.add(top_depth)
        for depth in sorted(contour_depths):
            contours.append({self.depth_col: depth, 'geometry': circle})
        contours = pd.DataFrame(contours)
        self.contours = gpd.GeoDataFrame(contours, geometry=contours['geometry'])

    @property
    def volume(self):
        vol = math.pi * self.radius ** 2 * abs(self.top_depth - self.bottom_depth)
        return vol

    @property
    def height(self):
        return abs(self.top_depth - self.bottom_depth)

    def plot_3d_to_ax(self, ax=None, c=None):
        df = self.contours
        # Plot each contour in a different color, and prove that the linestrings are starting in the right places
        for i, (row_index, row) in enumerate(df.iterrows()):
            ls = row.geometry.boundary
            depth = row[self.depth_col]
            # Plot the points of the contour
            ax.scatter(ls.xy[0], ls.xy[1], [depth for _ in ls.xy[0]],
                       color=c,
                       label=depth,
                       s=10)
            # # Plot the contour as a line
            ax.plot(list(ls.xy[0]) + [ls.xy[0][0]], list(ls.xy[1]) + [ls.xy[1][0]],
                    [depth for _ in list(ls.xy[1]) + [ls.xy[1][0]]],
                    color=c,
                    label=depth)

    @property
    def info_df(self):
        d = {'center': self.center, 'radius': self.radius, 'top_depth': self.top_depth,
             'bottom_depth': self.bottom_depth, 'height': self.height, 'volume': self.volume}
        return pd.DataFrame([d])


class SaltDome:

    def __init__(self, dome_name: str, contours: gpd.GeoDataFrame | pd.DataFrame, depth_col: str, geometry_col: str, ):
        self.dome_name = dome_name

        self.contours = gpd.GeoDataFrame(contours, geometry=contours[geometry_col])

        self.depth_col = depth_col
        self.geometry_col = geometry_col

        self.original_contours = self.contours.copy()

        self.caverns = []

    @property
    def info_df(self):
        d = {'dome_name': dome_name}
        return pd.DataFrame(d)

    def move_to_origin(self):
        x_avg = self.contours.geometry.get_coordinates()['x'].mean()
        y_avg = self.contours.geometry.get_coordinates()['y'].mean()
        self.contours.geometry = self.contours.geometry.translate(-x_avg, -y_avg)
        return self

    def resample_contours(self, n=100):
        """ Make sure there are exactly n points in each contour

        :param n:
        :return:
        """
        self.contours.geometry = self.contours.geometry.apply(resample_linestring, n=n)
        return self

    def resample_depths(self, interval=100, force_keep_original=False):
        """ Resample the contours so they have a consistent interval.

        :param interval: Depth interval to resample contours.
        :param force_keep_original: Keep original contours, in addition to the resampled contours.
        :return:
        """
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
        return self

    def truncate_depths(self, depth_min, depth_max):
        indexer = (self.contours[self.depth_col] >= depth_min) & (self.contours[self.depth_col] <= depth_max)
        self.contours = self.contours[indexer]
        return self

    def offset_contours_to_interior(self, distance):
        self.contours.geometry = self.contours.geometry.apply(lambda g: offset_inner(g, distance=distance))
        return self

    def plot_contours(self, show=True, save_as=None, figsize=(10, 10), contour_df=None, ax=None):
        df = contour_df if contour_df is not None else self.contours
        # Plot each contour in a different color, and prove that the linestrings are starting in the right places
        cmap = matplotlib.colormaps.get_cmap('summer')
        color_list = cmap((df[self.depth_col] - df[self.depth_col].min()) /
                          (df[self.depth_col].max() - df[self.depth_col].min()))[::-1]
        for i in range(len(color_list)):
            color_list[i][-1] = 0.5  # Set the color transparency

        if not ax:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection='3d')
            ax.set_aspect('equal', adjustable='box')
        else:
            fig = ax.get_figure()
        for i, (row_index, row) in enumerate(df.iterrows()):
            ls = row.geometry
            # Plot the points of the contour
            plt.scatter(ls.xy[0], ls.xy[1],
                        color=color_list[i],
                        label=row[self.depth_col],
                        )
            # Plot the contour as a line
            ax.plot(list(ls.xy[0]) + [ls.xy[0][0]], list(ls.xy[1]) + [ls.xy[1][0]],
                    color=color_list[i],
                    label=row[self.depth_col])
            # Plot the first point in the contour in black
            plt.scatter(ls.xy[0][0], ls.xy[1][0], c='k')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(self.dome_name)
        plt.axis('scaled')  # same x and y scaling
        plt.tight_layout()
        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()
        return fig, ax

    def plot_3d(self, show=True, save_as=None, figsize=(10, 10), contour_df=None, ax=None, c=None):
        df = contour_df if contour_df is not None else self.contours
        # Plot each contour in a different color, and prove that the linestrings are starting in the right places
        if c is None:
            cmap = matplotlib.colormaps.get_cmap('summer')
            color_list = cmap((df[self.depth_col] - df[self.depth_col].min()) /
                              (df[self.depth_col].max() - df[self.depth_col].min()))[::-1]
            for i in range(len(color_list)):
                color_list[i][-1] = 0.5  # Set the color transparency
        else:
            color_list = [c for _ in range(len(df))]

        if not ax:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection='3d')
            ax.set_aspect('equal', adjustable='box')
        else:
            fig = ax.get_figure()
        for i, (row_index, row) in enumerate(df.iterrows()):
            ls = row.geometry
            depth = row[self.depth_col]
            # Plot the points of the contour
            ax.scatter(ls.xy[0], ls.xy[1], [depth for _ in ls.xy[0]],
                       color=color_list[i],
                       label=depth,
                       s=10)
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

    def project_exterior_to_xy(self, contour_df=None) -> shapely.Polygon:
        df = contour_df if contour_df is not None else self.contours
        geo = df.geometry
        geo = [shapely.Polygon(g) for g in geo]
        union = shapely.ops.unary_union(geo)
        cvxhull = shapely.Polygon(union)
        return cvxhull

    def place_caverns(self, grid: list[shapely.Polygon], radius) -> list[Cavern]:
        # Sort the contours by depth, with the shallowest first
        contours = self.contours.sort_values(by=self.depth_col, ascending=False)
        caverns = []

        for circle in grid:
            cavern_eligible_depths = []
            for row_number, (depth, geo) in contours[[self.depth_col, 'geometry']].iterrows():
                if shapely.Polygon(geo).contains(circle):
                    cavern_eligible_depths.append({'depth': depth, 'eligible': 1})
            cavern_eligible_depths = pd.DataFrame(cavern_eligible_depths)
            cavern_eligible_depths['set'] = cavern_eligible_depths['eligible'].cumsum() - cavern_eligible_depths.index
            set_grp = cavern_eligible_depths.groupby(['set'])
            cavern_eligible_depths['set_height'] = set_grp['depth'].transform("max") - set_grp['depth'].transform("min")
            cavern_eligible_depths = cavern_eligible_depths.sort_values(['set_height', 'set', 'depth'], ascending=[False, True, False])
            best_set_indexer = cavern_eligible_depths['set'] == cavern_eligible_depths['set'][0]
            top = cavern_eligible_depths.loc[best_set_indexer, 'depth'].max()
            bottom = cavern_eligible_depths.loc[best_set_indexer, 'depth'].min()
            if top - bottom > dome_settings.max_cavern_height:
                # The cavern is too tall, so restrict it
                bottom = top - dome_settings.max_cavern_height
                cavern = Cavern(circle.centroid, radius, top, bottom)
            elif top - bottom < dome_settings.min_cavern_height:
                # The cavern is too small, discard it
                cavern = None
            else:
                cavern = Cavern(circle.centroid, radius, top, bottom)

            if cavern:
                caverns.append(cavern)
        self.caverns = caverns
        return self.caverns

    def total_cavern_volume(self) -> float:
        cavern_volume = 0
        for cavern in self.caverns:
            cavern_volume += cavern.volume
        return cavern_volume


if __name__ == '__main__':

    all_domes = gpd.read_file(
        r"C:\Users\jschu\PycharmProjects\salt_dome_geometry\salt_dome_geometry\dataverse_files\gis\txSaltDiapStruct_v01.shp")

    # Get only domes of interest
    dome_names = ['mount_sylvan', 'bethel', 'hainesville', 'steen', 'boggy_creek']
    # dome_names = list(all_domes['domeName'].unique())

    domes = []
    info_df = []
    for dome_name in dome_names:
        this_dome_df = all_domes[all_domes['domeName'] == dome_name].copy()
        dome = SaltDome(dome_name, this_dome_df, depth_col='struct_ft', geometry_col='geometry')
        dome.move_to_origin()
        dome.original_contours = dome.contours.copy()

        interval = 100
        dome.resample_depths(interval=interval, force_keep_original=False)
        dome.truncate_depths(dome_settings.bottom_allowed_cavern, dome_settings.top_allowed_cavern)
        outer_cvxhull = dome.project_exterior_to_xy()
        dome.offset_contours_to_interior(dome_settings.edge_of_salt_to_cavern)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('equal', adjustable='box')
        dome.plot_3d(show=False, ax=ax)
        dome.plot_3d(show=False, ax=ax, contour_df=dome.original_contours, c='grey')

        plt.show()

        cvxhull = dome.project_exterior_to_xy()
        grid = grid_search_best_hex_grid_of_circles_inside_polygon(cvxhull,
                                                                   circle_radius=dome_settings.salt_cavern_diameter / 2,
                                                                   spacing=dome_settings.inter_cavern_spacing)

        dome.place_caverns(grid, radius=dome_settings.salt_cavern_diameter / 2)

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.plot(cvxhull.boundary.xy[0], cvxhull.boundary.xy[1], 'g')
        ax.plot(outer_cvxhull.boundary.xy[0], outer_cvxhull.boundary.xy[1], 'k')
        for g in grid:
            ax.plot(g.boundary.xy[0], g.boundary.xy[1])
        plt.axis('scaled')  # same x and y scaling
        plt.show()


        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('equal', adjustable='box')
        ax.view_init(elev=30, azim=118, roll=0)
        dome.plot_3d(show=False, ax=ax)
        # dome.plot_3d(show=False, ax=ax, contour_df=dome.original_contours, c='grey')
        for cav in dome.caverns:
            cav.plot_3d_to_ax(ax, c=(1, 0, 0, 0.2))
        plt.show()


        fig, ax = plt.subplots(figsize=(20, 20))
        ax.plot(cvxhull.boundary.xy[0], cvxhull.boundary.xy[1], 'g')
        ax.plot(outer_cvxhull.boundary.xy[0], outer_cvxhull.boundary.xy[1], 'k')
        for cav in dome.caverns:
            ax.plot(g.boundary.xy[0], g.boundary.xy[1])
        plt.axis('scaled')  # same x and y scaling
        plt.show()
        # dome.plot_contours()
        # plt.show()
        #
        # fig, ax = dome.plot_3d(show=False, save_as=util.gen_png_filename(f"{dome_name} 3d {interval=}"))
        # plt.show()

        domes.append(dome)

        for i, cav in enumerate(dome.caverns):
            df = cav.info_df
            df['dome_name'] = dome.dome_name
            df['cavern_number'] = i + 1
            info_df.append(df)

    info_df = pd.concat(info_df)
