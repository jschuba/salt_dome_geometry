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


def resample_linestring(ls: shapely.LineString, start_closest_to=shapely.Point(-100000000, 0),
                        make_convex=False) -> shapely.LineString:
    # Resample each contour linestring so it has 100 points and the linestring starts on the west side.
    if make_convex:
        convex_hull = ls.convex_hull
        new = [convex_hull.boundary.interpolate(n / 100 * convex_hull.boundary.length) for n in range(101)]
    else:
        new = [ls.interpolate(n / 100 * ls.length) for n in range(101)]
    distances = [start_closest_to.distance(pt) for pt in new]
    start_index = np.argmin(distances)
    new_n = new[start_index:] + new[:start_index]
    return shapely.LineString(new_n)


def interpolate_between_linestrings(ls1: shapely.geometry.LineString, ls2: shapely.geometry.LineString,
                                    percent: float) -> shapely.geometry.LineString:
    # interpolates between two different linestrings, assuming that they have the same number of points
    # and that we should interpolate between points with the same index.
    assert len(ls1.coords) == len(ls2.coords), f"{len(ls1.coords)=} != {len(ls2.coords)=}"
    assert 0 <= percent <= 1, f"Percent must be between 0 and 1, got {percent}"
    if percent == 0:
        return ls1
    if percent == 1:
        return ls2

    start_closest_to = shapely.Point(ls1.coords[0])
    distances = [start_closest_to.distance(shapely.Point(pt)) for pt in ls2.coords]
    start_index = np.argmin(distances)
    if start_index != 0:
        print(f"Realigning linestring two by starting at {start_index=}")
        ls2 = [shapely.Point(pt) for pt in ls2.coords]
        ls2 = ls2[start_index:] + ls2[:start_index]
        ls2 = shapely.LineString(ls2)

    vec1 = (ls1.coords[0][0] - ls1.coords[1][0], ls1.coords[0][1] - ls1.coords[1][1])
    vec2 = (ls2.coords[0][0] - ls2.coords[1][0], ls2.coords[0][1] - ls2.coords[1][1])
    if np.dot(vec1, vec2) < 0:
        ls2 = ls2.reverse()

    new = []
    for pt1, pt2 in zip(ls1.coords, ls2.coords):
        pt = shapely.Point(pt1[0] + percent * (pt2[0] - pt1[0]), pt1[1] + percent * (pt2[1] - pt1[1]))
        new.append(pt)
    return shapely.geometry.LineString(new)


def offset_inner(geom: shapely.geometry.LineString,
                 distance=dome_settings.edge_of_salt_to_cavern) -> shapely.geometry.LineString:
    geom_p = shapely.Polygon(geom)
    inner = shapely.offset_curve(geom, distance=distance)
    if all(geom_p.contains(shapely.Point(*pt)) for pt in inner.coords):
        return inner
    inner = shapely.offset_curve(geom, distance=-distance)
    if all(geom_p.contains(shapely.Point(*pt)) for pt in inner.coords):
        return inner
    return None
    # raise AssertionError("Something went wrong")


def plot_dome(df: pd.DataFrame, show=True, save_file=""):
    # Plot each contour in a different color, and prove that the linestrings are starting in the right places
    cmap = matplotlib.colormaps.get_cmap('summer')
    color_list = cmap((df['struct_ft'] - df['struct_ft'].min()) / (df['struct_ft'].max() - df['struct_ft'].min()))[::-1]
    for i in range(len(color_list)):
        color_list[i][-1] = 0.5
    fig, ax = plt.subplots(figsize=(20, 20))
    for i, (row_index, row) in enumerate(df.iterrows()):
        try:
            ls = resample_linestring(row.geometry)
            plt.scatter(ls.xy[0], ls.xy[1], color=color_list[i], label=row['struct_ft'], s=40)
            ax.plot(list(ls.xy[0]) + [ls.xy[0][0]], list(ls.xy[1]) + [ls.xy[1][0]], color=color_list[i],
                    label=row['struct_ft'])
            plt.scatter(ls.xy[0][0], ls.xy[1][0], c='k')
        except Exception as e:
            print(e)
    ax = plt.gca()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(row['domeName'])
    plt.axis('scaled')
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def interpolate_at_interval(df: pd.DataFrame, interval: int = 1000) -> pd.DataFrame:
    domes = []
    depth_col = 'struct_ft'
    df = df.sort_values(depth_col).reset_index(drop=True)
    for i, row in df.iterrows():
        # print(i, row['struct_ft'])
        if i == 0:
            lower_depth = row[depth_col]
            lower_geo = resample_linestring(row.geometry)
            lower_first_point = shapely.Point(lower_geo.coords[0])
            continue
        up_depth = row[depth_col]
        up_geo = resample_linestring(row.geometry)
        up_geo = [shapely.Point(pt) for pt in up_geo.coords]
        # Find point in up that is closest to the first point in lower, and reindex
        distances = [lower_first_point.distance(pt) for pt in up_geo]
        start_index = np.argmin(distances)
        up_geo = shapely.LineString(up_geo[start_index:] + up_geo[:start_index])

        # print(f"starting interpolation between {lower_depth=}, {up_depth=}")
        this_row_new = []
        for depth in np.arange(lower_depth, up_depth, interval):
            if depth == lower_depth:
                continue
            if depth >= up_depth:
                break
            percent = (depth - lower_depth) / (up_depth - lower_depth)
            # print(f"{depth=}, {lower_depth=}, {up_depth=}, {percent=}")
            new = interpolate_between_linestrings(lower_geo, up_geo, percent)
            new_row = [{'struct_ft': depth, 'geometry': new, 'domeName': row['domeName'], 'interp': True}]
            new_row = pd.DataFrame(new_row)
            this_row_new.append(new_row)
            # plot_dome(pd.concat(this_row_new))

        domes.extend(this_row_new)
        new_row = [{'struct_ft': up_depth, 'geometry': up_geo, 'domeName': row['domeName'], 'interp': False}]
        new_row = pd.DataFrame(new_row)
        domes.append(new_row)

        lower_depth = up_depth
        lower_geo = up_geo
        lower_first_point = shapely.Point(up_geo.coords[0])
    domes = pd.concat(domes)
    return domes


if __name__ == "__main__":

    all_domes = gpd.read_file(
        r"C:\Users\jschu\PycharmProjects\salt_dome_geometry\salt_dome_geometry\dataverse_files\gis\txSaltDiapStruct_v01.shp")

    # Get only domes of interest
    # dome_names = ['mount_sylvan', 'bethel', 'hainesville', 'steen', 'boggy_creek']
    dome_names_with_sufficient_data = ['allen', 'arriola', 'batson',
                                       'big_creek', 'big_hill', 'blue_ridge', 'boggy_creek', 'boling',
                                       'brenham', 'brooks', 'brushy_creek', 'bryan_mound', 'bullard',
                                       'butler', 'cedar_point', 'clam_lake',
                                       'concord', 'damond_mound', 'davis_hill', 'day',
                                       'dilworth_ranch', 'east_tyler', 'elkhart', 'esperson', 'fannett',
                                       'ferguson_crossing', 'girlie_caldwell', 'grand_saline', 'gulf',
                                       'gyp_hill', 'hainesville', 'hankamer', 'hawkinsville',
                                       'hoskins_mound', 'hull', 'humble',
                                       'keechi', 'kittrell', 'la_rue', 'long_point', 'lost_lake',
                                       'manvel', 'markham', 'millican', 'moca',
                                       'mount_sylvan', 'mykawa', 'nash', 'north_dayton', 'oakwood',
                                       'orange', 'palangana', 'palestine', 'pescadito',
                                       'pierce_junction', 'port_neches', 'raccoon_bend',
                                       'red_fish', 'san_felipe', 'slocum', 'sour_lake',
                                       'south_houston', 'spindletop', 'steen',
                                       'stratton_ridge', 'thompson', 'webster',
                                       'west_columbia', 'whitehouse']
    dome_names = ['allen', 'arriola', 'batson', 'big_creek', 'big_hill', 'blue_ridge', 'boggy_creek', 'boling',
                  'brenham', 'brooks', 'brushy_creek', 'bryan_mound', 'bullard', 'butler', 'damond_mound', 'davis_hill',
                  'day', 'east_tyler', 'fannett', 'ferguson_crossing', 'grand_saline', 'gulf', 'gyp_hill',
                  'hainesville', 'hawkinsville', 'hoskins_mound', 'hull', 'humble',
                  'keechi', 'kittrell', 'long_point', 'markham',
                  'mount_sylvan', 'nash', 'north_dayton', 'oakwood', 'palangana', 'palestine',
                  'pierce_junction', 'sour_lake', 'spindletop', 'steen',
                  'stratton_ridge', 'west_columbia', 'whitehouse']

    domes = []
    for dome_name in dome_names:
        this_dome = all_domes[all_domes['domeName'] == dome_name].copy()
        # shift the domes so each dome is centered on the origin
        x_avg = this_dome.geometry.get_coordinates()['x'].mean()
        y_avg = this_dome.geometry.get_coordinates()['y'].mean()
        this_dome.geometry = this_dome.geometry.translate(-x_avg, -y_avg)
        domes.append(this_dome)
    domes = pd.concat(domes)

    domes.geometry = domes.geometry.apply(resample_linestring)

    domes_interp = []
    for name, grp in domes.groupby('domeName'):
        new_grp = interpolate_at_interval(grp, interval=500)
        plot_dome(grp)
        # plot_dome(new_grp)
        domes_interp.append(new_grp)
    domes_interp = pd.concat(domes_interp)

    domes_cut = domes_interp[(domes_interp['struct_ft'] <= dome_settings.top_allowed_cavern) &
                             (domes_interp['struct_ft'] >= dome_settings.bottom_allowed_cavern)]

    inner_domes = domes_cut.copy()
    inner_domes.geometry = inner_domes.geometry.apply(offset_inner)
    inner_domes = inner_domes[inner_domes.geometry.notna()]
    inner_domes['inner_outer'] = 'inner'

    domes_cut['inner_outer'] = 'outer'

    inner_outer_domes = pd.concat([inner_domes, domes_cut])

    for name, grp in inner_domes.groupby('domeName'):
        plot_dome(grp)
        break

    inner_domes = gpd.GeoDataFrame(inner_domes, geometry=inner_domes.geometry)
    dome_cavern_intersections = []
    for name, grp in inner_domes.groupby('domeName'):
        print(name)
        coords = grp.geometry.get_coordinates()
        cvxhull = ConvexHull(coords)
        convex_points = []
        for i in cvxhull.vertices:
            convex_points.append(geo.Point(cvxhull.points[i][0], cvxhull.points[i][1], 0))
        projection = geo.ConvexPolygon(convex_points)
        hex_grid = grid_search_best_hex_grid_of_circles_inside_polygon(
            projection, circle_radius=dome_settings.salt_cavern_diameter / 2,
            spacing=dome_settings.inter_cavern_spacing)

        for circ_number, circ in enumerate(hex_grid):
            circle = shapely.geometry.Polygon([(pt.x, pt.y) for pt in circ.points])
            intersection = grp.copy()
            intersection['cavern_number'] = circ_number
            intersection['cavern'] = circle
            intersection['cavern_center'] = circ.center_point
            intersection['intersection'] = intersection.geometry.apply(
                lambda geo: shapely.Polygon(geo).intersection(circle))
            intersection['intersection_area'] = intersection['intersection'].apply(lambda geo: geo.area)
            intersection = intersection.sort_values(['struct_ft'])
            max_area = math.pi * (dome_settings.salt_cavern_diameter / 2) ** 2
            intersection['has_max_area'] = abs(intersection['intersection_area'] - max_area) / max_area <= 0.001
            intersection = intersection.reset_index(drop=True)

            # for volume, there are three cases:
            # 1) there are enough consecutive contours with max area, that we can have a full-volume cavern
            # 2) There is at least one max-area contour.  We can expand outward from there.
            # 3) We expand outward from the largest-area contour.

            first_largest_index = intersection['intersection_area'].argmax()

            next_down_index = first_largest_index - 1 if first_largest_index > 0 else 0
            this_down_index = first_largest_index
            this_up_index = first_largest_index
            next_up_index = first_largest_index + 1 if first_largest_index < len(intersection) - 1 else len(
                intersection) - 1

            max_height = dome_settings.max_cavern_height
            height_used = 0
            volume_found = 0
            n_iter = 0

            while True:
                if n_iter > 5:
                    break
                n_iter += 1
                print(f"+++++Start Iteration {n_iter}+++++")
                print(f"{height_used=}, volume_found={volume_found}")

                if height_used >= max_height:
                    break
                if this_down_index == 0 and this_up_index == len(intersection) - 1:
                    break

                next_down_depth = intersection.loc[next_down_index, 'struct_ft']
                next_down_area = intersection.loc[next_down_index, 'intersection_area']

                this_down_depth = intersection.loc[this_down_index, 'struct_ft']
                this_down_area = intersection.loc[this_down_index, 'intersection_area']

                this_up_depth = intersection.loc[this_up_index, 'struct_ft']
                this_up_area = intersection.loc[this_up_index, 'intersection_area']

                next_up_depth = intersection.loc[next_up_index, 'struct_ft']
                next_up_area = intersection.loc[next_up_index, 'intersection_area']

                if abs(next_down_area - this_down_area) < 100 and next_down_depth < this_down_depth:
                    # Expand Fully downward
                    print("Down Area is the same as here. Expanding Down.")
                    height = abs(this_down_depth - next_down_depth)
                    if height + height_used > max_height:
                        height = max_height - height_used
                    this_volume = height * (this_down_area + next_down_area) * 2
                    height_used += height
                    volume_found += this_volume
                    intersection.loc[next_down_index, 'take'] = n_iter
                    this_down_index = this_down_index - 1 if this_down_index > 0 else 0
                    next_down_index = next_down_index - 1 if next_down_index > 0 else 0
                    continue

                if abs(next_up_area - this_up_area) < 100 and next_up_depth > this_up_depth:
                    # Expand Fully Up
                    print("Up Area is the same as here. Expanding Up.")
                    height = abs(this_up_depth - next_up_depth)
                    if height + height_used > max_height:
                        height = max_height - height_used
                    this_volume = height * (this_up_area + next_up_area) * 2
                    height_used += height
                    volume_found += this_volume
                    intersection.loc[next_up_index, 'take'] = n_iter
                    this_up_index = this_up_index + 1 if this_up_index < len(intersection) - 1 else len(
                        intersection) - 1
                    next_up_index = next_up_index + 1 if next_up_index < len(intersection) - 1 else len(
                        intersection) - 1
                    continue

                height_up = abs(this_up_depth - next_up_depth)
                height_down = abs(this_down_depth - next_down_depth)

                if max_height - height_used > height_up + height_down:
                    # we can Take both up and down.  We take the larger at this time.

                    up_volume = height_up * (this_up_area + next_up_area) * 2
                    down_volume = height_down * (this_down_area + next_down_area) * 2

                    if up_volume > down_volume:
                        print("Up volume is greater than down volume.  Taking all of up")
                        height_used += height_up
                        volume_found += up_volume
                        intersection.loc[next_up_index, 'take'] = n_iter

                        this_up_index = this_up_index + 1 if this_up_index < len(intersection) - 1 else len(
                            intersection) - 1
                        next_up_index = next_up_index + 1 if next_up_index < len(intersection) - 1 else len(
                            intersection) - 1
                        continue

                    else:
                        print("Down volume is greater.  Taking all of Down")
                        height_used += height_down
                        volume_found += down_volume
                        intersection.loc[next_down_index, 'take'] = n_iter

                        this_down_index = this_down_index - 1 if this_down_index > 0 else 0
                        next_down_index = next_down_index - 1 if next_down_index > 0 else 0
                        continue


                def find_hstar(height_to_take: float):
                    # hstar is the amount of height to take from down, if we want to take from both sides
                    # https://www.wolframalpha.com/input?i=a11+*+%28h1+-+h%29+%2F+h1+%2B+a12+*+h+%2F+h1+%3D+a21+*+%28h2+-+%28c-h%29%29+%2F+h2+%2B+a22+*+%28c-h%29+%2F+h2%3B++solve+for+h
                    num = height_down * (-this_down_area * height_up + this_up_area * (
                            height_up - height_to_take) + next_up_area * height_to_take)
                    den = height_up * (
                            next_down_area - this_down_area) - this_up_area * height_down + next_up_area * height_down
                    return num / den


                height_to_take = min(max_height - height_used, height_down + height_up)
                hstar = find_hstar(height_to_take)
                if pd.isna(hstar):
                    hstar = 0
                print(f"{hstar=}")
                hstar = min(hstar, height_down)
                height_down = hstar
                height_up = height_to_take - hstar

                up_volume = height_up * (this_up_area + next_up_area) * 2
                down_volume = height_down * (this_down_area + next_down_area) * 2
                print(f"Using {hstar=} and expanding both directions {height_down=} and {height_up=}")

                height_used += height_up
                volume_found += up_volume
                intersection.loc[next_up_index, 'take'] = n_iter
                this_up_index = this_up_index + 1 if this_up_index < len(intersection) - 1 else len(intersection) - 1
                next_up_index = next_up_index + 1 if next_up_index < len(intersection) - 1 else len(intersection) - 1

                height_used += height_down
                volume_found += down_volume
                intersection.loc[next_down_index, 'take'] = n_iter
                this_down_index = this_down_index - 1 if this_down_index > 0 else 0
                next_down_index = next_down_index - 1 if next_down_index > 0 else 0

            intersection['cavern_volume'] = volume_found
            intersection['cavern_height'] = height_used

            #
            # for i, row in intersection.iterrows():
            #     print(i, row)

            dome_cavern_intersections.append(intersection)

        fig, ax = plot_dome(grp, show=False)
        for circ_number, circ in enumerate(hex_grid):
            circle = shapely.LineString([(pt.x, pt.y) for pt in circ.points])
            plt.scatter(circle.xy[0], circle.xy[1], label=f"Dome {circ_number}")

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(name)
        plt.axis('scaled')
        plt.tight_layout()
        plt.savefig(f"{name}_with_domes_2d.png")
        plt.show()

    dome_cavern_intersections = pd.concat(dome_cavern_intersections)

    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    dome_cavern_intersections.to_csv(f"real_domes_and_caverns_{now}.csv")
    dome_cavern_intersections.to_pickle(f"real_domes_and_caverns_{now}.pkl")

    dome_cavern_intersections = pd.read_pickle("salt_dome_geometry/real_domes_and_caverns_250209_210243.pkl")

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

        cmap = matplotlib.colormaps.get_cmap('coolwarm')
        color_list = cmap(np.linspace(0,1,grp['cavern_number'].max()+1))[::-1]
        for i in range(len(color_list)):
            color_list[i][-1] = 0.5  # Set the color transparency

        for i, (cavern, cavern_grp) in enumerate(grp.groupby('cavern_number')):
            cavern_grp = cavern_grp[cavern_grp['take'] > 0]
            for row_i, (row_index, row) in enumerate(df.iterrows()):
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
        plt.show()
        plt.savefig(f"{name} 3d with caverns.png")
