from multiprocessing import dummy

import math

import numpy as np
import matplotlib.pyplot as plt
import shapely
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm
import datetime

import pandas as pd

import Geometry3D as geo
import geopandas as gpd

from salt_dome_geometry import dome_settings

geo.set_eps(1e-6)


def make_cylinder_dome(center_base: geo.Point, radius: float, height: float,
                       angle_degrees: float, inner_offset: float,
                       ) -> (geo.Cylinder, geo.Cylinder):
    if angle_degrees == 0:
        height_unit_vector = geo.z_unit_vector()
    else:
        height_unit_vector = geo.Vector(np.cos(np.deg2rad(angle_degrees)), 0, np.sin(np.deg2rad(angle_degrees)))

    dome = geo.Cylinder(center_base, radius, height * height_unit_vector, n=100)
    dome_inner = geo.Cylinder(center_base, radius - inner_offset, height * height_unit_vector, n=100)

    return dome, dome_inner


def make_truncated_cone_dome(center_base: geo.Point, radius_base: float, radius_top: float, height: float,
                             inner_offset: float,
                             angle_degrees: float = 90) -> (geo.Cone, geo.Cone):
    dome = make_truncated_cone(center_base, radius_base - inner_offset, radius_top - inner_offset, height,
                               angle_degrees)
    dome_inner = make_truncated_cone(center_base, radius_base - inner_offset, radius_top - inner_offset, height,
                                     angle_degrees)
    return dome, dome_inner


def make_truncated_cone(center_base, radius_base, radius_top, height, angle_degrees):
    """ Helper method to make a tuncated cone with radius_base and radius_top.  Similar to the geo.Cylinder class

    :param center_base:
    :param radius_base:
    :param radius_top:
    :param height:
    :param angle_degrees:
    :return:
    """
    n = 100
    height_unit_vector = geo.Vector(np.cos(np.deg2rad(angle_degrees)), 0, np.sin(np.deg2rad(angle_degrees)))
    height_vector = height * height_unit_vector
    import copy
    bottom_circle = geo.Circle(center=center_base, normal=height_vector, radius=radius_base, n=n)
    bottom_circle_point_list = geo.get_circle_point_list(center=center_base, normal=height_vector, radius=radius_base,
                                                         n=n)
    top_point = copy.deepcopy(center_base).move(height_vector)
    top_circle = geo.Circle(center=top_point, normal=height_vector, radius=radius_top, n=n)
    top_circle_point_list = geo.get_circle_point_list(center=top_point, normal=height_vector, radius=radius_top, n=n)
    cpg_list = [top_circle, bottom_circle]
    for i in range(len(top_circle_point_list)):
        start = i
        end = (i + 1) % len(top_circle_point_list)
        cpg_list.append(geo.ConvexPolygon((top_circle_point_list[start], top_circle_point_list[end],
                                           bottom_circle_point_list[end], bottom_circle_point_list[start])))
    return geo.ConvexPolyhedron(tuple(cpg_list))


def make_elliptical_cylinder(center_base: geo.Point, major_radius: float, minor_radius: float, height: float, angle_degrees: float,):
    n = 100
    height_unit_vector = geo.Vector(np.cos(np.deg2rad(angle_degrees)), 0, np.sin(np.deg2rad(angle_degrees)))
    height_vector = height * height_unit_vector
    import copy

    raise NotImplementedError

def make_top_bottom_mask(top_cutoff_depth: float, bottom_cutoff_depth: float, horizontal_size: float = None,
                         center_point: geo.Point = geo.Point(0, 0, 0)):
    if not horizontal_size:
        horizontal_size = abs(top_cutoff_depth - bottom_cutoff_depth) * 2
    top_center_point = geo.Point(center_point.x, center_point.y, top_cutoff_depth)

    mask = geo.Parallelepiped(top_center_point,
                              horizontal_size * geo.x_unit_vector(),
                              horizontal_size * geo.y_unit_vector(),
                              (bottom_cutoff_depth - top_cutoff_depth) * geo.z_unit_vector())
    mask.move(-horizontal_size / 2 * geo.x_unit_vector())
    mask.move(-horizontal_size / 2 * geo.y_unit_vector())
    return mask


def mask_shape(shape: geo.ConvexPolyhedron, mask: geo.ConvexPolyhedron) -> geo.ConvexPolyhedron:
    return shape.intersection(mask)


def project_on_xy(shape: geo.ConvexPolyhedron) -> geo.ConvexPolygon:
    points = shape.point_set
    projected_points = [(p.x, p.y) for p in points]
    from scipy.spatial import ConvexHull
    pts = ConvexHull(projected_points)
    convex_points = []
    for i in pts.vertices:
        convex_points.append(geo.Point(projected_points[i][0], projected_points[i][1], 0))
    return geo.ConvexPolygon(convex_points)


def make_hex_grid_of_circles(corner_1: geo.Point, corner_2: geo.Point, circle_radius: float, spacing: float):
    x1, x2 = min(corner_1.x, corner_2.x), max(corner_1.x, corner_2.x)
    y1, y2 = min(corner_1.y, corner_2.y), max(corner_1.y, corner_2.y)
    x_step = circle_radius * 2 + spacing
    y_step = (circle_radius * 2 + spacing) * np.sin(np.deg2rad(60))
    grid = []
    for y_i, y in enumerate(np.arange(y1, y2, y_step)):
        for x in np.arange(x1, x2, x_step):
            # print(x,y)
            if y_i % 2 == 0:
                grid.append(geo.Circle(geo.Point(x, y, 0), normal=geo.z_unit_vector(), radius=circle_radius, n=100))
            else:
                grid.append(
                    geo.Circle(geo.Point(x + (circle_radius * 2 + spacing) / 2, y, 0), normal=geo.z_unit_vector(),
                               radius=circle_radius, n=100))
    return grid


def make_hex_grid_of_circles_inside_polygon(poly: geo.ConvexPolygon, circle_radius: float, spacing: float,
                                            y_offset: float = 0, x_offset: float = 0) -> list[geo.ConvexPolygon]:
    """

    :param poly:
    :param circle_radius:
    :param spacing: # distance from cavern edge to edge
    :return:
    """
    x1 = min(pt.x for pt in poly.points)
    y1 = min(pt.y for pt in poly.points)
    x2 = max(pt.x for pt in poly.points)
    y2 = max(pt.y for pt in poly.points)
    x_step = circle_radius * 2 + spacing
    y_step = (circle_radius * 2 + spacing) * np.sin(np.deg2rad(60))

    def make_y_start(offset: float) -> float:
        center = (y1 + y2) / 2  # Find center of the poly shape
        left_shift = y_step * (((y2 - y1) // y_step) / 2)  # how far to shift over, so we start near the left edge
        centered_grid_start = center - left_shift  # this is where to start, so the grid is perfectly centered
        start = centered_grid_start - abs(offset)  # Introduce an offset, so we can try multiple grid starts
        return start

    def make_x_start(offset: float) -> float:
        center = (x1 + x2) / 2  # Find center of the polx shape
        left_shift = x_step * (((x2 - x1) // x_step) / 2)  # how far to shift over, so we start near the left edge
        centered_grid_start = center - left_shift  # this is where to start, so the grid is perfectlx centered
        start = centered_grid_start - abs(offset)  # Introduce an offset, so we can trx multiple grid starts
        return start

    y_start = make_y_start(y_offset)
    x_start = make_x_start(x_offset)
    grid = []
    for y_i, y in enumerate(np.arange(y_start, y2, y_step)):
        for x in np.arange(x_start, x2, x_step):
            if y_i % 2 == 0:
                x_circ = x
            else:
                x_circ = x + (circle_radius * 2 + spacing) / 2
            # print(x_circ, y)
            # make a circle of 10 points for the check
            circ = geo.Circle(geo.Point(x_circ, y, 0), normal=geo.z_unit_vector(), radius=circle_radius, n=10)
            if poly.__contains__(circ.center_point):
                if all(poly.__contains__(p) for p in circ.points):
                    # Append a circle with higher resolution
                    grid.append(
                        geo.Circle(geo.Point(x_circ, y, 0), normal=geo.z_unit_vector(), radius=circle_radius, n=100))
    return grid


def grid_search_best_hex_grid_of_circles_inside_polygon(poly: geo.ConvexPolygon, circle_radius: float, spacing: float,
                                                        stepsize: int = 100) -> list[geo.ConvexPolygon]:
    print(f"Searching for best hexagonal grid with {stepsize=}")
    best_number = 0
    best_hex_grid = list()

    for y_offset in range(0, int(spacing + circle_radius), stepsize):
        for x_offset in range(0, int(spacing + circle_radius), stepsize):
            hex_grid = make_hex_grid_of_circles_inside_polygon(poly,
                                                               circle_radius=circle_radius,
                                                               spacing=spacing,
                                                               y_offset=y_offset,
                                                               x_offset=x_offset)
            if len(hex_grid) > best_number:
                best_number = len(hex_grid)
                best_hex_grid = hex_grid
    if not best_hex_grid:
        raise AssertionError('No best hex grid found')
    return best_hex_grid


def make_square_grid_of_circles_inside_polygon(poly: geo.ConvexPolygon, circle_radius: float, spacing: float) -> list[
    geo.ConvexPolygon]:
    x1 = min(pt.x for pt in poly.points)
    y1 = min(pt.y for pt in poly.points)
    x2 = max(pt.x for pt in poly.points)
    y2 = max(pt.y for pt in poly.points)
    x_step = circle_radius * 2 + spacing
    y_step = circle_radius * 2 + spacing

    y_start = (y1 + y2) / 2 - (y_step) * (((y2 - y1) // y_step) / 2)
    x_start = (x1 + x2) / 2 - (x_step) * (((x2 - x1) // x_step) / 2)
    grid = []
    for y_i, y in enumerate(np.arange(y_start, y2, y_step)):
        for x in np.arange(x_start, x2, x_step):
            # print(x_circ, y)
            # make a circle of 10 points for the check
            circ = geo.Circle(geo.Point(x, y, 0), normal=geo.z_unit_vector(), radius=circle_radius, n=10)
            if poly.__contains__(circ.center_point):
                if all(poly.__contains__(p) for p in circ.points):
                    # Append a circle with higher resolution
                    grid.append(geo.Circle(geo.Point(x, y, 0), normal=geo.z_unit_vector(), radius=circle_radius, n=100))
    return grid


def grid_of_circles_to_grid_of_cylinders(hex_grid: list[geo.ConvexPolygon], circle_radius: float, depth: float = 7000):
    cylinders = []
    for circle in hex_grid:
        cylinder = geo.Cylinder(circle.center_point, circle_radius, height_vector=-abs(depth) * geo.z_unit_vector())
        cylinders.append(cylinder)
    return cylinders


def make_bounding_box_for_3d_plot(poly: geo.ConvexPolyhedron) -> geo.ConvexPolygon:
    x1 = min(pt.x for pt in poly.point_set)
    x2 = max(pt.x for pt in poly.point_set)
    y1 = min(pt.y for pt in poly.point_set)
    y2 = max(pt.y for pt in poly.point_set)
    z1 = min(pt.z for pt in poly.point_set)
    z2 = max(pt.z for pt in poly.point_set)

    center = geo.Point((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)
    half_length = max((x2 - x1), (y1 - y2), (z2 - z1)) / 2
    lower_left_point = geo.Point(center.x - half_length, center.y - half_length, center.z - half_length)
    box = geo.Parallelepiped(lower_left_point,
                             half_length * 2 * geo.x_unit_vector(),
                             half_length * 2 * geo.y_unit_vector(),
                             half_length * 2 * geo.z_unit_vector())
    return box


def intersect_caverns_with_dome(dome: geo.ConvexPolyhedron, caverns: list[geo.ConvexPolyhedron]) -> list[
    geo.ConvexPolyhedron]:
    new_caverns = []
    for cavern in caverns:
        if cavern:
            new = cavern.intersection(dome)
            new_caverns.append(new)
    return new_caverns


def calc_total_cavern_volume(caverns: list[geo.ConvexPolyhedron]) -> float:
    cavern_volume = 0
    for cavern in caverns:
        if not cavern:
            continue
        z1 = min(pt.z for pt in cavern.point_set)
        z2 = max(pt.z for pt in cavern.point_set)
        # cavern too small
        if abs(z1 - z2) < dome_settings.min_cavern_height:
            continue
        # cavern too large
        elif abs(z2 - z1) > dome_settings.max_cavern_height:
            cavern_volume += dome_settings.max_cavern_height * math.pi * (dome_settings.salt_cavern_diameter / 2) ** 2
        else:
            cavern_volume += cavern.volume()
    return cavern_volume


def mask_dome_and_create_caverns(dome, dome_inner):
    dome_mask = make_top_bottom_mask(top_cutoff_depth=dome_settings.top_of_salt,
                                     bottom_cutoff_depth=dome_settings.bottom_of_salt, horizontal_size=60000,
                                     center_point=dome.center_point)

    dome_cut = mask_shape(dome, dome_mask)

    cavern_mask = make_top_bottom_mask(top_cutoff_depth=dome_settings.top_allowed_cavern,
                                       bottom_cutoff_depth=dome_settings.bottom_allowed_cavern, horizontal_size=60000,
                                       center_point=dome.center_point)
    dome_inner_cut = mask_shape(dome_inner, cavern_mask)

    projection = project_on_xy(dome_inner_cut)

    print(f"Gridding Caverns")
    hex_grid = grid_search_best_hex_grid_of_circles_inside_polygon(projection,
                                                                   circle_radius=dome_settings.salt_cavern_diameter / 2,
                                                                   spacing=dome_settings.inter_cavern_spacing)
    caverns_prelim = grid_of_circles_to_grid_of_cylinders(hex_grid,
                                                          circle_radius=dome_settings.salt_cavern_diameter / 2,
                                                          depth=dome_settings.bottom_allowed_cavern)
    print(f"Intersecting Caverns")
    intersected_caverns = intersect_caverns_with_dome(dome_inner_cut, caverns_prelim)

    return dome_cut, dome_inner_cut, intersected_caverns


def do_everything_cylinder(dome_radius: float, dome_angle: float, ):
    print(f"Creating Cylinder Domes and Caverns: {dome_radius=}, {dome_angle=}")
    dome, dome_inner = make_cylinder_dome(geo.Point(0, 0, 5000), radius=dome_radius, height=-60000,
                                          angle_degrees=dome_angle,
                                          inner_offset=dome_settings.edge_of_salt_to_cavern)

    dome, dome_inner, intersected_caverns = mask_dome_and_create_caverns(dome, dome_inner)
    box = make_bounding_box_for_3d_plot(dome)
    cavern_volume = calc_total_cavern_volume(intersected_caverns)

    print(f"Plotting Domes and Caverns: {dome_radius=}, {dome_angle=}")
    r2 = geo.Renderer()
    r2.add((dome, 'b', 1))
    r2.add((dome_inner, 'k', 1))
    r2.add((box, 'k', 1))
    for cyl in intersected_caverns:
        if cyl:
            r2.add((cyl, 'y', 1))
    fig, ax = r2.get_fig((12, 12))
    fig.suptitle(
        f"Cylinder Dome diameter: {dome_radius * 2: .2f} ft, angle: {dome_angle: .2f}deg,  total cavern volume:{cavern_volume:.2f} ft^3, {len(intersected_caverns)} caverns")
    plt.savefig(f"Cylinder {dome_radius=:.2f} {dome_angle=:.2f} {cavern_volume=:.2f}.png")
    fig.show()
    print(f"Done Plotting")
    return dome, dome_inner, intersected_caverns, cavern_volume


def do_everything_cone(radius_bottom: float, radius_top, dome_angle: float = 90,
                       base_depth=dome_settings.bottom_of_salt, top_depth=dome_settings.top_of_salt):
    print(f"Creating Cone Domes and Caverns: {radius_bottom=}, {radius_top=}, {dome_angle=}")
    height = abs(base_depth - top_depth)
    dome, dome_inner = make_truncated_cone_dome(
        center_base=geo.Point(0, 0, base_depth),
        radius_base=radius_bottom, radius_top=radius_top,
        height=height,
        angle_degrees=dome_angle, inner_offset=dome_settings.edge_of_salt_to_cavern)

    dome, dome_inner, intersected_caverns = mask_dome_and_create_caverns(dome, dome_inner)
    box = make_bounding_box_for_3d_plot(dome)
    cavern_volume = calc_total_cavern_volume(intersected_caverns)

    print(f"Plotting Domes and Caverns: {radius_bottom=}, {dome_angle=}")
    r2 = geo.Renderer()
    r2.add((dome, 'b', 1))
    r2.add((dome_inner, 'k', 1))
    r2.add((box, 'k', 1))
    for cyl in intersected_caverns:
        if cyl:
            r2.add((cyl, 'y', 1))
    fig, ax = r2.get_fig((12, 12))
    fig.suptitle(
        f"Cylinder Dome diameter: {radius_bottom * 2: .2f} ft, angle: {dome_angle: .2f}deg,  total cavern volume:{cavern_volume:.2f} ft^3, {len(intersected_caverns)} caverns")
    plt.savefig(f"Cylinder {radius_bottom=:.2f} {dome_angle=:.2f} {cavern_volume=:.2f}.png")
    fig.show()
    print(f"Done Plotting")
    return dome, dome_inner, intersected_caverns, cavern_volume


if __name__ == "__main__":

    results = []
    angles = [90, 85, 80, 75, 70, 65, 60, 55, 50]
    for dome_angle in angles:
        dome_radius = 2500
        try:
            dome, dome_inner, intersected_caverns, cavern_volume = do_everything_cylinder(dome_radius=dome_radius,
                                                                                          dome_angle=dome_angle)
            results.append({"type": 'cylinder',
                            "dome_radius": dome_radius,
                            "dome_angle": dome_angle,
                            "cavern_volume": cavern_volume,
                            "cavern_number": len(intersected_caverns),
                            "dome": dome,
                            "dome_inner": dome_inner,
                            "intersected_caverns": intersected_caverns})
        except Exception as e:
            pass
    plt.scatter([res['dome_angle'] for res in results if res['type'] == 'cylinder'],
                [res['cavern_volume'] for res in results if res['type'] == 'cylinder'], c='b', label="Volume cu.ft.")
    plt.scatter([res['dome_angle'] for res in results if res['type'] == 'cylinder'],
                [res['cavern_number'] for res in results if res['type'] == 'cylinder'], c='r', label="Number of Domes")
    plt.legend()
    plt.title("Cavern Volume vs Dome Angle for Cylinder Dome")
    plt.savefig("Cavern Volume vs Dome Angle for Cylinder Dome.png")
    plt.show()

    bottom_radius = 2500
    dome_angle = 90
    tapers = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    for taper in tapers:
        top_radius = bottom_radius * taper
        dome, dome_inner, intersected_caverns, cavern_volume = do_everything_cone(
            radius_bottom=bottom_radius, radius_top=top_radius, dome_angle=dome_angle)

        results.append({"type": 'cone',
                        "bottom_radius": bottom_radius,
                        "top_radius": top_radius,
                        "taper": taper,
                        "dome_angle": dome_angle,
                        "cavern_volume": cavern_volume,
                        "cavern_number": len(intersected_caverns),
                        "dome": dome,
                        "dome_inner": dome_inner,
                        "intersected_caverns": intersected_caverns})
    plt.scatter([res['dome_angle'] for res in results if res['type'] == 'cone'],
                [res['cavern_volume'] for res in results if res['type'] == 'cone'], c='b', label="Volume cu.ft.")
    plt.scatter([res['dome_angle'] for res in results if res['type'] == 'cone'],
                [res['cavern_number'] for res in results if res['type'] == 'cone'], c='r', label="Number of Domes")
    plt.legend()
    plt.title("Cavern Volume vs Dome Angle for Cylinder Dome")
    plt.savefig("Cavern Volume vs Dome Angle for Cylinder Dome.png")
    plt.show()



    all_domes = gpd.read_file(r"C:\Users\jschu\PycharmProjects\salt_dome_geometry\salt_dome_geometry\dataverse_files\gis\txSaltDiapStruct_v01.shp")

    # Get only domes of interest
    dome_names = ['mount_sylvan', 'bethel','hainesville','steen','boggy_creek']

    domes = []
    for dome_name in dome_names:
        this_dome = all_domes[all_domes['domeName'] == dome_name].copy()
        # shift the domes so each dome is centered on the origin
        x_avg = this_dome.geometry.get_coordinates()['x'].mean()
        y_avg = this_dome.geometry.get_coordinates()['y'].mean()
        this_dome.geometry = this_dome.geometry.translate(-x_avg, -y_avg)
        domes.append(this_dome)
    domes = pd.concat(domes)

    # figure out which domes need to have a contour line interpolated
    # this is done by figuring out which row is the shallowest depth that is below the cutoff, and so on.
    domes['lower_interp_point_lower'] = (domes[domes['struct_ft'] <= dome_settings.bottom_allowed_cavern].groupby('domeName')['struct_ft'].transform('max'))
    domes['lower_interp_point_upper'] = (domes[domes['struct_ft'] >= dome_settings.bottom_allowed_cavern].groupby('domeName')['struct_ft'].transform('min'))
    domes['upper_interp_point_lower'] = (domes[domes['struct_ft'] <= dome_settings.top_allowed_cavern].groupby('domeName')['struct_ft'].transform('max'))
    domes['upper_interp_point_upper'] = (domes[domes['struct_ft'] >= dome_settings.top_allowed_cavern].groupby('domeName')['struct_ft'].transform('min'))

    domes['is_lower_interp_point'] = (domes['struct_ft'] == domes['lower_interp_point_lower']) | (domes['struct_ft'] == domes['lower_interp_point_upper'])
    domes['is_upper_interp_point'] = (domes['struct_ft'] == domes['upper_interp_point_lower']) | (domes['struct_ft'] == domes['upper_interp_point_upper'])

    domes = domes.drop(columns=['lower_interp_point_lower', 'lower_interp_point_upper', 'upper_interp_point_lower', 'upper_interp_point_upper'])

    def resample_linestring(ls: shapely.LineString, start_closest_to=shapely.Point(-100000000, 0), make_convex=False) -> shapely.LineString:
        # Resample each contour linestring so it has 100 points and the linestring starts on the west side.
        if make_convex:
            convex_hull = ls.convex_hull
            new = [convex_hull.boundary.interpolate(n/100 * convex_hull.boundary.length) for n in range(101)]
        else:
            new = [ls.interpolate(n / 100 * ls.length) for n in range(101)]
        distances = [start_closest_to.distance(pt) for pt in new]
        start_index = np.argmin(distances)
        new_n = new[start_index:] + new[:start_index]
        return shapely.LineString(new_n)

    def plot_dome(df: pd.DataFrame):
        # Plot each contour in a different color, and prove that the linestrings are starting in the right places
        for i, row in df.iterrows():
            ls = resample_linestring(row.geometry)
            plt.scatter(ls.xy[0], ls.xy[1], label=row['struct_ft'])
            plt.scatter(ls.xy[0][0],ls.xy[1][0], c='k')
        plt.legend()
        plt.title(row['domeName'])
        plt.show()


    domes.geometry = domes.geometry.apply(resample_linestring)

    def interpolate_between_linestrings(ls1: shapely.geometry.LineString, ls2: shapely.geometry.LineString, percent: float) -> shapely.geometry.LineString:
        # interpolates between two different linestrings, assuming that they have the same number of points
        # and that we should interpolate between points with the same index.
        assert len(ls1.coords) == len(ls2.coords)
        assert 0 <= percent < 1
        new = []
        for pt1, pt2 in zip(ls1.coords, ls2.coords):
            pt = shapely.Point(pt1[0] + percent * (pt2[0] - pt1[0]), pt1[1] + percent * (pt2[1] - pt1[1]))
            new.append(pt)
        return shapely.geometry.LineString(new)


    for dome_name in dome_names:
        dome_data = domes[(domes['domeName'] == dome_name) & (domes['is_lower_interp_point'])]
        dome_data = dome_data.sort_values('struct_ft')
        if len(dome_data) == 2:
            # we need to add a new row for interp
            lower, upper = dome_data.iloc[0], dome_data.iloc[1]
            ls1 = lower.geometry
            ls2 = upper.geometry
            percent = (dome_settings.bottom_allowed_cavern - lower['struct_ft']) / (upper['struct_ft'] - lower['struct_ft'])
            new = interpolate_between_linestrings(ls1, ls2, percent)
            new_row = [{'struct_ft': dome_settings.bottom_allowed_cavern, 'geometry': new, 'domeName': dome_name}]
            new_row = pd.DataFrame(new_row)
            domes = pd.concat([domes, new_row])
        dome_data = domes[(domes['domeName'] == dome_name) & (domes['is_upper_interp_point'])]
        dome_data = dome_data.sort_values('struct_ft')
        if len(dome_data) == 2:
            # we need to add a new row for interp
            lower, upper = dome_data.iloc[0], dome_data.iloc[1]
            ls1 = lower.geometry
            ls2 = upper.geometry
            percent = (dome_settings.top_allowed_cavern - lower['struct_ft']) / (
                        upper['struct_ft'] - lower['struct_ft'])
            new = interpolate_between_linestrings(ls1, ls2, percent)
            new_row = [{'struct_ft': dome_settings.top_allowed_cavern, 'geometry': new, 'domeName': dome_name}]
            new_row = pd.DataFrame(new_row)
            domes = pd.concat([domes, new_row])

    def inward_offset(ls: shapely.geometry.LineString, offset_amount: float) -> shapely.geometry.LinearRing:
        ls_p = shapely.geometry.Polygon(ls)
        new_list = []
        def normalize_and_reverse_if_not_inside(this: tuple, normal: tuple) -> tuple:
            length = np.linalg.norm(normal)
            normal = tuple(coord / length for coord in normal)
            check1 = (this[0] + normal[0], this[1] + normal[1])
            check2 = (this[0] - normal[0], this[1] - normal[1])
            if ls_p.contains(shapely.Point(check1)):
                return normal
            elif ls_p.contains(shapely.Point(check2)):
                return tuple(-x for x in normal)
            else:
                return None

        for i in range(len(ls.coords)):
            prev = ls.coords[i - 1 if i>0 else -1]
            this = ls.coords[i]
            next = ls.coords[i + 1 if i<len(ls.coords)-1 else 0]
            if prev == this or this == next or next == prev:
                continue
            try:
                normal1 = (prev[1] - this[1], prev[0] - this[0])
                normal1 = normalize_and_reverse_if_not_inside(this, normal1)

                normal2 = (next[1] - this[1], next[0] - this[0])
                normal2 = normalize_and_reverse_if_not_inside(this, normal2)

                if not (normal1 and normal2):
                    print(f"Something wrong, {i=}")
                    continue

                normal = (normal1[0] + normal2[0], normal1[1] + normal2[1])
                normal = normalize_and_reverse_if_not_inside(this, normal)

                new = (this[0] + offset_amount*normal[0], this[1] + offset_amount*normal[1])
                new_list.append(new)
            except Exception as e:
                pass
        return shapely.geometry.LinearRing(new_list)


    for name, grp in domes.groupby('domeName'):
        plot_dome(grp)


    a = shapely.LinearRing(ls)
    b = inward_offset(a, 400)
    b = shapely.offset_curve(a, distance=400)


    plt.scatter(a.xy[0], a.xy[1], label='a')
    plt.scatter(a.xy[0][0],a.xy[1][0], c='k')

    plt.scatter(b.xy[0], b.xy[1], label='b')
    plt.scatter(b.xy[0][0], b.xy[1][0], c='k')

    plt.legend()
    plt.show()




    results_df = pd.DataFrame(results)
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    results_df.to_csv(f"results_{now}.csv")
    results_df.to_pickle(f"results_{now}.pkl")

    #
    #
    # dome, dome_inner = make_cylinder_dome(geo.Point(0, 0, 5000), radius=2500, height=-60000, angle_degrees=50,
    #                                       inner_offset=dome_settings.edge_of_salt_to_cavern)
    # dome_mask = make_top_bottom_mask(top_cutoff_depth=dome_settings.top_of_salt, bottom_cutoff_depth=dome_settings.bottom_of_salt, horizontal_size=60000,
    #                             center_point=dome.center_point)
    #
    # dome_cut = mask_shape(dome, dome_mask)
    #
    # cavern_mask = make_top_bottom_mask(top_cutoff_depth=dome_settings.top_allowed_cavern, bottom_cutoff_depth=dome_settings.bottom_allowed_cavern, horizontal_size=60000,
    #                             center_point=dome.center_point)
    # dome_inner_cut = mask_shape(dome_inner, cavern_mask)
    #
    # projection = project_on_xy(dome_inner_cut)
    # box = make_bounding_box_for_3d_plot(dome_cut)
    # hex_grid = make_hex_grid_of_circles_inside_polygon(projection, circle_radius=dome_settings.salt_cavern_diameter/2, spacing=dome_settings.inter_cavern_spacing)
    # caverns_prelim = grid_of_circles_to_grid_of_cylinders(hex_grid, circle_radius=dome_settings.salt_cavern_diameter/2, depth=dome_settings.bottom_allowed_cavern)
    # intersected_caverns = intersect_caverns_with_dome(dome_inner_cut, caverns_prelim)
    # cavern_volume = calc_total_cavern_volume(intersected_caverns)
    #
    #
    # r2 = geo.Renderer()
    # # r2.add((dome, 'b', 1))
    # # r2.add((dome_inner, 'r', 1))
    # # r2.add((mask, 'y', 1))
    # r2.add((dome_cut, 'b', 1))
    # r2.add((dome_inner_cut, 'k', 1))
    # r2.add((projection, 'k', 1))
    # r2.add((box, 'k', 1))
    # for cyl in intersected_caverns:
    #     if cyl:
    #         r2.add((cyl, 'y', 1))
    # fig, ax = r2.get_fig()
    # fig.suptitle(f"Test Title {cavern_volume:.2f} ft^3")
    # fig.show()
    #
    #
    #
    #
    #
    #
    #
    #
    # dome, dome_inner = make_cone_dome(geo.Point(0, 0, -30000), radius=5000, side_angle=85, angle_degrees=90,
    #                                   inner_offset=400)
    # mask = make_top_bottom_mask(top_cutoff_depth=-900, bottom_cutoff_depth=-30000, center_point=dome.center_point)
    # dome_cut = mask_shape(dome, mask)
    # dome_inner_cut = mask_shape(dome_inner, mask)
    # projection = project_on_xy(dome_cut)
    #
    # r2 = geo.Renderer()
    # # r2.add((dome,'b',1))
    # # r2.add((dome_inner,'r',1))
    # r2.add((mask, 'y', 1))
    # r2.add((dome_cut, 'b', 1))
    # r2.add((dome_inner_cut, 'r', 1))
    # r2.add((projection, 'k', 1))
    # r2.show()
    #
    # hex_grid = make_hex_grid_of_circles_inside_polygon(projection, circle_radius=100, spacing=400)
    #
    # fig = plt.figure(figsize=(20, 10))
    # ax = plt.gca()
    # ax.set_aspect('equal', adjustable='box')
    # ax.plot([pt.x for pt in projection.points], [pt.y for pt in projection.points], c='r')
    # for circ in hex_grid:
    #     ax.plot([pt.x for pt in circ.points], [pt.y for pt in circ.points], c='k')
    #
    # plt.show()
    #
    # square_grid = make_square_grid_of_circles_inside_polygon(projection, circle_radius=100, spacing=400)
    #
    # fig = plt.figure(figsize=(20, 10))
    # ax = plt.gca()
    # ax.set_aspect('equal', adjustable='box')
    # ax.plot([pt.x for pt in projection.points], [pt.y for pt in projection.points], c='r')
    # for circ in square_grid:
    #     ax.plot([pt.x for pt in circ.points], [pt.y for pt in circ.points], c='k')
    #
    # plt.show()
