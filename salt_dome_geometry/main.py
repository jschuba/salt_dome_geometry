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
from scipy.spatial import ConvexHull

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
    dome = make_truncated_cone(center_base, radius_base, radius_top, height,
                               angle_degrees)
    dome_inner = make_truncated_cone(center_base, radius_base - inner_offset, radius_top - inner_offset, height,
                                     angle_degrees)
    return dome, dome_inner


def make_ellipse_cylinder_dome(center_base: geo.Point, radius_base: float, b_over_a: float, height: float, angle_degrees: float, inner_offset: float, ):
    dome = make_ellipse_cylinder(center_base, radius_base, b_over_a, height,
                                 angle_degrees)
    dome_inner = make_ellipse_cylinder(center_base, radius_base - inner_offset, b_over_a, height,
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


def make_ellipse_cylinder(center_base: geo.Point, radius_base: float, b_over_a: float, height: float, angle_degrees: float, ):
    # assert 0 <= top_eccentricity <= 1
    # b_over_a = np.sqrt(1-top_eccentricity**2)
    b_over_a = b_over_a
    assert 0 <= b_over_a <= 1

    n = 100
    height_unit_vector = geo.Vector(np.cos(np.deg2rad(angle_degrees)), 0, np.sin(np.deg2rad(angle_degrees)))
    height_vector = height * height_unit_vector
    import copy

    bottom_circle = geo.Circle(center=center_base, normal=height_vector, radius=radius_base, n=n)
    bottom_ellipse_point_list = [geo.Point(pt.x, b_over_a * pt.y, pt.z) for pt in bottom_circle.points]
    bottom_ellipse = geo.ConvexPolygon(bottom_ellipse_point_list)
    
    top_point = copy.deepcopy(center_base).move(height_vector)
    top_circle = geo.Circle(center=top_point, normal=height_vector, radius=radius_base, n=n)
    top_ellipse_point_list = [geo.Point(pt.x, b_over_a * pt.y, pt.z) for pt in top_circle.points]
    top_ellipse = geo.ConvexPolygon(top_ellipse_point_list)

    cpg_list = [top_ellipse, bottom_ellipse]
    for i in range(len(top_ellipse_point_list)):
        start = i
        end = (i + 1) % len(top_ellipse_point_list)
        cpg_list.append(geo.ConvexPolygon((top_ellipse_point_list[start], top_ellipse_point_list[end],
                                           bottom_ellipse_point_list[end], bottom_ellipse_point_list[start])))
    return geo.ConvexPolyhedron(tuple(cpg_list))


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


def  make_hex_grid_of_circles_inside_polygon(poly: geo.ConvexPolygon, circle_radius: float, spacing: float,
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
                        geo.Circle(geo.Point(x_circ, y, 0), normal=geo.z_unit_vector(), radius=circle_radius, n=30))
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

    if best_number == 1:
        # Put the cavern in the middle
        center = poly.center_point
        circ = geo.Circle(geo.Point(center.x, center.y, 0), normal=geo.z_unit_vector(), radius=circle_radius, n=30)
        best_hex_grid = [circ]

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
        f"Cone Dome radius_bottom: {radius_bottom: .2f} ft, radius_top: {radius_top: .2f} ft,  total cavern volume:{cavern_volume:.2f} ft^3, {len(intersected_caverns)} caverns")
    plt.savefig(f"Cone {radius_bottom=:.2f} {radius_top=:.2f} {cavern_volume=:.2f}.png")
    fig.show()
    print(f"Done Plotting")
    return dome, dome_inner, intersected_caverns, cavern_volume



def do_everything_ellipse(radius_bottom: float, b_over_a, dome_angle: float = 90,
                          base_depth=dome_settings.bottom_of_salt, top_depth=dome_settings.top_of_salt):
    print(f"Creating Ellipse Domes and Caverns: {radius_bottom=}, {b_over_a=}, {dome_angle=}")
    height = abs(base_depth - top_depth)
    dome, dome_inner = make_ellipse_cylinder_dome(
        center_base=geo.Point(0, 0, base_depth),
        radius_base=radius_bottom, b_over_a=b_over_a,
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
        f"Ellipse Dome radius_bottom: {radius_bottom: .2f} ft, b_over_a: {b_over_a: .2f},  total cavern volume:{cavern_volume:.2f} ft^3, {len(intersected_caverns)} caverns")
    plt.savefig(f"Ellipse {radius_bottom=:.2f} {b_over_a=:.2f} {cavern_volume=:.2f}.png")
    fig.show()
    print(f"Done Plotting")
    return dome, dome_inner, intersected_caverns, cavern_volume


if __name__ == "__main__":

    results = []


    angles = [90, 85, 80, 75, 70, 65, 60, 55, 50]
    for dome_angle in angles:
        dome_radius = 2500

        dome, dome_inner, intersected_caverns, cavern_volume = do_everything_cylinder(dome_radius=dome_radius,
                                                                                      dome_angle=dome_angle)
        results.append({"type": 'cylinder',
                        "dome_radius": dome_radius,
                        "dome_angle": dome_angle,
                        "cavern_volume": cavern_volume,
                        "cavern_number": len(intersected_caverns),
                        "dome_volume": dome.volume(),
                        "dome_usable_volume": dome_inner.volume(),
                        "dome": dome,
                        "dome_inner": dome_inner,
                        "intersected_caverns": intersected_caverns})

    plt.scatter([res['dome_angle'] for res in results if res['type'] == 'cylinder'],
                [res['cavern_volume'] for res in results if res['type'] == 'cylinder'], c='b', label="Volume cu.ft.")
    plt.scatter([res['dome_angle'] for res in results if res['type'] == 'cylinder'],
                [res['cavern_number'] for res in results if res['type'] == 'cylinder'], c='r', label="Number of Caverns")
    plt.legend()
    plt.title("Cavern Volume vs Dome Angle for Cylinder Dome")
    plt.savefig("Cavern Volume vs Dome Angle for Cylinder Dome.png")
    plt.show()


    #
    # bottom_radius = 2500
    # dome_angle = 90
    # tapers = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    # for taper in tapers:
    #     top_radius = bottom_radius * taper
    #     dome, dome_inner, intersected_caverns, cavern_volume = do_everything_cone(
    #         radius_bottom=bottom_radius, radius_top=top_radius, dome_angle=dome_angle)
    #
    #     results.append({"type": 'cone',
    #                     "bottom_radius": bottom_radius,
    #                     "top_radius": top_radius,
    #                     "taper": taper,
    #                     "dome_angle": dome_angle,
    #                     "cavern_volume": cavern_volume,
    #                     "cavern_number": len(intersected_caverns),
    #                     "dome_volume": dome.volume(),
    #                     "dome_usable_volume": dome_inner.volume(),
    #                     "dome": dome,
    #                     "dome_inner": dome_inner,
    #                     "intersected_caverns": intersected_caverns})
    # plt.scatter([res['taper'] for res in results if res['type'] == 'cone'],
    #             [res['cavern_volume'] for res in results if res['type'] == 'cone'], c='b', label="Volume cu.ft.")
    # plt.scatter([res['taper'] for res in results if res['type'] == 'cone'],
    #             [res['cavern_number'] for res in results if res['type'] == 'cone'], c='r', label="Number of Caverns")
    # plt.legend()
    # plt.title("Cavern Volume vs Taper for Cone Dome")
    # plt.savefig("Cavern Volume vs Taper for Cone Dome.png")
    # plt.show()

    #
    #
    # bottom_radius = 2500
    # dome_angle = 90
    # b_over_as = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # for b_over_a in b_over_as:
    #     dome, dome_inner, intersected_caverns, cavern_volume = do_everything_ellipse(
    #         radius_bottom=bottom_radius, b_over_a=b_over_a, dome_angle=dome_angle)
    #
    #     results.append({"type": 'ellipse',
    #                     "bottom_radius": bottom_radius,
    #                     "b_over_a": b_over_a,
    #                     "dome_angle": dome_angle,
    #                     "cavern_volume": cavern_volume,
    #                     "cavern_number": len(intersected_caverns),
    #                     "dome_volume": dome.volume(),
    #                     "dome_usable_volume": dome_inner.volume(),
    #                     "dome": dome,
    #                     "dome_inner": dome_inner,
    #                     "intersected_caverns": intersected_caverns})
    # plt.scatter([res['b_over_a'] for res in results if res['type'] == 'ellipse'],
    #             [res['cavern_volume'] for res in results if res['type'] == 'ellipse'], c='b', label="Volume cu.ft.")
    # plt.scatter([res['b_over_a'] for res in results if res['type'] == 'ellipse'],
    #             [res['cavern_number'] for res in results if res['type'] == 'ellipse'], c='r', label="Number of Caverns")
    # plt.legend()
    # plt.title("Cavern Volume vs b_over_a for ellipse Dome")
    # plt.savefig("Cavern Volume vs b_over_a for ellipse Dome.png")
    # plt.show()

    results_df = pd.DataFrame(results)
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    results_df.to_csv(f"results_{now}.csv")
    results_df.to_pickle(f"results_{now}.pkl")


