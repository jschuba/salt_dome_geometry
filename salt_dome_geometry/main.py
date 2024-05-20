from multiprocessing import dummy

import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm

import Geometry3D as geo

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


def make_cone_dome(center_base: geo.Point, radius: float, side_angle: float,
                   angle_degrees: float, inner_offset: float,
                   ) -> (geo.Cone, geo.Cone):
    if side_angle >= 90 or side_angle <= 0:
        raise ValueError("Side angle must be between 0 and 90.")

    height_unit_vector = geo.Vector(np.cos(np.deg2rad(angle_degrees)), 0, np.sin(np.deg2rad(angle_degrees)))
    height_full = radius * np.tan(np.deg2rad(side_angle))
    height_inner = (radius - inner_offset) * np.tan(np.deg2rad(side_angle))

    dome = geo.Cone(center_base, radius, height_full * height_unit_vector, n=100)
    dome_inner = geo.Cone(center_base, radius - inner_offset, height_inner * height_unit_vector, n=100)

    return dome, dome_inner


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


def make_hex_grid_of_circles_inside_polygon(poly: geo.ConvexPolygon, circle_radius: float, spacing: float) -> list[geo.ConvexPolygon]:
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

    y_start = (y1 + y2) / 2 - (y_step) * (((y2 - y1) // y_step) / 2)
    x_start = (x1 + x2) / 2 - (x_step) * (((x2 - x1) // x_step) / 2)
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


def make_square_grid_of_circles_inside_polygon(poly: geo.ConvexPolygon, circle_radius: float, spacing: float) -> list[geo.ConvexPolygon]:
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

def grid_of_circles_to_grid_of_cylinders(hex_grid: list[geo.ConvexPolygon], circle_radius: float, depth: float=7000):
    cylinders = []
    for circle in hex_grid:
        cylinder = geo.Cylinder(circle.center_point, circle_radius, height_vector=-abs(depth)*geo.z_unit_vector())
        cylinders.append(cylinder)
    return cylinders

def make_bounding_box_for_3d_plot(poly: geo.ConvexPolyhedron) -> geo.ConvexPolygon:
    x1 = min(pt.x for pt in poly.point_set)
    x2 = max(pt.x for pt in poly.point_set)
    y1 = min(pt.y for pt in poly.point_set)
    y2 = max(pt.y for pt in poly.point_set)
    z1 = min(pt.z for pt in poly.point_set)
    z2 = max(pt.z for pt in poly.point_set)

    center = geo.Point((x1+x2)/2, (y1+y2)/2, (z1+z2)/2)
    half_length = max((x2-x1), (y1-y2), (z2-z1)) / 2
    lower_left_point = geo.Point(center.x-half_length, center.y-half_length, center.z-half_length)
    box = geo.Parallelepiped(lower_left_point,
                              half_length * 2 * geo.x_unit_vector(),
                              half_length * 2 * geo.y_unit_vector(),
                              half_length * 2 * geo.z_unit_vector())
    return box

def intersect_caverns_with_dome(dome: geo.ConvexPolyhedron, caverns: list[geo.ConvexPolyhedron]) -> list[geo.ConvexPolyhedron]:
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


def create_cylinder_dome_and_cavern(dome_radius: float, dome_angle: float, ):
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

    print(f"Gridding Caverns: {dome_radius=}, {dome_angle=}")
    hex_grid = make_hex_grid_of_circles_inside_polygon(projection, circle_radius=dome_settings.salt_cavern_diameter / 2,
                                                       spacing=dome_settings.inter_cavern_spacing)
    caverns_prelim = grid_of_circles_to_grid_of_cylinders(hex_grid,
                                                          circle_radius=dome_settings.salt_cavern_diameter / 2,
                                                          depth=dome_settings.bottom_allowed_cavern)
    print(f"Intersecting Caverns: {dome_radius=}, {dome_angle=}")
    intersected_caverns = intersect_caverns_with_dome(dome_inner_cut, caverns_prelim)

    return dome_cut, dome_inner_cut, intersected_caverns


def do_everything_cylinder(dome_radius: float, dome_angle: float, ):
    dome, dome_inner, intersected_caverns = create_cylinder_dome_and_cavern(dome_radius, dome_angle)
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
    fig.suptitle(f"Cylinder Dome diameter: {dome_radius*2: .2f} ft, angle: {dome_angle: .2f}deg,  total cavern volume:{cavern_volume:.2f} ft^3, {len(intersected_caverns)} caverns")
    plt.savefig(f"Cylinder {dome_radius=:.2f} {dome_angle=:.2f} {cavern_volume=:.2f}.png")
    fig.show()
    print(f"Done Plotting")


do_everything_cylinder(dome_radius=2500, dome_angle=90)
do_everything_cylinder(dome_radius=2500, dome_angle=85)
do_everything_cylinder(dome_radius=2500, dome_angle=80)
do_everything_cylinder(dome_radius=2500, dome_angle=75)
do_everything_cylinder(dome_radius=2500, dome_angle=70)
do_everything_cylinder(dome_radius=2500, dome_angle=65)
do_everything_cylinder(dome_radius=2500, dome_angle=60)
do_everything_cylinder(dome_radius=2500, dome_angle=55)
do_everything_cylinder(dome_radius=2500, dome_angle=50)

a = """
Cylinder dome_radius=2500.00 dome_angle=50.00 cavern_volume=1283681807.39.png
Cylinder dome_radius=2500.00 dome_angle=55.00 cavern_volume=1118639288.55.png
Cylinder dome_radius=2500.00 dome_angle=60.00 cavern_volume=998628810.03.png
Cylinder dome_radius=2500.00 dome_angle=65.00 cavern_volume=1021001621.35.png
Cylinder dome_radius=2500.00 dome_angle=70.00 cavern_volume=828060991.63.png
Cylinder dome_radius=2500.00 dome_angle=75.00 cavern_volume=864063643.44.png
Cylinder dome_radius=2500.00 dome_angle=80.00 cavern_volume=648047732.58.png
Cylinder dome_radius=2500.00 dome_angle=85.00 cavern_volume=684050384.39.png
Cylinder dome_radius=2500.00 dome_angle=90.00 cavern_volume=540039777.15.png
"""

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
