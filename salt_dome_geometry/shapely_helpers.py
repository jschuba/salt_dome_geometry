import datetime
import math
from typing import Iterator, List, Tuple, Union

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


def resample_linestring(ls: shapely.LineString, start_closest_to=shapely.Point(-100000000, 0),
                        n=100, make_convex=False) -> shapely.LineString:
    # Resample each contour linestring so it has n points and the linestring starts on the west side.
    if make_convex:
        convex_hull = ls.convex_hull
        new = [convex_hull.boundary.interpolate(n / 100 * convex_hull.boundary.length) for n in range(101)]
    else:
        new = [ls.interpolate(i / n * ls.length) for i in range(n+1)]
    if isinstance(start_closest_to, shapely.geometry.Point):
        distances = [start_closest_to.distance(pt) for pt in new]
        start_index = np.argmin(distances)
        new = new[start_index:] + new[:start_index]
    return shapely.LineString(new)


def interpolate_between_linestrings(ls1: shapely.geometry.LineString, ls2: shapely.geometry.LineString,
                                    percent: float, align_loops=True) -> shapely.geometry.LineString:
    """  interpolates between two different linestrings, assuming that they have the same number of points
     and that we should interpolate between points with the same index.

     """
    assert len(ls1.coords) == len(ls2.coords), f"{len(ls1.coords)=} != {len(ls2.coords)=}"
    assert 0 <= percent <= 1, f"Percent must be between 0 and 1, got {percent}"
    if percent == 0:
        return ls1
    if percent == 1:
        return ls2

    if align_loops:
        # assuming each linestring forms a loop ...
        # shift the indexes of the points in ls2, so that the start of ls2 is close to the start of ls1.
        start_closest_to = shapely.Point(ls1.coords[0])
        distances = [start_closest_to.distance(shapely.Point(pt)) for pt in ls2.coords]
        start_index = np.argmin(distances)
        if start_index != 0:
            # print(f"Realigning linestring two by starting at {start_index=}")
            ls2 = [shapely.Point(pt) for pt in ls2.coords]
            ls2 = ls2[start_index:] + ls2[:start_index]
            ls2 = shapely.LineString(ls2)

        # Check if both linestrings start going in the same direction.
        # If not, reverse ls2.
        vec1 = (ls1.coords[0][0] - ls1.coords[1][0], ls1.coords[0][1] - ls1.coords[1][1])
        vec2 = (ls2.coords[0][0] - ls2.coords[1][0], ls2.coords[0][1] - ls2.coords[1][1])
        if np.dot(vec1, vec2) < 0:
            ls2 = ls2.reverse()

    new = []
    for pt1, pt2 in zip(ls1.coords, ls2.coords):
        pt = shapely.Point(pt1[0] + percent * (pt2[0] - pt1[0]), pt1[1] + percent * (pt2[1] - pt1[1]))
        new.append(pt)
    return shapely.geometry.LineString(new)


def offset_inner(ls: shapely.geometry.LinearRing | shapely.geometry.LineString,
                 distance: float | int) -> shapely.geometry.LineString:
    geom_p = shapely.Polygon(ls)
    inner = shapely.offset_curve(ls, distance=distance)
    if all(geom_p.contains(shapely.Point(*pt)) for pt in inner.coords):
        return inner
    inner = shapely.offset_curve(ls, distance=-distance)
    if all(geom_p.contains(shapely.Point(*pt)) for pt in inner.coords):
        return inner
    return None
    # raise AssertionError("Something went wrong")

def make_hex_grid_of_circles_inside_polygon(poly: shapely.Polygon, circle_radius: float, spacing: float,
                                            y_offset: float = 0, x_offset: float = 0) -> list[shapely.Polygon]:
    """

    :param poly:
    :param circle_radius:
    :param spacing: # distance from cavern edge to edge
    :return:
    """
    x1 = min(poly.exterior.xy[0])
    y1 = min(poly.exterior.xy[1])
    x2 = max(poly.exterior.xy[0])
    y2 = max(poly.exterior.xy[1])
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
            circ = make_circle(shapely.Point(x_circ, y), radius=circle_radius, n=10)
            if poly.contains(circ):
                grid.append(circ)
    return grid


def yield_hex_grids_of_circles_inside_polygon(poly: shapely.Polygon, circle_radius: float, spacing: float,
                                                        stepsize: int | float = 100) -> Iterator[List[shapely.Polygon]]:

    for y_offset in np.arange(0, spacing + circle_radius, stepsize):
        for x_offset in np.arange(0, spacing + circle_radius, stepsize):
            hex_grid = make_hex_grid_of_circles_inside_polygon(poly,
                                                               circle_radius=circle_radius,
                                                               spacing=spacing,
                                                               y_offset=y_offset,
                                                               x_offset=x_offset)
            yield hex_grid

def grid_search_best_hex_grid_of_circles_inside_polygon(poly: shapely.Polygon, circle_radius: float, spacing: float,
                                                        stepsize: int | float = 100) -> list[shapely.Polygon]:
    print(f"Searching for best hexagonal grid with {stepsize=}")
    best_number = 0
    best_hex_grid = list()

    for hex_grid in yield_hex_grids_of_circles_inside_polygon(poly, circle_radius, spacing, stepsize):
        if len(hex_grid) > best_number:
            best_number = len(hex_grid)
            best_hex_grid = hex_grid

    if best_number == 1:
        # Put the cavern in the middle
        center = poly.centroid
        circ = make_circle((center.x, center.y), radius=circle_radius)
        best_hex_grid = [circ]

    if not best_hex_grid:
        raise AssertionError('No best hex grid found')
    return best_hex_grid


def make_circle(center: shapely.Point | tuple[float, float], radius: float, n=10) -> shapely.Polygon:
    center = shapely.Point(center)
    l = list()
    for i in range(n):
        angle = i * 2 * math.pi / n
        x_offset = math.cos(angle) * radius
        y_offset = math.sin(angle) * radius
        l.append(shapely.Point(center.x + x_offset, center.y + y_offset))
    return shapely.Polygon(l)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    circle = make_circle(shapely.Point(0, 0), 1, 1000)
    assert (circle.area - math.pi * 1 * 1) < 10e-4

    poly = shapely.Polygon([[-10, -10], [-10, 10],[10, 10], [10, -10]])

    grid = make_hex_grid_of_circles_inside_polygon(poly, circle_radius=2, spacing=0)

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.plot(poly.boundary.xy[0], poly.boundary.xy[1], 'k')
    for g in grid:
        ax.plot(g.boundary.xy[0], g.boundary.xy[1])
    plt.axis('scaled')  # same x and y scaling
    plt.show()

    grid = grid_search_best_hex_grid_of_circles_inside_polygon(poly, circle_radius=2, spacing=0, stepsize=0.5)

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.plot(poly.boundary.xy[0], poly.boundary.xy[1], 'k')
    for g in grid:
        ax.plot(g.boundary.xy[0], g.boundary.xy[1])
    plt.axis('scaled')  # same x and y scaling
    plt.show()