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
