
import Geometry3D as geo

#salt dome parameters:
top_of_salt = -500
bottom_of_salt = -18000

top_allowed_cavern = -900
bottom_allowed_cavern = -5500

salt_cavern_diameter = 200
inter_cavern_spacing = salt_cavern_diameter * 4  # distance from cavern edge to edge

edge_of_salt_to_cavern = salt_cavern_diameter * 2

min_cavern_height = 573
max_cavern_height = 1146

domes = [
    {"type": "cylinder", "radius": 2500, "height": -60000, "angle_degrees": 0, "inner_offset": 400,
     "top_cutoff_depth": -900, "bottom_cutoff_depth": -5500,},
    {"type": "cylinder", "radius": 2500, "height": -60000, "angle_degrees": 0, "inner_offset": 400,
     "top_cutoff_depth": -900, "bottom_cutoff_depth": -5500, },
]