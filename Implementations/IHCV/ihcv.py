import sys
sys.path.append("../")
from TSP_Utilities import tsp_utility_functions as tsp

import numpy as np, time, pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
np.set_printoptions(threshold=np.inf, linewidth=np.get_printoptions()['linewidth'])

#============================================================IHCV Helper Functions============================================================
# Compute Distance Matrix - Euclidean Distances between each Node
def compute_distance_matrix(coordinate_array):
    distance_matrix = cdist(coordinate_array, coordinate_array, metric='euclidean')
    
    return distance_matrix

#============================================================IHCV Algorithm============================================================
# Compute the initial Convex Hull to be built upon
def compute_convex_hull(nodes):
    coordinate_list = list(zip(nodes['X'], nodes['Y']))
    coordinate_array = np.array(coordinate_list) # Coordinates as Numpy Array
    convex_hull = ConvexHull(coordinate_array)

    return coordinate_array, convex_hull

# Take the Convex Hull, and iteratively find the "cheapest" insertion of each remaining node
def cheapest_insertion(coordinate_array, convex_hull, distance_matrix, nodes):
    subtour_indices = list(convex_hull.vertices)
    not_in_subtour = []
    for n in range(len(nodes)):
        if n not in subtour_indices:
            not_in_subtour.append(n)

    while len(not_in_subtour) > 0:
        lowest_cost = np.inf
        for node in not_in_subtour:
            # Go through the current subtour and try to insert nodes from not_in_subtour in all postions to see where the "cheapest" insertion for that node is
            for i in range(len(subtour_indices)):
                previous_node = subtour_indices[i]
                next_node = subtour_indices[(i + 1) % len(subtour_indices)]
                cost = distance_matrix[previous_node, node] + distance_matrix[node, next_node] - distance_matrix[previous_node, next_node]

                if cost < lowest_cost:
                    lowest_cost = cost
                    optimal_insertion = (node, i + 1)

        new_node, position = optimal_insertion
        subtour_indices.insert(position, new_node)
        not_in_subtour.remove(new_node)

    # Calculate total distance travelled in route
    subtour_indices.append(subtour_indices[0])
    route = coordinate_array[subtour_indices]
    difference = np.diff(route, axis=0)
    distances = np.sqrt(np.sum(np.square(difference), axis= 1))
    distance_travelled = np.sum(distances)

    # Return route w/ node numbers instead of indices
    numbered_route = list(nodes['Node'].iloc[subtour_indices])

    # Reorder route to begin with start node
    start_node_row = nodes[nodes["Type"] == "Start"]
    start_node = start_node_row["Node"].iloc[0]
    
    start_node_index = numbered_route.index(start_node)

    numbered_route.pop()
    reordered_route = numbered_route[start_node_index:] + numbered_route[:start_node_index] + [numbered_route[start_node_index]]


    # print(len(reordered_route))

    return reordered_route, np.round(distance_travelled, 2)

#============================================================Run Algorithm============================================================
def run_ihcv_generic(file_name, import_node_data_func, display_route=True, name="Insertion Heuristic w/ Convex Hull"):
    nodes = import_node_data_func(file_name)
    coordinate_array, hull = compute_convex_hull(nodes)
    distance_matrix = compute_distance_matrix(coordinate_array)
    # print(distance_matrix.tolist())
    solution = cheapest_insertion(coordinate_array, hull, distance_matrix, nodes)
    if display_route:
        tsp.display_route_as_graph(nodes, solution, name)

    return solution, nodes

def run_ihcv(file_name, display_route=True):
    return run_ihcv_generic(file_name, tsp.import_node_data, display_route)[0]

def run_ihcv_tsp_lib(file_name, display_route=True):
    spreadsheet_name = tsp.convert_tsp_lib_instance_to_spreadsheet(file_name)
    return run_ihcv_generic(spreadsheet_name, tsp.import_node_data_tsp_lib, display_route)[0]

def run_ihcv_initial_solution(file_name, display_route=True):
    return run_ihcv_generic(file_name, tsp.import_node_data, display_route)

def run_ihcv_tsp_lib_initial_solution(file_name, display_route=True):
    spreadsheet_name = tsp.convert_tsp_lib_instance_to_spreadsheet(file_name)
    return run_ihcv_generic(spreadsheet_name, tsp.import_node_data_tsp_lib, display_route)


# print(run_ihcv_tsp_lib("tsp225.tsp"))
# print(run_ihcv_tsp_lib("berlin52.tsp"))
# print(run_ihcv("test_city_1.xlsx"))
