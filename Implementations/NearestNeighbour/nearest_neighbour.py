import sys
sys.path.append("../")
from TSP_Utilities import tsp_utility_functions as tsp

import numpy as np, time

#==================================================Nearest Neighbour Algorithm==================================================
def nearest_neighbour(nodes, start_node_index=None):
    n_nodes = len(nodes)
    distance_matrix = tsp.compute_distance_matrix(nodes)
    # print(f"\n{distance_matrix}\n")

    start_node_index = nodes[nodes["Type"] == "Start"].index[0]

    visited = np.full(n_nodes, False)
    route_indexed = [start_node_index]
    current_node = start_node_index
    distance_travelled = 0

    while len(route_indexed) < n_nodes:
        visited[current_node] = True
        distances = distance_matrix[current_node]
        
        mask_distances = np.where(visited, np.inf, distances) # If (visited) == true, set corresponding value in distances to np.inf. If (visited) == false, keep corresponding value in distances. We're doing this so that we can select the nearest distance from the current node that hasn't been visited yet. This is an efficient way to allow us to update the array/matrix values without modifying/making copies of the original, because it is temporary and will be set back later. This is the process of 'masking'.

        nearest_node = np.argmin(mask_distances)
        nearest_distance = mask_distances[nearest_node]

        distance_travelled += nearest_distance
        route_indexed.append(nearest_node)
        current_node = nearest_node

    distance_travelled += distance_matrix[current_node][start_node_index]
    route_indexed.append(start_node_index)

    route = [nodes.iloc[n]["Node"] for n in route_indexed]

    return route, np.round(distance_travelled, 2)

#==================================================Run Algorithm==================================================
def run_nearest_neighbour_generic(file_name, import_node_data_func, display_route=True, name="Nearest Neighbour"):
    nodes = import_node_data_func(file_name)
    solution = nearest_neighbour(nodes)
    if display_route:
        tsp.display_route_as_graph(nodes, solution, name)

    return solution, nodes

def run_nearest_neighbour(file_name, display_route=True):
    return run_nearest_neighbour_generic(file_name, tsp.import_node_data, display_route)[0]

def run_nearest_neighbour_tsp_lib(file_name, display_route=True):
    spreadsheet_name = tsp.convert_tsp_lib_instance_to_spreadsheet(file_name)
    return run_nearest_neighbour_generic(spreadsheet_name, tsp.import_node_data_tsp_lib, display_route)[0]

def run_nearest_neighbour_initial_solution(file_name, display_route=True):
    return run_nearest_neighbour_generic(file_name, tsp.import_node_data, display_route)

def run_nearest_neighbour_tsp_lib_initial_solution(file_name, display_route=True):
    spreadsheet_name = tsp.convert_tsp_lib_instance_to_spreadsheet(file_name)
    return run_nearest_neighbour_generic(spreadsheet_name, tsp.import_node_data_tsp_lib, display_route)


# print(run_nearest_neighbour_tsp_lib("tsp225.tsp"))
# print(run_nearest_neighbour("test_city_1.xlsx"))
# print(run_nearest_neighbour_tsp_lib('berlin52.tsp'))