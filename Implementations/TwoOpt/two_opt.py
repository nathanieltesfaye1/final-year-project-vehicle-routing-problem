import sys, numpy as np
sys.path.append("../")
from TSP_Utilities import tsp_utility_functions as tsp

from NearestNeighbour import nearest_neighbour as nn
from IHCV import ihcv as ih

PROPORTION = 1
MIN_NEIGHBOURS = 5

#==================================================Two-Opt Algorithm==================================================
# Remember to come back to this after lunch and optimise using numpy if possible...
def two_opt_swap_move(route, i, j):
    route_array = np.array(route)
    route_array[i:j + 1] = route_array[i:j + 1][::-1]  # inline swap
    swapped_route = route_array.tolist()

    return swapped_route

# Pruning search space through use of nearest neighbours allows us to speed up k-opt, according to Steiglitz & Weiner in the paper "TSP Heuristics - Nillson 2003"
"""Proportion argument allows us to dynamically adjust the number of nearest neighbours. Min_Neighbours sets a minimum threshold for this number. 
This ensures a minimum level of exploration, regardless of size of inputted dataset. Adjusting the Proportion Argument allows us to decide on
the trade-off between runtime efficiency & solution quality. Higher proportion generally = more accurate solution + slower runtime, and vice versa.
"""
def compute_nearest_neighbours(distance_matrix, proportion=PROPORTION, min_neighbours=MIN_NEIGHBOURS): 
    n_nodes = len(distance_matrix)
    n = max(int(proportion * n_nodes), min_neighbours)
    nearest_neighbours = []

    for i in range(n_nodes):
        distances = distance_matrix[i, :] # Get all elements in the i'th row
        neighbours_indexed = np.argsort(distances) 
        nearest_neighbours.append(neighbours_indexed[1:n + 1].tolist()) # Get the "n" nearest neighbours to the node

    return nearest_neighbours

def two_opt(initial_solution, distance_matrix, nearest_neighbours, node_index_mapping):
    initial_route = initial_solution[0]
    initial_distance = initial_solution[1]
    route = initial_route
    shortest_distance = initial_distance

    while True:
        no_improvement_found = True   # Assume we won't find an improvement in this 2-opt iteration

        for i in range(1, len(route) - 2):
            for j in nearest_neighbours[node_index_mapping[route[i]]]:  # We only consider the nearest neighbours for swapping! 
                i_index = node_index_mapping[route[i - 1]]
                j_index = node_index_mapping[route[j]]
                next_node_index = node_index_mapping[route[(j + 1) % len(route)]]

                current_distance = distance_matrix[i_index, node_index_mapping[route[i]]] + distance_matrix[j_index, next_node_index]
                swapped_distance = distance_matrix[i_index, j_index] + distance_matrix[node_index_mapping[route[i]], next_node_index]

                if current_distance > swapped_distance:  # We then prune the search space by only proceeding to swapping if there's potential for the swap to lead to improvement in decreasing total distance!
                    swapped_route = two_opt_swap_move(route, i, j % len(route))
                    swapped_route_distance = sum([distance_matrix[node_index_mapping[swapped_route[k]], node_index_mapping[swapped_route[k + 1]]] for k in range(len(swapped_route) - 1)])

                    if swapped_route_distance < shortest_distance:
                        route = swapped_route
                        shortest_distance = swapped_route_distance
                        no_improvement_found = False  # Improvement found!...
                        break  # ...therefore, we'll break out this loop.

            if no_improvement_found == False:  # Resume outer loop because we found improvement
                break

        if no_improvement_found:  # Didn't find an improvement, so we stop
            break

    return route, np.round(shortest_distance, 2)


#==================================================Run Algorithm==================================================
def run_two_opt_generic(initial_solution_func, file_name, name="Insertion Heuristic w/ Convex Hull --> Two-Opt", display_route=True):
    initial_solution, nodes = initial_solution_func(file_name, False)
    distance_matrix = tsp.compute_distance_matrix(nodes)
    nearest_neighbours = compute_nearest_neighbours(distance_matrix)
    node_index_mapping = tsp.map_nodes_to_index(nodes)  
    solution = two_opt(initial_solution, distance_matrix, nearest_neighbours, node_index_mapping)
    if display_route:
        tsp.display_route_as_graph(nodes, solution, name)
        
    return solution, nodes

def run_two_opt_nearest_neighbour(file_name, display_route=True):
    return run_two_opt_generic(nn.run_nearest_neighbour_initial_solution, file_name, "Nearest Neighbour --> Two-Opt", display_route)[0]

def run_two_opt_ihcv(file_name, display_route=True):
    return run_two_opt_generic(ih.run_ihcv_initial_solution, file_name, "Insertion Heuristic w/ Convex Hull --> Two-Opt", display_route)[0]

def run_two_opt_nearest_neighbour_tsp_lib(file_name, display_route=True):
    return run_two_opt_generic(nn.run_nearest_neighbour_tsp_lib_initial_solution, file_name, "Nearest Neighbour --> Two-Opt", display_route)[0]

def run_two_opt_ihcv_tsp_lib(file_name, display_route=True):
    return run_two_opt_generic(ih.run_ihcv_tsp_lib_initial_solution, file_name, "Insertion Heuristic w/ Convex Hull --> Two-Opt", display_route)[0]


def run_two_opt_nearest_neighbour_initial_solution(file_name, display_route=True):
    return run_two_opt_generic(nn.run_nearest_neighbour_initial_solution, file_name, "Nearest Neighbour --> Two-Opt", display_route)

def run_two_opt_ihcv_initial_solution(file_name, display_route=True):
    return run_two_opt_generic(ih.run_ihcv_initial_solution, file_name, "Insertion Heuristic w/ Convex Hull --> Two-Opt", display_route)

def run_two_opt_nearest_neighbour_tsp_lib_initial_solution(file_name, display_route=True):
    return run_two_opt_generic(nn.run_nearest_neighbour_tsp_lib_initial_solution, file_name, "Nearest Neighbour --> Two-Opt", display_route)

def run_two_opt_ihcv_tsp_lib_initial_solution(file_name, display_route=True):
    return run_two_opt_generic(ih.run_ihcv_tsp_lib_initial_solution, file_name, "Insertion Heuristic w/ Convex Hull --> Two-Opt", display_route)



# print(run_two_opt_ihcv_tsp_lib("tsp225.tsp")) #4329
# print(run_two_opt_ihcv_tsp_lib("ulysses16.tsp"))
# print(run_two_opt_ihcv_tsp_lib("berlin52.tsp"))
# print(run_two_opt_nearest_neighbour_tsp_lib("tsp225.tsp"))
# print(run_two_opt_ihcv_tsp_lib("pr76.tsp"))
# print(run_two_opt_nearest_neighbour_tsp_lib("pr76.tsp"))
# print(run_two_opt_nearest_neighbour("test_city_1.xlsx"))