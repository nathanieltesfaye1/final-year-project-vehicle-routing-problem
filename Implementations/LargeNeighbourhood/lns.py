import sys, numpy as np
sys.path.append("../")
from TSP_Utilities import tsp_utility_functions as tsp
from NearestNeighbour import nearest_neighbour as nn
from IHCV import ihcv as ih
from TwoOpt import two_opt as to

PROPORTION = 0.15
MAX_ITERATIONS = 100000
IMPROVEMENT_THRESHOLD = 0.000001

#==================================================Large Neighbourhood Search Algorithm==================================================
def destroy(route, proportion=PROPORTION):
    route_array = np.array(route)
    n_nodes_to_remove = np.maximum(1, int(len(route_array) * proportion))

    removable_indices = np.arange(1, len(route_array) - 1) # All nodes except the start node can be a destroyed node
    indices_to_remove = np.random.choice(removable_indices, size = n_nodes_to_remove, replace = False)
    mask_removed = np.full(len(route_array), True, dtype=bool)
    mask_removed[indices_to_remove] = False

    remaining_route = (route_array[mask_removed]).tolist()
    removed_nodes = (route_array[indices_to_remove]).tolist()

    return remaining_route, removed_nodes

def repair(remaining_route, removed_nodes, distance_matrix, node_index_mapping):
    remaining_route_array = np.array(remaining_route)

    for node in removed_nodes:
        optimal_position = None
        min_cost_increase = np.inf

        node_index = node_index_mapping[node]
        costs_to_node = distance_matrix[:, node_index] # Retrieve the column corresponding to node_index inside the dm - all distances to the node at node_index
        costs_from_node = distance_matrix[node_index, :] # Retrieve the row, hence gets all distances from the node

        for i in range(1, len(remaining_route_array)):
            cost_increase = ((costs_to_node[node_index_mapping[remaining_route_array[i - 1]]]) + (costs_from_node[node_index_mapping[remaining_route_array[i]]]) -
                             (distance_matrix[node_index_mapping[remaining_route_array[i - 1]], node_index_mapping[remaining_route_array[i]]]))

            if cost_increase < min_cost_increase:
                min_cost_increase = cost_increase
                optimal_position = i

        remaining_route_array = (np.insert(remaining_route_array, optimal_position, node)).tolist()

    return remaining_route_array

def lns(initial_route, distance_matrix, node_index_mapping, max_iterations=MAX_ITERATIONS, improvement_threshold=IMPROVEMENT_THRESHOLD):
    if improvement_threshold >= 1:
        raise ValueError("improvement_threshold must be < 1")
                         
    shortest_route = initial_route
    shortest_distance = tsp.compute_route_distance(shortest_route, distance_matrix, node_index_mapping)
    n_iterations_without_improvement = 0 # Tracks the number of iterations w/out improvement in solution

    for _ in range(max_iterations):
        remaining_route, removed_nodes = destroy(shortest_route)
        possible_shortest_route = repair(remaining_route, removed_nodes, distance_matrix, node_index_mapping)
        possible_shortest_distance = tsp.compute_route_distance(possible_shortest_route, distance_matrix, node_index_mapping)
        
        if possible_shortest_distance < (shortest_distance * (1 - improvement_threshold)):
            shortest_route = possible_shortest_route
            shortest_distance = possible_shortest_distance
            n_iterations_without_improvement = 0

        else:
            n_iterations_without_improvement += 1
        
        if n_iterations_without_improvement > (max_iterations * 0.1):  # If the number of iterations w/out improvement exceeds 10% of the maximum no. of iterations, then algorithm terminates early
            break
    
    return shortest_route, np.round(shortest_distance, 2)

#============================================================Run Algorithm============================================================
def run_lns_generic(initial_solution_func, file_name, name, display_route=True):
    initial_solution, nodes = initial_solution_func(file_name, False)
    initial_route = initial_solution[0]
    distance_matrix = tsp.compute_distance_matrix(nodes)
    node_index_mapping = tsp.map_nodes_to_index(nodes)
    solution = lns(initial_route, distance_matrix, node_index_mapping)
    if display_route:
        tsp.display_route_as_graph(nodes, solution, name)

    return solution

# NN --> LNS
def run_lns_nearest_neighbour(file_name, display_route=True):
    return run_lns_generic(nn.run_nearest_neighbour_initial_solution, file_name, "Nearest Neighbour --> Large Neighbourhood Search", display_route)

# IHCV --> LNS
def run_lns_ihcv(file_name, display_route=True):
    return run_lns_generic(ih.run_ihcv_initial_solution, file_name, "Insertion Heuristic w/ Convex Hull --> Large Neighbourhood Search", display_route)

# NN --> LNS
def run_lns_nearest_neighbour_tsp_lib(file_name, display_route=True):
    return run_lns_generic(nn.run_nearest_neighbour_tsp_lib_initial_solution, file_name, "Nearest Neighbour --> Large Neighbourhood Search", display_route)

# IHCV --> LNS
def run_lns_ihcv_tsp_lib(file_name, display_route=True):
    return run_lns_generic(ih.run_ihcv_tsp_lib_initial_solution, file_name, "Insertion Heuristic w/ Convex Hull --> Large Neighbourhood Search", display_route)

# NN --> 2-Opt --> LNS
def run_lns_two_opt_nearest_neighbour(file_name, display_route=True):
    return run_lns_generic(to.run_two_opt_nearest_neighbour_initial_solution, file_name, "Nearest Neighbour --> Two-Opt --> Large Neighbourhood Search", display_route)

# IHCV --> 2-Opt --> LNS
def run_lns_two_opt_ihcv(file_name, display_route=True):
    return run_lns_generic(to.run_two_opt_ihcv_initial_solution, file_name, "Insertion Heuristic w/ Convex Hull --> Two-Opt --> Large Neighbourhood Search", display_route)

# NN --> 2-Opt --> LNS
def run_lns_two_opt_nearest_neighbour_tsp_lib(file_name, display_route=True):
    return run_lns_generic(to.run_two_opt_nearest_neighbour_tsp_lib_initial_solution, file_name, "Nearest Neighbour --> Two-Opt --> Large Neighbourhood Search", display_route)

# IHCV --> 2-Opt --> LNS
def run_lns_two_opt_ihcv_tsp_lib(file_name, display_route=True):
    return run_lns_generic(to.run_two_opt_ihcv_tsp_lib_initial_solution, file_name, "Insertion Heuristic w/ Convex Hull --> Two-Opt --> Large Neighbourhood Search", display_route)


# print(run_lns_ihcv_tsp_lib("pr76.tsp"))
# print(run_lns_ihcv_tsp_lib("berlin52.tsp"))
# run_lns_two_opt_ihcv_tsp_lib("berlin52.tsp")
# run_lns_two_opt_nearest_neighbour_tsp_lib("berlin52.tsp")
# run_lns_ihcv_tsp_lib("tsp225.tsp")

# nodes = tsp.import_node_data("test_city_1.xlsx")
# print(nodes)
# node_to_retrieve = 1
# node_index = nodes.index[nodes["Node"] == node_to_retrieve][0]
# print(node_index)

# print(run_lns_nearest_neighbour("test_city_1.xlsx"))

# print(run_lns_nearest_neighbour_tsp_lib("tsp225.tsp"))

# print(run_lns_two_opt_ihcv_tsp_lib("berlin52.tsp", False))