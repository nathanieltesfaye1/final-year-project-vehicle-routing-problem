import sys
sys.path.append("../")
from TSP_Utilities import tsp_utility_functions as tsp
from TwoOpt import two_opt as to
from NearestNeighbour import nearest_neighbour as nn

import numpy as np, matplotlib.pyplot as plt
import time

#==============================GA Params==============================
POPULATION_SIZE = 100
GENERATIONS = 20
MUTATION_RATE = 0.35
CROSSOVER_RATE = 0.7
# POPULATION_SIZE = 250
# GENERATIONS = 80
# MUTATION_RATE = 0.35
# CROSSOVER_RATE = 0.7
# POPULATION_SIZE = 750
# GENERATIONS = 500
# MUTATION_RATE = 0.35
# CROSSOVER_RATE = 0.65

#==========Further Params==========
STAGNATION_TOLERANCE = 0.01
TWO_OPT_APPLY_THRESHOLD = 15
NN_FRACTION = 0.2
ELITISM_THRESHOLDS = {
    10: 0.5,  # For TSP instance of 10 nodes = preserve top 50%. This top 50% will be considered the "Elite" Individuals
    30: 0.35,  # For 30 nodes = preserve top 35%
    50: 0.3,  # For 50 nodes = preserve top 25%
    100: 0.25,  # For 100 nodes = preserve top 15%
    250: 0.08,  # For 250 nodes = preserve top 8%
    500: 0.05,  # For 500 nodes = preserve top 5%
    np.inf: 0.01  # For 500+ nodes = preserve top 1%
}

#==============================GA Algorithm Design==============================
#==========Setup Individuals==========
def generate_route(n_nodes):
    # A nice way of randomising route using Numpy...
    unvisited = np.arange(2, n_nodes + 1)
    np.random.shuffle(unvisited)
    route = np.concatenate(([1], unvisited, [1]))

    return route

def encode_individual(route, n_nodes):
    individual = np.zeros((n_nodes, n_nodes))

    for i in range(len(route) - 1):
        start_index = route[i] - 1   # Adjust index because we're now working w/ Numpy
        end_index = route[i + 1] - 1
        individual[start_index, end_index] = 1  # Link strat node to end node

    # print(individual)

    return individual

def generate_individual(n_nodes):
    route = generate_route(n_nodes)
    individual_encoded = encode_individual(route, n_nodes)
    
    return (individual_encoded, route)  

def spawn_new_individuals(population, n_new_individuals, n_nodes):
    replace_indices = np.random.choice(len(population), n_new_individuals, replace=False)

    for i in replace_indices:
        population[i] = generate_individual(n_nodes)
    
    return population

#==========Generate Population==========
def initialise_population(nodes, n_nodes):
    nn_population_size = int(POPULATION_SIZE * NN_FRACTION)
    nn_population = []

    for _ in range(nn_population_size):
        start_node_index = np.random.choice(np.arange(1, n_nodes))
        route, _ = nn.nearest_neighbour(nodes, start_node_index)
        individual_encoded = encode_individual(route, n_nodes)
        nn_population.append((individual_encoded, np.array(route)))

    random_population = [generate_individual(n_nodes) for _ in range(POPULATION_SIZE - nn_population_size)]
    random_population = [(np.array(individual[0]), np.array(individual[1])) for individual in random_population]

    population = nn_population + random_population

    return population

def compute_fitness(individual_route, distance_matrix):
    individual_route = np.array(individual_route)
    individual_route_adjusted = individual_route - 1 # for Numpy
    distances = distance_matrix[individual_route_adjusted[:-1], individual_route_adjusted[1:]]
    
    travelled_distance = np.sum(distances)
    
    return travelled_distance

def compute_population_fitness(population, distance_matrix):
    fitness_values = []
    fitness_values = [compute_fitness(route, distance_matrix) for _, route in population]

    return fitness_values

def sort_population(population, fitness_values):
    population_fitness_values = list(zip(population, fitness_values))
    population_fitness_values.sort(key = lambda x: x[1])
  
    sorted_population = [i[0] for i in population_fitness_values]
    sorted_fitness_values = [i[1] for i in population_fitness_values]

    return sorted_population, sorted_fitness_values

#==============================GA Operators==============================
#==========Select Parents==========
def select_parent_binary_tournament(population, distance_matrix):
    population_size = len(population)

    candidate_indices = np.random.choice(population_size, 2, replace=False)
    candidate_parent_1 = population[candidate_indices[0]]
    candidate_parent_2 = population[candidate_indices[1]]

    parent_1_fitness = compute_fitness(candidate_parent_1[1], distance_matrix)
    parent_2_fitness = compute_fitness(candidate_parent_2[1], distance_matrix)

    if parent_1_fitness < parent_2_fitness:
        return candidate_parent_1
    else:
        return candidate_parent_2

#==========Crossover==========
def fill_chromosome(offspring, parent, crossover_point_1, crossover_point_2):
    offspring_to_fill = np.where(offspring == -1)[0]
    parent_genes_not_in_offspring = parent[~np.isin(parent, offspring[crossover_point_1:crossover_point_2])]

    offspring[offspring_to_fill] = parent_genes_not_in_offspring[:offspring_to_fill.size]

# Specifically, this is an "Order Crossover"
def crossover(parent_1, parent_2):
    parent_1 = np.array(parent_1)
    parent_2 = np.array(parent_2)
    length = parent_1.size
    
    crossover_point_1, crossover_point_2 = np.sort(np.random.choice(np.arange(1, length - 1), 2, replace=False))

    # Placeholders for offspring_1 & 2's Genes before Parents Crossover
    offspring_1 = np.full(length, -1)
    offspring_2 = np.full(length, -1)

    # Copy corresponding Crossover Segment from Parents to Offspring
    offspring_1[crossover_point_1:crossover_point_2] = parent_1[crossover_point_1:crossover_point_2]
    offspring_2[crossover_point_1:crossover_point_2] = parent_2[crossover_point_1:crossover_point_2]
    
    # Fill out empty Genes in offspring_1's Chromosome w/ Genes from parent_2
    fill_chromosome(offspring_1, parent_2, crossover_point_1, crossover_point_2)
    # Fill out empty Genes in offspring_2's Chromosome w/ Genes from parent_1
    fill_chromosome(offspring_2, parent_1, crossover_point_1, crossover_point_2)

    offspring_1[-1] = offspring_1[0]
    offspring_2[-1] = offspring_2[0]

    return offspring_1.tolist(), offspring_2.tolist()

#==========Mutation==========
def is_stagnant(generations_best_fitness_values, generation):
    stagnation_check_percentage = 0
    if GENERATIONS <= 100:
        stagnation_check_percentage = 0.5  # Check the last 50% of generations for small generation counts
    elif GENERATIONS <= 500:
        stagnation_check_percentage = 0.3  # 30% for medium
    else:
        stagnation_check_percentage = 0.2  # 20% for larger generation counts

    stagnation_check_span = int(GENERATIONS * stagnation_check_percentage)
    if generation < stagnation_check_span or len(generations_best_fitness_values) < stagnation_check_span:
        return False

    recent_fitness_values = generations_best_fitness_values[-stagnation_check_span:]
    if recent_fitness_values[0] != 0:
        improvement = (recent_fitness_values[0] - recent_fitness_values[-1])/(recent_fitness_values[0])
    else:
        improvement = 0

    return improvement < STAGNATION_TOLERANCE

# An "Adaptive Mutation" Approach has been taken here; mutation rate may change over time. The idea of this is to mitigate the possibility of converging to local optima
def mutation(individual_route, generation, generations_best_fitness_values, n_nodes):
    individual_route = np.array(individual_route)
    stagnant = is_stagnant(generations_best_fitness_values, generation)

    if stagnant:
        base_rate = MUTATION_RATE * 2
    else:
        base_rate = MUTATION_RATE

    adaptive_rate = base_rate * (1 - generation /GENERATIONS)

    if np.random.rand() < adaptive_rate:
        mutation_point1, mutation_point2 = np.random.choice(np.arange(1, n_nodes), 2, replace=False)
        individual_route[mutation_point1], individual_route[mutation_point2] = individual_route[mutation_point2], individual_route[mutation_point1]
    
    return individual_route.tolist()

#==========Two Opt for further Optimisation==========
def should_apply_two_opt(generations_best_fitness_values):
    if len(generations_best_fitness_values) < TWO_OPT_APPLY_THRESHOLD:
        return False

    recent_best_value = generations_best_fitness_values[-TWO_OPT_APPLY_THRESHOLD:]
    if recent_best_value[-1] == recent_best_value[0]:
        return True  # no improvement - apply two opt

    return False

def apply_two_opt(route, distance_matrix, nearest_neighbours, nodes):
    optimised_route, distance_travelled = to.two_opt((route, compute_fitness(route, distance_matrix)), distance_matrix, nearest_neighbours, tsp.map_nodes_to_index(nodes))

    return optimised_route

#==========Generate Next Generation==========
def compute_elitism_count(n_nodes, population_size):
    for threshold, percentage in ELITISM_THRESHOLDS.items():
        if n_nodes <= threshold:
            return int(population_size * percentage)

def generate_next_generation(sorted_population, generation, generations_best_fitness_values, n_nodes, distance_matrix, nearest_neighbours, nodes):
    new_population = []
    
    # Elitism
    elitism_count = compute_elitism_count(n_nodes, POPULATION_SIZE)
    elite_individuals = sorted_population[:elitism_count]
    new_population.extend(elite_individuals)

    while len(new_population) < POPULATION_SIZE:
        parent_1, parent_1_route = select_parent_binary_tournament(sorted_population, distance_matrix)
        parent_2, parent_2_route = select_parent_binary_tournament(sorted_population, distance_matrix)

        while np.array_equal(parent_1_route, parent_2_route): # If they happen to be equal, then change parent_2
            parent_2, parent_2_route = select_parent_binary_tournament(sorted_population, distance_matrix)

        if np.random.rand() < CROSSOVER_RATE:
            offspring_1_route, offspring_2_route = crossover(parent_1_route, parent_2_route)
        else:
            offspring_1_route, offspring_2_route = parent_1_route, parent_2_route

        if should_apply_two_opt(generations_best_fitness_values):
            offspring_1_route = apply_two_opt(offspring_1_route, distance_matrix, nearest_neighbours, nodes)
            offspring_2_route = apply_two_opt(offspring_2_route, distance_matrix, nearest_neighbours, nodes)

        offspring_1_route = mutation(offspring_1_route, generation, generations_best_fitness_values, n_nodes)
        offspring_2_route = mutation(offspring_2_route, generation, generations_best_fitness_values, n_nodes)

        offspring_1 = encode_individual(offspring_1_route, n_nodes)
        offspring_2 = encode_individual(offspring_2_route, n_nodes)

        new_population.append((offspring_1, offspring_1_route))

        if len(new_population) < POPULATION_SIZE:
            new_population.append((offspring_2, offspring_2_route))

    return new_population

#==============================GA Run/Performance Tracking==============================
generations_best_fitness_value = np.inf
generations_best_route = None
generations_best_fitness_values = []
generations_average_fitness_values = []

#==========Run GA - Functions==========
def run_ga_generic(file_name, import_node_data_func,population_size=POPULATION_SIZE, generations=GENERATIONS, mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,display_route=True):
    global POPULATION_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE
    # Update Global Variable values
    POPULATION_SIZE = population_size
    GENERATIONS = generations
    MUTATION_RATE = mutation_rate
    CROSSOVER_RATE = crossover_rate

    nodes = import_node_data_func(file_name)

    n_nodes = len(nodes)
    distance_matrix = tsp.compute_distance_matrix(nodes)
    nearest_neighbours = to.compute_nearest_neighbours(distance_matrix)

    start_time = time.time()
    population = initialise_population(nodes, n_nodes)
    population_fitness = compute_population_fitness(population, distance_matrix)
    sorted_population, sorted_fitness_values = sort_population(population, population_fitness)

    if sorted_fitness_values[0] < generations_best_fitness_value:
        generations_best_fitness = sorted_fitness_values[0]
        generations_best_route = sorted_population[0][1]

    generations_best_fitness_values.append(generations_best_fitness)
    generations_average_fitness_values.append(sum(sorted_fitness_values) / len(sorted_fitness_values))

    stagnation_counter = 0  

    for generation in range(1, GENERATIONS):
        if generation % 50 == 0:  # Every 50 generations, introduce some new individuals into the population (and remove the POPULATION_SIZE/50 least fit individuals) in order to introduce some genetic diversity and hopefully mitigate convergence towards local optima, hence pushing towards global optimum
            sorted_population = spawn_new_individuals(sorted_population, int(POPULATION_SIZE/25), n_nodes) 
        new_population = generate_next_generation(sorted_population, generation, generations_best_fitness_values, n_nodes, distance_matrix, nearest_neighbours, nodes)
        new_population_fitness = compute_population_fitness(new_population, distance_matrix)
        sorted_new_population, sorted_new_fitness_values = sort_population(new_population, new_population_fitness)

        if sorted_new_fitness_values[0] < generations_best_fitness:
            generations_best_fitness = sorted_new_fitness_values[0]
            generations_best_route = sorted_new_population[0][1]
            stagnation_counter = 0  # Improvement - reset stagnant counter
        else:
            stagnation_counter += 1 

        generations_best_fitness_values.append(generations_best_fitness)
        generations_average_fitness_values.append(sum(sorted_new_fitness_values) / len(sorted_new_fitness_values))

        # Check if we've stagnated and therefore need to terminate prematurely
        if is_stagnant(generations_best_fitness_values, generation):
            print(f"Early termination (generation {generation}/{GENERATIONS}) due to stagnation")
            break

        sorted_population = sorted_new_population

        print(np.round(generations_best_fitness, 2))
        print(f"^{np.round((generation/GENERATIONS) * 100, 2)}% - Generation {generation}")

    end_time = time.time()
    time_elapsed = end_time - start_time

    print(f"Time Elapsed: {np.round(time_elapsed, 6)}s")
    
    if display_route:
        tsp.display_route_as_graph(nodes, (generations_best_route, generations_best_fitness), "Genetic Algorithm")

    return list(generations_best_route), np.round(generations_best_fitness, 2)

def run_ga(file_name, display_route=True):
    return run_ga_generic(file_name, tsp.import_node_data, display_route=display_route)

def run_ga_tsp_lib(file_name, display_route=True):
    spreadsheet_name = tsp.convert_tsp_lib_instance_to_spreadsheet(file_name)
    return run_ga_generic(spreadsheet_name, tsp.import_node_data_tsp_lib, display_route=display_route)

#==========Run GA - Execution==========
# print(run_ga("test_city_1.xlsx"))
# print(run_ga_tsp_lib("pr76.tsp"))
# print(run_ga_tsp_lib("berlin52.tsp"))
# print(run_ga_tsp_lib("ulysses16.tsp"))
# print(run_ga_tsp_lib("pr107.tsp"))
# print(run_ga_tsp_lib("tsp225.tsp"))
# print(run_ga_tsp_lib("pr136.tsp"))

#==========Display GA Performance Graphs==========
# plt.plot(generations_best_fitness_values)
# plt.xlabel('Generation Number')
# plt.ylabel('Best Fitness Value')
# plt.title('Best Fitness Value per Generation')
# plt.show()

# plt.plot(generations_average_fitness_values)
# plt.xlabel('Generation Number')
# plt.ylabel('Average Fitness Value')
# plt.title('Average Fitness Value per Generation')
# plt.show()