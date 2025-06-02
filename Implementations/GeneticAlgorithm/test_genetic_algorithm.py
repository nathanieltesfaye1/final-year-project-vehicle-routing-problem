import sys
sys.path.append("../")
from TSP_Utilities.Test_Inputs import mock_route_data as mock
from TSP_Utilities import tsp_utility_functions as tsp
from GeneticAlgorithm import genetic_algorithm as ga
from TwoOpt import two_opt as to

import unittest, numpy as np, copy

class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self.test_city_1_nodes = mock.get_test_city_1_nodes()
        self.berlin52_nodes = mock.get_berlin52_nodes()
        self.tsp225_nodes = mock.get_tsp225_nodes()

        self.original_population_size = ga.POPULATION_SIZE
        self.original_generations = ga.GENERATIONS
        self.original_mutation_rate = ga.MUTATION_RATE
        self.original_crossover_rate = ga.CROSSOVER_RATE

    def tearDown(self):
        ga.POPULATION_SIZE = self.original_population_size
        ga.GENERATIONS = self.original_generations
        ga.MUTATION_RATE = self.original_mutation_rate
        ga.CROSSOVER_RATE = self.original_crossover_rate

    def test_generate_route(self):
        n_nodes = 5
        route = ga.generate_route(n_nodes)
        
        self.assertEqual(len(route), n_nodes + 1)
        self.assertTrue(np.array_equal(np.sort(route[1:-1]), np.arange(2, n_nodes + 1)))

    def test_encode_individual(self):
        route = [1, 2, 3, 4, 1]
        n_nodes = 4

        individual = ga.encode_individual(route, n_nodes)
        expected_individual = np.array([[0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1],
                                        [1, 0, 0, 0]])
        
        self.assertTrue(np.array_equal(individual, expected_individual))

    def test_generate_individual(self):
        n_nodes = 5
        individual_encoded, route = ga.generate_individual(n_nodes)

        self.assertEqual(len(route), n_nodes + 1)
        self.assertEqual(individual_encoded.shape, (n_nodes, n_nodes))

    def test_spawn_new_individuals(self):
        n_nodes = 5
        population = [ga.generate_individual(n_nodes) for i in range(10)]
        n_new_individuals = 2

        new_population = ga.spawn_new_individuals(population, n_new_individuals, n_nodes)

        self.assertEqual(len(new_population), 10)

    def test_initialise_population(self):
        nodes = self.test_city_1_nodes
        n_nodes = len(nodes)

        population = ga.initialise_population(nodes, n_nodes)

        self.assertEqual(len(population), ga.POPULATION_SIZE)

    def test_compute_fitness(self):
        individual_route = [1, 2, 3, 1]
        distance_matrix = np.array([[0, 1, 2],
                                    [1, 0, 3],
                                    [2, 3, 0]])
        
        fitness = ga.compute_fitness(individual_route, distance_matrix)

        self.assertEqual(6, fitness)

    def test_compute_population_fitness(self):
        population = [([1, 2, 3, 1], [1, 3, 2, 1])]
        distance_matrix = np.array([[0, 1, 2],
                                    [1, 0, 3],
                                    [2, 3, 0]])

        fitness_values = ga.compute_population_fitness(population, distance_matrix)

        self.assertEqual(len(fitness_values), 1)
        self.assertEqual(fitness_values[0], 6)

    def test_sort_population(self):
        population = [([1, 2, 3, 1], [1, 3, 2, 1])]
        fitness_values = [6]

        sorted_population, sorted_fitness_values = ga.sort_population(population, fitness_values)

        self.assertEqual(sorted_fitness_values[0], 6)

    def test_select_parent_binary_tournament(self):
        population = [ga.generate_individual(5) for _ in range(10)]
        distance_matrix = np.random.rand(5, 5) 
        
        parent = ga.select_parent_binary_tournament(population, distance_matrix)
        self.assertIsInstance(parent, tuple)
        self.assertEqual(len(parent), 2)

    def test_fill_chromosome(self):
        parent = np.array([1, 3, 2, 5, 4, 1])
        crossover_point_1 = 1
        crossover_point_2 = 3
        offspring = np.array([1, -1, -1, 2, -1, 1])

        expected_filled_offspring = np.array([1, 1, 3, 2, 2, 1])
        ga.fill_chromosome(offspring, parent, crossover_point_1, crossover_point_2)

        self.assertTrue(np.array_equal(offspring, expected_filled_offspring))

    def test_crossover(self):
        parent_1_route = np.array([1, 2, 3, 4, 1])
        parent_2_route = np.array([1, 4, 3, 2, 1])

        (offspring_1_route, offspring_2_route) = ga.crossover(parent_1_route, parent_2_route)

        self.assertEqual(len(offspring_1_route), len(parent_1_route))
        self.assertEqual(len(offspring_2_route), len(parent_2_route))

        for route in [offspring_1_route, offspring_2_route]:
            self.assertEqual(route[0], route[-1], msg="Offspring Route doesn't start and end at start node.")
            
            unique_nodes = set(route[:-1])
            expected_unique_count = len(parent_1_route) - 1
            
            self.assertEqual(len(unique_nodes), expected_unique_count, msg="Offspring Route doesn't contain unique nodes (barring start and end node)")


    def test_mutation(self):
        original_route = [1, 2, 3, 4, 1]
        n_nodes = 4  
        generation = 1
        generations_best_fitness_values = [1000]
        
        ga.MUTATION_RATE = 1.0  # Guarantee mutation for testing purposes; attempts to mitigate effect of the stochastic nature of the Algorithm

        has_mutated = False
        for i in range(100): # Try several times
            mutated_route = ga.mutation(copy.deepcopy(original_route), generation, generations_best_fitness_values, n_nodes)
            if mutated_route != original_route:
                has_mutated = True
                break
            else:
                continue

        self.assertTrue(has_mutated, msg="Did not Mutate, despite the fact that Mutation was expected")

    def test_should_apply_two_opt(self):
        generations_best_fitness_values = [1000, 995, 995, 990, 990, 985]  

        self.assertEqual(ga.should_apply_two_opt(generations_best_fitness_values), False)

    def test_compute_elitism_count(self):
        n_nodes = 100
        population_size = 750
        elitism_count = ga.compute_elitism_count(n_nodes, population_size)
        
        expected_elitism_count = int(population_size * ga.ELITISM_THRESHOLDS[100])

        self.assertEqual(elitism_count, expected_elitism_count)

    def test_generate_next_generation(self):
        n_nodes = len(self.test_city_1_nodes)
        
        distance_matrix = tsp.compute_distance_matrix(self.test_city_1_nodes)
        nearest_neighbours = to.compute_nearest_neighbours(distance_matrix)

        population = ga.initialise_population(self.test_city_1_nodes, n_nodes)
        population_fitness = ga.compute_population_fitness(population, distance_matrix)
        sorted_population, sorted_fitness_values = ga.sort_population(population, population_fitness)

        generations_best_fitness_values = [min(sorted_fitness_values)]
        new_population = ga.generate_next_generation(sorted_population, 1, generations_best_fitness_values, n_nodes, distance_matrix, nearest_neighbours, self.test_city_1_nodes)

        # print(new_population)

        self.assertEqual(len(new_population), ga.POPULATION_SIZE)

        # Check that the n most elite individuals are the same in the new gen
        elitism_count = ga.compute_elitism_count(n_nodes, len(population))
        for i in range(elitism_count):
            self.assertTrue(np.array_equal(sorted_population[i][1], new_population[i][1]))

        # Check diversity in new gen
        unique_individuals_count = len(set([tuple(individual[1]) for individual in new_population])) # Obtains number of unique routes in the population
        proportion_unique_individuals = unique_individuals_count  / len(new_population) * 100
        self.assertTrue(proportion_unique_individuals >= 40, f"New Generation isn't adequately diverse; only {np.round(proportion_unique_individuals)}% are unique") # Checks that there at least 40% of the population are unique individuals

    def test_run_ga_small(self):
        ga.POPULATION_SIZE = 300
        ga.GENERATIONS = 40
        ga.MUTATION_RATE = 0.35
        ga.CROSSOVER_RATE = 0.65

        generated_route, generated_distance = ga.run_ga_generic(file_name="test_city_1.xlsx",
                                                                import_node_data_func=tsp.import_node_data,  
                                                                population_size=ga.POPULATION_SIZE,
                                                                generations=ga.GENERATIONS,
                                                                mutation_rate=ga.MUTATION_RATE,
                                                                crossover_rate=ga.CROSSOVER_RATE,
                                                                display_route=False)

        expected_distance = 23.63

        start_node_row = self.test_city_1_nodes[self.test_city_1_nodes['Type'] == 'Start']
        start_node_number = start_node_row['Node'].iloc[0]

        self.assertAlmostEqual(expected_distance, generated_distance)
        self.assertEqual(generated_route[0], start_node_number)
        self.assertEqual(generated_route[-1], start_node_number)

    def test_run_ga_mid(self):
        ga.POPULATION_SIZE = 500
        ga.GENERATIONS = 220
        ga.MUTATION_RATE = 0.2
        ga.CROSSOVER_RATE = 0.8

        spreadsheet_name = tsp.convert_tsp_lib_instance_to_spreadsheet("berlin52.tsp")
        generated_route, generated_distance = ga.run_ga_generic(file_name=spreadsheet_name,
                                                                import_node_data_func=tsp.import_node_data_tsp_lib,  
                                                                population_size=ga.POPULATION_SIZE,
                                                                generations=ga.GENERATIONS,
                                                                mutation_rate=ga.MUTATION_RATE,
                                                                crossover_rate=ga.CROSSOVER_RATE,
                                                                display_route=False)

        expected_distance = 7544.37

        start_node_row = self.test_city_1_nodes[self.test_city_1_nodes['Type'] == 'Start']
        start_node_number = start_node_row['Node'].iloc[0]

        self.assertAlmostEqual(expected_distance, generated_distance)
        self.assertEqual(generated_route[0], start_node_number)
        self.assertEqual(generated_route[-1], start_node_number)


if __name__ == "__main__":
    unittest.main(warnings = "ignore")