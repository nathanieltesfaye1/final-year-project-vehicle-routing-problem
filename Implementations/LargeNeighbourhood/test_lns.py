import sys
sys.path.append("../")
from TSP_Utilities.Test_Inputs import mock_route_data as mock
from TSP_Utilities import tsp_utility_functions as tsp
from LargeNeighbourhood import lns

import unittest, numpy as np

class TestLNS(unittest.TestCase):
    def setUp(self):
        self.test_city_1_nodes = mock.get_test_city_1_nodes()
        self.berlin52_nodes = mock.get_berlin52_nodes()
        self.tsp225_nodes = mock.get_tsp225_nodes()

        self.test_city_1_distance_matrix = mock.get_test_city_1_distance_matrix()
        self.test_city_1_node_index_mapping = mock.get_test_city_1_node_index_mapping()

        self.berlin52_distance_matrix = mock.get_berlin52_distance_matrix()
        self.berlin52_node_index_mapping = mock.get_berlin52_node_index_mapping()

        self.test_city_1_sample_route = [1, 2, 3, 4, 5, 6, 7, 1]

    def test_destroy(self):
        destroyed_route, _ = lns.destroy(self.test_city_1_sample_route, proportion=0.15)
        self.assertTrue(len(destroyed_route) < len(self.test_city_1_sample_route), msg=f"Route has not been destroyed; there are {len(destroyed_route)} nodes in the supposedly destroyed route, while there were {len(self.test_city_1_sample_route)} nodes in the original route.")
        self.assertTrue(all(node in self.test_city_1_sample_route for node in destroyed_route), msg=f"The nodes in the destroyed route are {destroyed_route}, some of which are not inside the original route, which was {self.test_city_1_sample_route}.")

    def test_repair(self):
        reamaining_route, removed_nodes = lns.destroy(self.test_city_1_sample_route, proportion=0.15)
        repaired_route = lns.repair(reamaining_route, removed_nodes, self.test_city_1_distance_matrix, self.test_city_1_node_index_mapping)
        self.assertTrue(all(node in self.test_city_1_sample_route for node in repaired_route))

    def test_nn_lns_integrated_small(self):
        initial_solution = ([1, 3, 6, 4, 2, 7, 5, 1], 29.96)
        initial_route = initial_solution[0]
        generated_route, generated_distance = lns.lns(initial_route, self.test_city_1_distance_matrix, self.test_city_1_node_index_mapping)

        expected_distance = 23.63

        self.assertAlmostEqual(expected_distance, generated_distance)

    def test_ihcv_lns_integrated_mid(self):
        initial_solution = ([1, 22, 31, 18, 3, 17, 21, 7, 2, 42, 30, 23, 20, 50, 16, 29, 47, 26, 28, 27, 13, 14, 52, 11, 12, 51, 33, 43, 10, 9, 8, 41, 19, 45, 32, 49, 37, 46, 48, 24, 5, 6, 25, 4, 15, 38, 40, 39, 36, 35, 34, 44, 1], 8105.78)
        initial_route = initial_solution[0]
        initial_distance = initial_solution[1]
        generated_route, generated_distance = lns.lns(initial_route, self.berlin52_distance_matrix, self.berlin52_node_index_mapping, max_iterations=100000, improvement_threshold=0.000001)

        self.assertEqual(len(generated_route), len(initial_route))
        self.assertEqual(generated_route[0], initial_route[0])
        self.assertEqual(generated_route[-1], initial_route[-1])
        self.assertTrue(generated_distance < initial_distance)

    def test_nn_lns_integrated_large(self):
        initial_route = [1, 200, 3, 198, 4, 197, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 203, 19, 18, 22, 21, 23, 24, 208, 25, 26, 34, 33, 35, 30, 202, 206, 31, 216, 219, 217, 77, 78, 79, 80, 81, 95, 209, 94, 93, 92, 91, 90, 87, 210, 84, 83, 82, 85, 86, 131, 211, 130, 222, 129, 128, 127, 126, 125, 124, 123, 122, 121, 175, 120, 185, 119, 118, 186, 187, 117, 116, 223, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 220, 104, 103, 102, 101, 100, 99, 98, 97, 96, 221, 28, 204, 29, 32, 38, 39, 40, 41, 42, 43, 44, 46, 194, 218, 193, 196, 192, 191, 199, 224, 133, 190, 225, 47, 2, 207, 49, 51, 57, 56, 55, 52, 53, 54, 70, 71, 72, 73, 74, 75, 76, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 50, 48, 45, 195, 205, 189, 27, 188, 184, 182, 173, 181, 174, 180, 179, 176, 177, 178, 172, 171, 170, 169, 168, 212, 214, 151, 150, 149, 152, 153, 154, 155, 156, 157, 144, 143, 201, 142, 141, 140, 139, 138, 137, 136, 183, 135, 134, 215, 164, 165, 166, 167, 213, 158, 163, 162, 161, 160, 159, 146, 147, 148, 145, 132, 88, 89, 37, 36, 1]
        initial_distance = 4829.0

        generated_route, generated_distance = lns.lns(initial_route, tsp.compute_distance_matrix(self.tsp225_nodes), mock.get_tsp225_node_index_mapping())

        self.assertEqual(len(generated_route), len(initial_route))
        self.assertEqual(generated_route[0], initial_route[0])
        self.assertEqual(generated_route[-1], initial_route[-1])
        self.assertTrue(generated_distance < initial_distance)

    def test_run_to_lns_integrated(self):
        generated_route, generated_distance = lns.run_lns_two_opt_ihcv_tsp_lib("berlin52.tsp", False)

        expected_route = [1, 22, 31, 18, 3, 17, 21, 42, 7, 2, 30, 23, 20, 50, 16, 29, 47, 26, 28, 27, 13, 14, 52, 11, 12, 51, 33, 43, 10, 9, 8, 41, 19, 45, 32, 49, 36, 35, 34, 39, 40, 37, 38, 48, 24, 5, 15, 6, 4, 25, 46, 44, 1]
        expected_distance = 7887.23

        self.assertEqual(generated_route, expected_route)
        self.assertAlmostEqual(generated_distance, expected_distance)

if __name__ == "__main__":
    unittest.main(warnings = "ignore")