import sys
sys.path.append("../")
from TSP_Utilities.Test_Inputs import mock_route_data as mock
from TSP_Utilities import tsp_utility_functions as tsp

import unittest
from IHCV import ihcv as ih

class TestIHCV(unittest.TestCase):
    def setUp(self):
        self.test_city_1_nodes = mock.get_test_city_1_nodes()
        self.berlin52_nodes = mock.get_berlin52_nodes()
        self.tsp225_nodes = mock.get_tsp225_nodes()

    def test_compute_convex_hull(self):
        expected_hull = [32, 8, 16, 6, 1, 13, 51, 10]

        generated_hull = ih.compute_convex_hull(self.berlin52_nodes)[1]
        generated_hull_as_list = list(generated_hull.vertices)

        self.assertEqual(expected_hull, generated_hull_as_list)

    def test_ihcv_integrated_small(self):
        expected_route = [1, 3, 6, 4, 5, 7, 2, 1]
        expected_distance = 23.63

        coordinate_array, hull = ih.compute_convex_hull(self.test_city_1_nodes)
        distance_matrix = ih.compute_distance_matrix(coordinate_array)
        generated_route, generated_distance = ih.cheapest_insertion(coordinate_array, hull, distance_matrix, self.test_city_1_nodes)

        self.assertEqual(generated_route, expected_route)
        self.assertAlmostEqual(expected_distance, generated_distance, delta=0.5)

    def test_ihcv_integrated_mid(self):
        expected_route = [1, 22, 31, 18, 3, 17, 21, 7, 2, 42, 30, 23, 20, 50, 16, 29, 47, 26, 28, 27, 13, 14, 52, 11, 12, 51, 33, 43, 10, 9, 8, 41, 19, 45, 32, 49, 37, 46, 48, 24, 5, 6, 25, 4, 15, 38, 40, 39, 36, 35, 34, 44, 1]
        expected_distance = 8105.78

        coordinate_array, hull = ih.compute_convex_hull(self.berlin52_nodes)
        distance_matrix = ih.compute_distance_matrix(coordinate_array)
        generated_route, generated_distance = ih.cheapest_insertion(coordinate_array, hull, distance_matrix, self.berlin52_nodes)

        self.assertEqual(generated_route, expected_route)
        self.assertAlmostEqual(expected_distance, generated_distance, delta=0.5)

    def test_ihcv_integrated_large(self):
        expected_route = [1, 200, 3, 198, 4, 197, 195, 46, 194, 218, 193, 45, 48, 196, 192, 191, 224, 199, 133, 205, 189, 190, 225, 49, 50, 51, 57, 52, 53, 54, 55, 56, 58, 59, 207, 2, 47, 27, 188, 117, 187, 118, 116, 223, 114, 62, 60, 61, 63, 66, 68, 69, 67, 65, 64, 112, 110, 111, 113, 115, 119, 186, 185, 120, 175, 121, 124, 123, 122, 184, 182, 171, 170, 172, 173, 181, 174, 180, 179, 176, 178, 177, 148, 149, 150, 169, 168, 212, 214, 125, 126, 127, 167, 151, 152, 153, 154, 156, 157, 155, 147, 146, 145, 144, 143, 201, 142, 141, 140, 139, 138, 137, 136, 183, 135, 163, 161, 159, 160, 162, 158, 213, 166, 165, 164, 134, 215, 132, 129, 128, 222, 130, 211, 131, 86, 85, 210, 87, 84, 82, 83, 92, 90, 88, 89, 91, 94, 93, 96, 97, 209, 95, 221, 81, 80, 79, 78, 77, 217, 219, 216, 98, 99, 100, 101, 103, 105, 107, 109, 108, 106, 220, 104, 102, 73, 71, 70, 72, 75, 74, 76, 31, 32, 206, 202, 30, 29, 28, 204, 35, 33, 34, 26, 25, 36, 208, 24, 23, 13, 12, 14, 16, 17, 15, 22, 21, 20, 203, 19, 18, 11, 37, 38, 39, 10, 9, 40, 41, 42, 44, 43, 8, 7, 6, 5, 1]
        expected_distance = 4442.27

        coordinate_array, hull = ih.compute_convex_hull(self.tsp225_nodes)
        distance_matrix = ih.compute_distance_matrix(coordinate_array)
        generated_route, generated_distance = ih.cheapest_insertion(coordinate_array, hull, distance_matrix, self.tsp225_nodes)

        self.assertEqual(generated_route, expected_route)
        self.assertAlmostEqual(expected_distance, generated_distance, delta=0.5)

    def test_run_ihcv_generic(self):
        generated_route, generated_distance = ih.run_ihcv_generic("berlin52.xlsx", tsp.import_node_data_tsp_lib, False)[0]

        expected_route = [1, 22, 31, 18, 3, 17, 21, 7, 2, 42, 30, 23, 20, 50, 16, 29, 47, 26, 28, 27, 13, 14, 52, 11, 12, 51, 33, 43, 10, 9, 8, 41, 19, 45, 32, 49, 37, 46, 48, 24, 5, 6, 25, 4, 15, 38, 40, 39, 36, 35, 34, 44, 1]
        expected_distance = 8105.78

        self.assertEqual(generated_route, expected_route)
        self.assertAlmostEqual(expected_distance, generated_distance, delta=0.5)



if __name__ == "__main__":
    unittest.main(warnings = "ignore")