import sys
sys.path.append("../")
from TSP_Utilities.Test_Inputs import mock_route_data as mock
from TSP_Utilities import tsp_utility_functions as tsp

import unittest
from NearestNeighbour import nearest_neighbour as nn

class TestNearestNeighbour(unittest.TestCase):
    def setUp(self):
        self.test_city_1_nodes = mock.get_test_city_1_nodes()
        self.berlin52_nodes = mock.get_berlin52_nodes()
        self.tsp225_nodes = mock.get_tsp225_nodes()

    def test_nearest_neighbour_small(self):
        expected_route = [1, 3, 6, 4, 2, 7, 5, 1]
        expected_distance = 29.96

        generated_route, generated_distance = nn.nearest_neighbour(self.test_city_1_nodes)

        self.assertEqual(generated_route, expected_route)
        self.assertAlmostEqual(expected_distance, generated_distance, delta=0.5)

    def test_nearest_neighbour_mid(self):
        expected_route = [1, 22, 49, 32, 36, 35, 34, 39, 40, 38, 37, 48, 24, 5, 15, 6, 4, 25, 46, 44, 16, 50, 20, 23, 31, 18, 3, 19, 45, 41, 8, 10, 9, 43, 33, 51, 12, 28, 27, 26, 47, 13, 14, 52, 11, 29, 30, 21, 17, 42, 7, 2, 1]
        expected_distance = 8980.92

        generated_route, generated_distance = nn.nearest_neighbour(self.berlin52_nodes)

        self.assertEqual(generated_route, expected_route)
        self.assertAlmostEqual(generated_distance, expected_distance)

    def test_nearest_neighbour_large(self):
        expected_route = [1, 200, 3, 198, 4, 197, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 203, 19, 18, 22, 21, 23, 24, 208, 25, 26, 34, 33, 35, 30, 202, 206, 31, 216, 219, 217, 77, 78, 79, 80, 81, 95, 209, 94, 93, 92, 91, 90, 87, 210, 84, 83, 82, 85, 86, 131, 211, 130, 222, 129, 128, 127, 126, 125, 124, 123, 122, 121, 175, 120, 185, 119, 118, 186, 187, 117, 116, 223, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 220, 104, 103, 102, 101, 100, 99, 98, 97, 96, 221, 28, 204, 29, 32, 38, 39, 40, 41, 42, 43, 44, 46, 194, 218, 193, 196, 192, 191, 199, 224, 133, 190, 225, 47, 2, 207, 49, 51, 57, 56, 55, 52, 53, 54, 70, 71, 72, 73, 74, 75, 76, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 50, 48, 45, 195, 205, 189, 27, 188, 184, 182, 173, 181, 174, 180, 179, 176, 177, 178, 172, 171, 170, 169, 168, 212, 214, 151, 150, 149, 152, 153, 154, 155, 156, 157, 144, 143, 201, 142, 141, 140, 139, 138, 137, 136, 183, 135, 134, 215, 164, 165, 166, 167, 213, 158, 163, 162, 161, 160, 159, 146, 147, 148, 145, 132, 88, 89, 37, 36, 1]
        expected_distance = 4829.0

        generated_route, generated_distance = nn.nearest_neighbour(self.tsp225_nodes)

        self.assertEqual(generated_route, expected_route)
        self.assertAlmostEqual(generated_distance, expected_distance)

    def test_run_nearest_neighbour_generic(self):
        generated_route, generated_distance = nn.run_nearest_neighbour_generic("berlin52.xlsx", tsp.import_node_data_tsp_lib, False)[0]

        expected_route = [1, 22, 49, 32, 36, 35, 34, 39, 40, 38, 37, 48, 24, 5, 15, 6, 4, 25, 46, 44, 16, 50, 20, 23, 31, 18, 3, 19, 45, 41, 8, 10, 9, 43, 33, 51, 12, 28, 27, 26, 47, 13, 14, 52, 11, 29, 30, 21, 17, 42, 7, 2, 1]
        expected_distance = 8980.92

        self.assertEqual(generated_route, expected_route)
        self.assertAlmostEqual(generated_distance, expected_distance)



if __name__ == "__main__":
    unittest.main(warnings = "ignore")