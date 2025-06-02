import sys
sys.path.append("../")
from TSP_Utilities import tsp_utility_functions

import unittest, pyomo.environ as pyo
from mtz import run_mtz

class TestMTZ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_name = "test_city_1.xlsx"

        cls.nodes = tsp_utility_functions.import_node_data(cls.file_name)
        cls.n_nodes = len(cls.nodes)

        cls.solution_with_model = run_mtz(cls.file_name)

        cls.model = cls.solution_with_model[2]
        cls.x = cls.model.x
        cls.u = cls.model.u
        cls.M = cls.model.M

        cls.variables = cls.model.component_map(pyo.Var)

    """Unit Tests"""
    def test_x_variable(self):
        # Verify the number of "x"
        n_x = sum(len(TestMTZ.variables[var]) for var in TestMTZ.variables if "x" in var)
        self.assertEqual(n_x, 49)

        # Equal 1
        self.assertEqual(pyo.value(TestMTZ.x[0, 1]), 1)
        self.assertEqual(pyo.value(TestMTZ.x[1, 6]), 1)
        self.assertEqual(pyo.value(TestMTZ.x[2, 0]), 1)
        self.assertEqual(pyo.value(TestMTZ.x[3, 5]), 1)
        self.assertEqual(pyo.value(TestMTZ.x[4, 3]), 1)
        self.assertEqual(pyo.value(TestMTZ.x[5, 2]), 1)
        self.assertEqual(pyo.value(TestMTZ.x[6, 4]), 1)

        # Selected a few to Test that Equal 0
        self.assertEqual(pyo.value(TestMTZ.x[0, 6]), 0)
        self.assertEqual(pyo.value(TestMTZ.x[2, 3]), 0)
        self.assertEqual(pyo.value(TestMTZ.x[3, 1]), 0)
        self.assertEqual(pyo.value(TestMTZ.x[5, 3]), 0)
        self.assertEqual(pyo.value(TestMTZ.x[4, 0]), 0)
        self.assertEqual(pyo.value(TestMTZ.x[3, 4]), 0)
        self.assertEqual(pyo.value(TestMTZ.x[5, 1]), 0)
        self.assertEqual(pyo.value(TestMTZ.x[5, 4]), 0)
        self.assertEqual(pyo.value(TestMTZ.x[6, 0]), 0)
        self.assertEqual(pyo.value(TestMTZ.x[6, 2]), 0)
        self.assertEqual(pyo.value(TestMTZ.x[6, 5]), 0)

    def test_u_variable(self):
        # Assert values of "u" variable
        self.assertEqual(pyo.value(TestMTZ.u[0]), 1)
        self.assertEqual(pyo.value(TestMTZ.u[1]), 2)
        self.assertEqual(pyo.value(TestMTZ.u[2]), 7)
        self.assertEqual(pyo.value(TestMTZ.u[3]), 5)
        self.assertEqual(pyo.value(TestMTZ.u[4]), 4)
        self.assertEqual(pyo.value(TestMTZ.u[5]), 6)
        self.assertEqual(pyo.value(TestMTZ.u[6]), 3)

    def test_M_variable(self):
        # Verify the number of "M"
        n_M = sum(len(TestMTZ.variables[var]) for var in TestMTZ.variables if "M" in var)
        self.assertEqual(n_M, 49)

        # Assert values of a few "M"
        self.assertEqual(pyo.value(TestMTZ.M[1, 2]), 6)
        self.assertEqual(pyo.value(TestMTZ.M[1, 4]), 6)
        self.assertEqual(pyo.value(TestMTZ.M[2, 3]), 6)
        self.assertEqual(pyo.value(TestMTZ.M[2, 5]), 6)
        self.assertEqual(pyo.value(TestMTZ.M[3, 1]), 6)
        self.assertEqual(pyo.value(TestMTZ.M[3, 2]), 6)
        self.assertEqual(pyo.value(TestMTZ.M[4, 2]), 6)
        self.assertEqual(pyo.value(TestMTZ.M[4, 6]), 6)
        self.assertEqual(pyo.value(TestMTZ.M[5, 3]), 6)
        self.assertEqual(pyo.value(TestMTZ.M[6, 5]), 6)

        self.assertEqual(pyo.value(TestMTZ.M[1, 6]), 0)
        self.assertEqual(pyo.value(TestMTZ.M[3, 5]), 0)
        self.assertEqual(pyo.value(TestMTZ.M[4, 3]), 0)
        self.assertEqual(pyo.value(TestMTZ.M[5, 2]), 0)
        self.assertEqual(pyo.value(TestMTZ.M[6, 4]), 0)

    def test_constraint_1(self):
        self.assertEqual(str(TestMTZ.model.C1[1].expr), "x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] + x[0,6]  ==  1")
        self.assertEqual(str(TestMTZ.model.C1[2].expr), "x[1,0] + x[1,2] + x[1,3] + x[1,4] + x[1,5] + x[1,6]  ==  1")
        self.assertEqual(str(TestMTZ.model.C1[3].expr), "x[2,0] + x[2,1] + x[2,3] + x[2,4] + x[2,5] + x[2,6]  ==  1")
        self.assertEqual(str(TestMTZ.model.C1[4].expr), "x[3,0] + x[3,1] + x[3,2] + x[3,4] + x[3,5] + x[3,6]  ==  1")
        self.assertEqual(str(TestMTZ.model.C1[5].expr), "x[4,0] + x[4,1] + x[4,2] + x[4,3] + x[4,5] + x[4,6]  ==  1")
        self.assertEqual(str(TestMTZ.model.C1[6].expr), "x[5,0] + x[5,1] + x[5,2] + x[5,3] + x[5,4] + x[5,6]  ==  1")
        self.assertEqual(str(TestMTZ.model.C1[7].expr), "x[6,0] + x[6,1] + x[6,2] + x[6,3] + x[6,4] + x[6,5]  ==  1")

    def test_constraint_2(self):
        self.assertEqual(str(TestMTZ.model.C2[1].expr), "x[1,0] + x[2,0] + x[3,0] + x[4,0] + x[5,0] + x[6,0]  ==  1")
        self.assertEqual(str(TestMTZ.model.C2[2].expr), "x[0,1] + x[2,1] + x[3,1] + x[4,1] + x[5,1] + x[6,1]  ==  1")
        self.assertEqual(str(TestMTZ.model.C2[3].expr), "x[0,2] + x[1,2] + x[3,2] + x[4,2] + x[5,2] + x[6,2]  ==  1")
        self.assertEqual(str(TestMTZ.model.C2[4].expr), "x[0,3] + x[1,3] + x[2,3] + x[4,3] + x[5,3] + x[6,3]  ==  1")
        self.assertEqual(str(TestMTZ.model.C2[5].expr), "x[0,4] + x[1,4] + x[2,4] + x[3,4] + x[5,4] + x[6,4]  ==  1")
        self.assertEqual(str(TestMTZ.model.C2[6].expr), "x[0,5] + x[1,5] + x[2,5] + x[3,5] + x[4,5] + x[6,5]  ==  1")
        self.assertEqual(str(TestMTZ.model.C2[7].expr), "x[0,6] + x[1,6] + x[2,6] + x[3,6] + x[4,6] + x[5,6]  ==  1")

    def test_constraint_3(self):
        self.assertEqual(str(TestMTZ.model.C3.expr), "u[0]  ==  1")  

    def test_constraint_4(self):
        self.assertEqual(str(TestMTZ.model.C4[1].expr), "2  <=  u[1]  <=  7")
        self.assertEqual(str(TestMTZ.model.C4[2].expr), "2  <=  u[2]  <=  7")
        self.assertEqual(str(TestMTZ.model.C4[3].expr), "2  <=  u[3]  <=  7")
        self.assertEqual(str(TestMTZ.model.C4[4].expr), "2  <=  u[4]  <=  7")
        self.assertEqual(str(TestMTZ.model.C4[5].expr), "2  <=  u[5]  <=  7")
        self.assertEqual(str(TestMTZ.model.C4[6].expr), "2  <=  u[6]  <=  7")
        
    def test_constraint_5(self):
        n_C5 = len(TestMTZ.model.component("C5"))
        self.assertEqual(n_C5, 30)

        self.assertEqual(str(TestMTZ.model.C5[1].expr), "u[1] - u[2] + 1 - M[1,2]  <=  0")
        self.assertEqual(str(TestMTZ.model.C5[3].expr), "u[1] - u[4] + 1 - M[1,4]  <=  0")
        self.assertEqual(str(TestMTZ.model.C5[9].expr), "u[2] - u[5] + 1 - M[2,5]  <=  0")
        self.assertEqual(str(TestMTZ.model.C5[12].expr), "u[3] - u[2] + 1 - M[3,2]  <=  0")
        self.assertEqual(str(TestMTZ.model.C5[15].expr), "u[3] - u[6] + 1 - M[3,6]  <=  0")
        self.assertEqual(str(TestMTZ.model.C5[16].expr), "u[4] - u[1] + 1 - M[4,1]  <=  0")
        self.assertEqual(str(TestMTZ.model.C5[22].expr), "u[5] - u[2] + 1 - M[5,2]  <=  0")
        self.assertEqual(str(TestMTZ.model.C5[24].expr), "u[5] - u[4] + 1 - M[5,4]  <=  0")
        self.assertEqual(str(TestMTZ.model.C5[28].expr), "u[6] - u[3] + 1 - M[6,3]  <=  0")
        self.assertEqual(str(TestMTZ.model.C5[30].expr), "u[6] - u[5] + 1 - M[6,5]  <=  0")

    # .replace(" ","")
    def test_objective_function(self):
        self.maxDiff = None
        if 'unittest.util' in __import__('sys').modules:
            __import__('sys').modules['unittest.util']._MAX_LENGTH = 999999999

        extracted_objective_expression = str(TestMTZ.model.obj.expr).replace(" ","")
        sense = "maximize" if TestMTZ.model.obj.sense == pyo.maximize else "minimize"
        
        expected_objective_expression = "0.0*x[0,0] + 3.1622776601683795*x[1,0] + 1.4142135623730951*x[2,0] + 5.0*x[3,0] + 9.486832980505138*x[4,0] + 3.1622776601683795*x[5,0] + 6.4031242374328485*x[6,0] + 3.1622776601683795*x[0,1] + 0.0*x[1,1] + 4.0*x[2,1] + 5.0*x[3,1] + 8.0*x[4,1] + 2.8284271247461903*x[5,1] + 3.605551275463989*x[6,1] + 1.4142135623730951*x[0,2] + 4.0*x[1,2] + 0.0*x[2,2] + 4.123105625617661*x[3,2] + 8.94427190999916*x[4,2] + 2.8284271247461903*x[5,2] + 6.708203932499369*x[6,2] + 5.0*x[0,3] + 5.0*x[1,3] + 4.123105625617661*x[2,3] + 0.0*x[3,3] + 5.0*x[4,3] + 2.23606797749979*x[5,3] + 5.0990195135927845*x[6,3] + 9.486832980505138*x[0,4] + 8.0*x[1,4] + 8.94427190999916*x[2,4] + 5.0*x[3,4] + 0.0*x[4,4] + 6.324555320336759*x[5,4] + 5.385164807134504*x[6,4] + 3.1622776601683795*x[0,5] + 2.8284271247461903*x[1,5] + 2.8284271247461903*x[2,5] + 2.23606797749979*x[3,5] + 6.324555320336759*x[4,5] + 0.0*x[5,5] + 4.123105625617661*x[6,5] + 6.4031242374328485*x[0,6] + 3.605551275463989*x[1,6] + 6.708203932499369*x[2,6] + 5.0990195135927845*x[3,6] + 5.385164807134504*x[4,6] + 4.123105625617661*x[5,6] + 0.0*x[6,6]".replace(" ","")

        self.assertEqual(extracted_objective_expression, expected_objective_expression)
        self.assertEqual(sense, "minimize")

    """Integration Tests"""
    def test_mtz(self):
        route, distance_travelled = TestMTZ.solution_with_model[:2]
        
        self.assertEqual(route, [1, 2, 7, 5, 4, 6, 3, 1])
        self.assertAlmostEqual(distance_travelled, 23.63)


if __name__ == "__main__":
    unittest.main(warnings = "ignore") 