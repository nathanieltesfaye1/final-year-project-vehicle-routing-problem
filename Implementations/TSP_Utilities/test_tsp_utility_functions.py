import unittest, pytest, os, pandas as pd, numpy as np
from tsp_utility_functions import convert_tsp_lib_instance_to_spreadsheet, import_node_data, compute_distance_matrix , compute_route_distance
from Test_Inputs.mock_route_data import get_test_city_1_distance_matrix, get_test_city_1_node_index_mapping
    
# Helper Function - Converts decimal numbers like 1.0, 2.0, etc. to their regular integer form (1, 2, etc.), while leaving other decimals (e.g. 3.2) untouched 
def downcast_to_int64(table):
    for column in table.columns:
        if table[column].dtype == float:
            table[column] = table[column].astype('int64')
    
    return table

class TestTSPUtilityFunctions(unittest.TestCase):
    def setUp(self):
        self.att48_name = "att48.tsp"

        self.test_city_1_file = "test_city_1.xlsx"

        self.nodes = pd.DataFrame({'Node': [1, 2, 3, 4, 5], 'X': [0, 4, 8, 7, 3], 'Y': [0, 3, 0, 5, 8], 'Type': ['Start', 'Waypoint', 'Waypoint', 'Waypoint', 'Waypoint']})
        self.n_nodes = len(self.nodes)

        file_path = "Test_Inputs/TSPLIB_Instances/att48.xlsx"
        if os.path.exists(file_path):
            os.remove(file_path)


        # Invalid TSP Instances Files:
        self.duplicate_columns_name = "DuplicateColumns.xlsx"
        self.empty_cells_name = "EmptyCells.xlsx"
        self.empty_row_name = "EmptyRow.xlsx"
        self.input_wrong_data_type_into_columns_name = "InputWrongDataTypeIntoColumns.xlsx"
        self.invalid_node_numbers_name = "InvalidNodeNumbers.xlsx"
        self.invalid_type_text_name = "InvalidTypeText.xlsx"
        self.misspelled_type_name = "MisspelledType.xlsx"
        self.multiple_issues_1_name = "MultipleIssues_1.xlsx"
        self.multiple_issues_2_name = "MultipleIssues_2.xlsx"
        self.multiple_nodes_with_same_node_number_name = "MultipleNodesWithSameNodeNumber.xlsx"
        self.multiple_start_nodes_name = "MultipleStartNodes.xlsx"
        self.node_numbers_not_sequential_name = "NodeNumbersNotSequential.xlsx"
        self.no_start_nodes_name = "NoStartNodes.xlsx"
        self.not_enough_nodes_name = "NotEnoughNodes.xlsx"
        self.rows_not_in_order_of_nodes_name = "RowsNotInOrderOfNodes.xlsx"
        self.start_node_is_not_1_name = "StartNodeIsNot1.xlsx"
        self.two_nodes_with_same_coordinates_name = "TwoNodesWithSameCoordinates.xlsx"
        self.valid_name_name = "Valid_Name.xlsx"
        self.valid_name = "Valid.xlsx"
        self.wrong_column_names_name = "WrongColumnNames.xlsx"
        self.wrong_number_of_columns_name = "WrongNumberOfColumns.xlsx"

    def tearDown(self):
        file_path = "Test_Inputs/TSPLIB_Instances/att48.xlsx"
        if os.path.exists(file_path):
            os.remove(file_path)

    def test_compute_distance_matrix(self):
        expected_distance_matrix = np.array([
            [0, np.sqrt(np.square(4-0) + np.square(3-0)), np.sqrt(np.square(8-0) + np.square(0-0)), np.sqrt(np.square(7-0) + np.square(5-0)), np.sqrt(np.square(3-0) + np.square(8-0))],
            [np.sqrt(np.square(4-0) + np.square(3-0)), 0, np.sqrt(np.square(8-4) + np.square(0-3)), np.sqrt(np.square(7-4) + np.square(5-3)), np.sqrt(np.square(3-4) + np.square(8-3))],
            [np.sqrt(np.square(8-0) + np.square(0-0)), np.sqrt(np.square(8-4) + np.square(0-3)), 0, np.sqrt(np.square(7-8) + np.square(5-0)), np.sqrt(np.square(3-8) + np.square(8-0))],
            [np.sqrt(np.square(7-0) + np.square(5-0)), np.sqrt(np.square(7-4) + np.square(5-3)), np.sqrt(np.square(7-8) + np.square(5-0)), 0, np.sqrt(np.square(3-7) + np.square(8-5))],
            [np.sqrt(np.square(3-0) + np.square(8-0)), np.sqrt(np.square(3-4) + np.square(8-3)), np.sqrt(np.square(3-8) + np.square(8-0)), np.sqrt(np.square(3-7) + np.square(8-5)), 0]
        ])

        generated_distance_matrix = compute_distance_matrix(self.nodes)

        np.testing.assert_array_almost_equal(expected_distance_matrix, generated_distance_matrix, decimal = 1, err_msg = "The computed distance matrix is not the same as the expected distance matrix", verbose = True)

    def test_compute_route_distance(self):
        generated_route_distance = compute_route_distance([1, 3, 6, 4, 5, 7, 2, 1], get_test_city_1_distance_matrix(), get_test_city_1_node_index_mapping())
        expected_route_distance = 23.63

        self.assertAlmostEqual(np.round(generated_route_distance, 2), expected_route_distance)

    def check_table_row_values(self, table_row, expected_values):
        for i, expected in expected_values.items():
            self.assertEqual(table_row[i], expected, f"{i} should have been {expected}, but was instead {table_row[i]}")
    
    # Test 'convert_tsp_lib_instance_to_spreadsheet' function
    def test_convert_tsp_lib_instance_to_spreadsheet(self):
        spreadsheet_name = convert_tsp_lib_instance_to_spreadsheet(self.att48_name)
        self.assertTrue(os.path.exists(f"Test_Inputs/TSPLIB_Instances/{spreadsheet_name}")) # Check if new .xlsx file exists in correct place

        # Verify contents of the generated .xlsx file
        generated_table = pd.read_excel(f"Test_Inputs/TSPLIB_Instances/{spreadsheet_name}")
        
        self.check_table_row_values(generated_table.iloc[0].to_dict(), {"Node": 1, "X": 6734, "Y": 1453, "Type": "Start"})
        self.check_table_row_values(generated_table.iloc[47].to_dict(), {"Node": 48, "X": 3023, "Y": 1942, "Type": "Waypoint"})
        self.check_table_row_values(generated_table.iloc[12].to_dict(), {"Node": 13, "X": 4706, "Y": 2674, "Type": "Waypoint"})

        self.assertEqual(len(generated_table), 48, f"The generated table should have 48 rows, but instead has {len(generated_table)}")

    # Test 'import_node_data' function
    def test_import_node_data(self):
        nodes = import_node_data(self.test_city_1_file)
        expected_table = pd.DataFrame({"Node": [1, 2, 3, 4, 5, 6, 7], "X": [1, 2, 2, 6, 10, 4, 5], "Y": [2, 5, 1, 2, 5, 3, 7], 
                                       "Type": ["Start", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint"]})
        pd.testing.assert_frame_equal(nodes, expected_table)

    # Invalidate Duplicate Columns in Input Data
    def test_duplicate_columns(self):
        with pytest.raises(ValueError) as e:
            import_node_data(self.duplicate_columns_name, for_testing_purposes=True)

    # Invalidate Empty Cells in Input Data
    def test_empty_cells(self):
        with pytest.raises(ValueError) as e:
            import_node_data(self.empty_cells_name, for_testing_purposes=True)
    
    # Invalidate/Correct Empty Row(s) in Input Data
    def test_empty_rows(self):
        nodes = import_node_data(self.empty_row_name, for_testing_purposes=True)
        expected_table = pd.DataFrame({"Node": [1, 2, 3, 4, 5, 6], "X": [1, 2, 2, 10, 4, 5], "Y": [2, 5, 1, 5, 3, 7], 
                                       "Type": ["Start", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint"]})
        
        pd.testing.assert_frame_equal(downcast_to_int64(nodes), expected_table)

    # Invalidate Wrong Data Type in Columns in Input Data
    def test_input_wrong_data_type_into_columns(self):
        with pytest.raises(ValueError) as e:
            import_node_data(self.input_wrong_data_type_into_columns_name, for_testing_purposes=True)

    # Invalidate Invalid Node Numbers in Input Data
    def test_invalid_node_numbers(self):
        with pytest.raises(ValueError) as e:
            import_node_data(self.invalid_node_numbers_name, for_testing_purposes=True)

    # Invalidate Invalid Type Text in Input Data     
    def test_invalid_type_text(self):
        with pytest.raises(ValueError) as e:
            import_node_data(self.invalid_type_text_name, for_testing_purposes=True)

    # Invalidate/Correct Misspelled Type in Input Data     
    def test_misspelled_type(self):
        nodes = import_node_data(self.misspelled_type_name, for_testing_purposes=True)
        expected_table = pd.DataFrame({"Node": [1, 2, 3, 4, 5, 6, 7], "X": [1, 2, 2, 6, 10, 4, 5], "Y": [2, 5, 1, 2, 5, 3, 7], 
                                       "Type": ["Start", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint"]})
        pd.testing.assert_frame_equal(downcast_to_int64(nodes), expected_table)

    # Invalidate/Correct TSP Instance (1) w/ Multiple Issues
    def test_multiple_issues_1(self):
        nodes = import_node_data(self.multiple_issues_1_name, for_testing_purposes=True)
        expected_table = pd.DataFrame({"Node": [1, 2, 3, 4, 5, 6], "X": [1, 2, 2, 6, 10, 4], "Y": [2, 5, 1, 2, 5, 3], 
                                       "Type": ["Start", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint"]})
        
        pd.testing.assert_frame_equal(downcast_to_int64(nodes), expected_table)

    # Invalidate/Correct TSP Instance (2) w/ Multiple Issues
    def test_multiple_issues_2(self):
        nodes = import_node_data(self.multiple_issues_2_name, for_testing_purposes=True)
        expected_table = pd.DataFrame({"Node": [1, 2, 3, 4, 5, 6], "X": [6, 1, 2, 2, 10, 4], "Y": [2, 2, 5, 1, 5, 3], 
                                       "Type": ["Start", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint"]})
        
        pd.testing.assert_frame_equal(downcast_to_int64(nodes), expected_table)

    # Invalidate/Correct Multiple Nodes w/ Same Node Number in Input Data
    def test_multiple_nodes_with_same_node_number(self):
        nodes = import_node_data(self.multiple_nodes_with_same_node_number_name, for_testing_purposes=True)
        expected_table = pd.DataFrame({"Node": [1, 2, 3, 4, 5, 6, 7], "X": [1, 2, 2, 10, 6, 4, 5], "Y": [2, 5, 1, 5, 2, 3, 7], 
                                       "Type": ["Start", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint"]})
        
        pd.testing.assert_frame_equal(downcast_to_int64(nodes), expected_table)

    # Invalidate Multiple Start Nodes in Input Data
    def test_multiple_start_nodes(self):
        with pytest.raises(ValueError) as e:
            import_node_data(self.multiple_start_nodes_name, for_testing_purposes=True)

    # Invalidate no start nodes in Input Data
    def test_no_start_nodes(self):
        with pytest.raises(ValueError) as e:
            import_node_data(self.no_start_nodes_name, for_testing_purposes=True)

    # Invalidate/Correct Nodes don't range from 1 --> n in Input Data     
    def test_node_numbers_not_sequential(self):
        nodes = import_node_data(self.node_numbers_not_sequential_name, for_testing_purposes=True)
        expected_table = pd.DataFrame({"Node": [1, 2, 3, 4, 5, 6, 7], "X": [1, 2, 2, 6, 10, 4, 5], "Y": [2, 5, 1, 2, 5, 3, 7], 
                                       "Type": ["Start", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint"]})
        
        pd.testing.assert_frame_equal(downcast_to_int64(nodes), expected_table)

    # Invalidate no Start Nodes in Input Data
    def test_not_enough_nodes(self):
        with pytest.raises(ValueError) as e:
            import_node_data(self.not_enough_nodes_name, for_testing_purposes=True)

    # Invalidate/Correct rows not in order of Nodes in Input Data
    def test_rows_not_in_order_of_nodes(self):
        nodes = import_node_data(self.node_numbers_not_sequential_name, for_testing_purposes=True)
        expected_table = pd.DataFrame({"Node": [1, 2, 3, 4, 5, 6, 7], "X": [1, 2, 2, 6, 10, 4, 5], "Y": [2, 5, 1, 2, 5, 3, 7], 
                                       "Type": ["Start", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint"]})
        
        pd.testing.assert_frame_equal(downcast_to_int64(nodes), expected_table)

    # Invalidate/Correct Start Node is not Node Number 1 in Input Data
    def test_start_node_is_not_1(self):
        nodes = import_node_data(self.start_node_is_not_1_name, for_testing_purposes=True)
        expected_table = pd.DataFrame({"Node": [1, 2, 3, 4, 5, 6, 7], "X": [6, 1, 2, 2, 10, 4, 5], "Y": [2, 2, 5, 1, 5, 3, 7], 
                                       "Type": ["Start", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint"]})
        
        pd.testing.assert_frame_equal(downcast_to_int64(nodes), expected_table)

    # Invalidate/Correct 2 Nodes have the Same Coordinates in Input Data
    def test_two_nodes_with_same_coordinates(self):
        nodes = import_node_data(self.two_nodes_with_same_coordinates_name, for_testing_purposes=True)
        expected_table = pd.DataFrame({"Node": [1, 2, 3, 4, 5, 6], "X": [1, 2, 2, 6, 10, 4], "Y": [2, 5, 1, 2, 5, 3], 
                                       "Type": ["Start", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint"]})
        
        pd.testing.assert_frame_equal(downcast_to_int64(nodes), expected_table)

    # Invalidate Column(s) having incorrect names
    def test_wrong_column_names(self):
        with pytest.raises(ValueError) as e:
            import_node_data(self.wrong_column_names_name, for_testing_purposes=True)

    # Invalidate incorrect number of columns
    def test_wrong_number_of_columns(self):
        with pytest.raises(ValueError) as e:
            import_node_data(self.wrong_number_of_columns_name, for_testing_purposes=True)

    # Validate that Valid Input Data is accepted
    def test_valid_input_data(self):
        nodes = import_node_data(self.valid_name, for_testing_purposes=True)
        expected_table = pd.DataFrame({"Node": [1, 2, 3, 4, 5, 6, 7], "X": [1, 2, 2, 6, 10, 4, 5], "Y": [2, 5, 1, 2, 5, 3, 7], 
                                       "Type": ["Start", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint"]})
        
        pd.testing.assert_frame_equal(downcast_to_int64(nodes), expected_table)

    # Validate that Valid Input Data w/ a 'Name' Column is accepted
    def test_valid_input_data_with_name_column(self):
        nodes = import_node_data(self.valid_name_name, for_testing_purposes=True)
        expected_table = pd.DataFrame({"Node": [1, 2, 3, 4, 5, 6, 7], "X": [1, 2, 2, 6, 10, 4, 5], "Y": [2, 5, 1, 2, 5, 3, 7], 
                                       "Type": ["Start", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint", "Waypoint"],
                                       "Name": [pd.NA, pd.NA, pd.NA, "Corner Shop", pd.NA, pd.NA, pd.NA]})
        
        pd.testing.assert_frame_equal(downcast_to_int64(nodes), expected_table)

if __name__ == "__main__":
    unittest.main(warnings = "ignore")