import numpy as np, plotly.graph_objects as go, pandas as pd, Levenshtein
from scipy.spatial.distance import cdist

# Import .tsp files from TSPLIB and convert them into Spreadsheets to be processed for extracting Node Data
def convert_tsp_lib_instance_to_spreadsheet(file_name):
    lines = []
    with open(f"../TSP_Utilities/Test_Inputs/TSPLIB_Instances/{file_name}", "r") as my_file:
        lines = my_file.readlines()

    name = [line for line in lines if line.startswith("NAME")][0]
    spreadsheet_file_name = name.split(":")[1].strip() + ".xlsx"

    node, x, y = [], [], []
    coordinate_section = False

    for line in lines:
        if coordinate_section:
            if line.strip() == "EOF":
                break
            else:
                node_columns = line.split()
                node.append(node_columns[0])
                x.append(node_columns[1])
                y.append(node_columns[2])
        elif line.startswith("NODE_COORD_SECTION"):
            coordinate_section = True

    nodes = pd.DataFrame({"Node": node, "X": x, "Y": y})
    nodes["Type"] = "Waypoint"
    nodes.loc[0, "Type"] = "Start"

    file_path = f"../TSP_Utilities/Test_Inputs/TSPLIB_Instances/{spreadsheet_file_name}"
    nodes.to_excel(file_path, sheet_name = "Nodes")

    return spreadsheet_file_name

#====================Import Node Data from Spreadsheet (Includes Robust Validation) ====================
def import_node_data(file_name, for_testing_purposes=False):
    if for_testing_purposes:
        nodes = pd.read_excel(f"../TSP_Utilities/Test_Inputs/Invalid_TSP_Instances_Testing/{file_name}", sheet_name = "Nodes")
    else:
        nodes = pd.read_excel(f"../TSP_Utilities/Test_Inputs/TSP_Instances/{file_name}", sheet_name = "Nodes")

    # Check for Duplicate Columns
    actual_column_names = [column.split('.')[0] for column in nodes.columns]
    if len(actual_column_names) != len(set(actual_column_names)):
        raise ValueError("Duplicate column names found. Please ensure that each column name is unique & valid. 'Node', 'X', 'Y', 'Type' are valid and required. 'Name' is an optional column.")
    
    # Check that all columns are valid
    required_columns = ['Node', 'X', 'Y', 'Type']
    optional_columns = ['Name']
    valid_columns = required_columns + optional_columns

    columns_set = set(nodes.columns)

    missing_required_columns = set(required_columns) - columns_set # Check for missing columns that're required
    if len(missing_required_columns) > 0:
        raise ValueError(f"Missing required column(s) found: {', '.join(missing_required_columns)}. Please ensure that each column name is unique & valid. 'Node', 'X', 'Y', 'Type' are valid and required. 'Name' is an optional column.")
    
    surplus_columns = columns_set - set(valid_columns) # Check for additional surplus columns that aren't valid
    if len(surplus_columns) > 0:
        raise ValueError(f"Surplus column(s) found: {', '.join(surplus_columns)}. Please ensure that each column name is unique & valid. 'Node', 'X', 'Y', 'Type' are valid and required. 'Name' is an optional column.")

    # Remove rows where all cells are empty (NaN)
    nodes_cleaned = nodes.dropna(how='all') 

    # Check for singular cells that're empty (NaN)... except in the 'Name' column, where it's OK to have empty Cells
    columns_to_check = nodes_cleaned.columns.difference(['Name'])
    if nodes_cleaned[columns_to_check].isnull().any(axis=1).any():
        raise ValueError("Please ensure that all cells have been filled out entirely, excluding cells in the 'Name' column")

    # Validate Column Data types ("Node", "X", "Y" Columns)
    if not np.issubdtype(nodes_cleaned['Node'].dtype, np.number) or (nodes_cleaned['Node'] < 0).any():
        raise ValueError("Please ensure that all values in the 'Node' column are non-negative integers.")
    
    if not np.issubdtype(nodes_cleaned['X'].dtype, np.number) or not np.issubdtype(nodes_cleaned['Y'].dtype, np.number):
        raise ValueError("Please ensure that all values in the 'X' and 'Y' columns are numeric values.")
    
    # Remove nodes w/ duplicate coordinates... keep the first one
    nodes_cleaned = nodes_cleaned.drop_duplicates(subset=['X', 'Y'], keep='first')

    # Validate (and potentially auto-correct) "Type" Column
    valid_types = ['Start', 'Waypoint']
    errors = []
    for (index, value) in nodes_cleaned.iterrows():
        if value['Type'] not in valid_types:
            distance_from_start = Levenshtein.distance(str(value['Type']), 'Start')
            distance_from_waypoint = Levenshtein.distance(str(value['Type']), 'Waypoint')
            
            # Auto-correct if Levenshtein distance is <= 2
            if distance_from_start <= 2:
                nodes_cleaned.at[index, 'Type'] = 'Start'
            elif distance_from_waypoint <= 2:
                nodes_cleaned.at[index, 'Type'] = 'Waypoint'
            else:
                errors.append(f" - '{value['Type']}'")

    if len(errors) > 0:
        raise ValueError("Misspelling error(s) found in 'Type' column. Only 'Start' and 'Waypoint' are valid. Instead, the following were found: \n" + "\n".join(errors))
    
    # More than 1 Start Node, or no Start Nodes
    start_node_count = len(nodes_cleaned[nodes_cleaned['Type'] == 'Start'])
    if start_node_count != 1:
        raise ValueError(f"Please ensure that there's exactly one 'Start' node. You entered {start_node_count}.")
    
    # Invalidate Inputs w/ < than 3 Nodes
    n_nodes = len(nodes_cleaned)
    if n_nodes < 3:
        raise ValueError(f"You've only entered {n_nodes}. Please ensure you enter at least 3.")
    
    # Order Nodes by Node Number
    nodes_cleaned = nodes_cleaned.sort_values(by='Node').reset_index(drop=True)

    # Start Node is not Node Number 1
    start_node_index = nodes_cleaned[nodes_cleaned['Type'] == 'Start'].index[0]
    start_node_number = nodes_cleaned.loc[start_node_index, 'Node']

    if start_node_number != 1:
        nodes_cleaned.loc[:start_node_index - 1, "Node"] += 1
        nodes_cleaned.loc[start_node_index, 'Node'] = 1
        nodes_cleaned = nodes_cleaned.sort_values(by='Node').reset_index(drop=True)

    # Ensure all Node Numbers are from 1 --> n
    nodes_cleaned['Node'] = range(1, len(nodes_cleaned) + 1)

    


    # print(nodes_cleaned)

    return nodes_cleaned

# Import TSPLIB Node Data from Spreadsheet
def import_node_data_tsp_lib(file_name):
    nodes = pd.read_excel(f"../TSP_Utilities/Test_Inputs/TSPLIB_Instances/{file_name}", sheet_name = "Nodes")

    return nodes

def map_nodes_to_index(nodes):
    return {node: index for (index, node) in enumerate(nodes['Node'])}

# Compute Distance Matrix - Euclidean Distances between each Node
def compute_distance_matrix(nodes):
    coordinate_list = list(zip(nodes['X'], nodes['Y']))
    coordinate_array = np.array(coordinate_list)
    
    distance_matrix = cdist(coordinate_array, coordinate_array, metric='euclidean')
                
    return distance_matrix

def compute_route_distance(route, distance_matrix, node_index_mapping):
    return sum([distance_matrix[node_index_mapping[route[i]], node_index_mapping[route[i + 1]]] for i in range(len(route) - 1)])

# Display Route as Graph
def display_route_as_graph(nodes, solution, name):
    route, distance_travelled = solution
    n_nodes = len(route)

    fig = go.Figure()

    # Plot Edges
    for i in range(n_nodes - 1):
        start_node_number = nodes[nodes["Node"] == route[i]]
        end_node_number = nodes[nodes["Node"] == route[i+1]]
        coordinates_start = start_node_number[["X", "Y"]].values[0]
        coordinates_end = end_node_number[["X", "Y"]].values[0]
        
        line_distance = np.linalg.norm(coordinates_start - coordinates_end)
        
        fig.add_trace(go.Scatter(x=[coordinates_start[0], coordinates_end[0]], 
                                 y=[coordinates_start[1], coordinates_end[1]],
                                 mode="lines, markers", line=dict(color="#007bff"), 
                                 marker=dict(size=10, color="#ffd700"),  
                                 text=[f"Node {route[i]} to Node {route[i+1]}<br>Distance: {np.round(line_distance, 2)}"], hoverinfo="text", 
                                 showlegend=False))

    # Plot Nodes
    for _, node in nodes.iterrows():
        x_coordinate = node["X"]
        y_coordinate = node["Y"]
        node_number = node["Node"]
        
        fig.add_trace(go.Scatter(x=[x_coordinate], 
                                 y=[y_coordinate],
                                 mode="text, markers", 
                                 marker=dict(size=12, symbol="star" if node["Type"] == "Start" else "circle", color="DarkOrange" if node["Type"] == "Start" else "#17becf"),
                                 text=[f"{node_number}"],
                                 textposition="bottom center", hoverinfo="text", hovertext=f"Node {node_number}<br>Coordinates: ({x_coordinate}, {y_coordinate})",
                                 showlegend=False))

    # Legend
    fig.add_trace(go.Scatter(x=[None], 
                             y=[None], 
                             mode='markers', 
                             marker=dict(size=12, symbol='star', color='DarkOrange'),
                             showlegend=True, 
                             name='Start Node'))
    
    # Configure Graph
    fig.update_layout(title=f"TSP Solution - {name} - Distance: {np.round(distance_travelled, 2)}",
                      title_font_size=20,
                      xaxis=dict(showline=True, showgrid=False, linecolor='#333'),
                      yaxis=dict(showline=True, showgrid=False, linecolor='#333'),
                      xaxis_title="Latitude",
                      yaxis_title="Longitude",
                      plot_bgcolor="rgba(0,0,0,0)", # Transparent background for later on when we integrate w/ HTML
                      legend_title_text="Legend: ",
                      legend=dict(orientation= "v", yanchor="middle", y=1.02, xanchor="right", x=1.05))

    fig.show()