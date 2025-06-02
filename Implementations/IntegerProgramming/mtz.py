import sys
sys.path.append("../")
from TSP_Utilities import tsp_utility_functions as tsp

import pyomo.environ as pyo, numpy as np, time
from pyomo.opt import SolverFactory

#============================================================Miller-Tucker-Zemlin Implementation============================================================
# Optional Parameter "gap": Configure Solver to stop once value within this "gap" is found
# Optional Parameter "time_limit": Configure Solver to have a Maximum Execution Time to Solve Problem
def formulate_and_solve_mtz(nodes, distance_matrix):
    #====================Create Model====================
    model = pyo.ConcreteModel()
    n_nodes = len(nodes)

    #====================Variables====================
    model.x = pyo.Var(range(n_nodes), range(n_nodes), within=pyo.Binary)
    x = model.x

    model.u = pyo.Var(range(n_nodes), within=pyo.Integers)
    u = model.u

    model.M = pyo.Var(range(n_nodes), range(n_nodes), within=pyo.NonNegativeReals) # Big-M
    M = model.M

    #====================Objective Function====================
    obj_expr = sum(sum(distance_matrix[i, j] * x[i, j] for i in range(n_nodes)) for j in range(n_nodes))
    model.obj = pyo.Objective(expr = obj_expr)

    #====================Constraints====================
    model.C1 = pyo.ConstraintList()
    for i in range(n_nodes):
        model.C1.add(sum([x[i, j] for j in range(n_nodes) if i != j]) == 1)

    model.C2 = pyo.ConstraintList()
    for i in range(n_nodes):
        model.C2.add(sum([x[j, i] for j in range(n_nodes) if i != j]) == 1)

    model.C3 = pyo.Constraint(expr = u[0] == 1)

    model.C4 = pyo.ConstraintList()
    for i in range(1, n_nodes):
        model.C4.add(pyo.inequality(2, u[i], n_nodes)) # model.C4.add(pyo.inequality(1, u[i], n_nodes - 1))
    # model.C4 = pyo.ConstraintList()
    # for i in range(1, n_nodes):
    #     model.C4.add(pyo.inequality(1, u[i], n_nodes - 1)) # model.C4.add(pyo.inequality(1, u[i], n_nodes - 1))


    #====================vvThe 5th Constraint has been split up into 2 Constraints in order to Linearise it according to Big-M====================
    model.C5 = pyo.ConstraintList()
    for i in range(1, n_nodes):
        for j in range(1, n_nodes):
            if i != j:
                model.C5.add(u[i] - u[j] + 1 - M[i, j] <= 0)  

    model.C6 = pyo.ConstraintList()
    for i in range(1, n_nodes):
        for j in range(1, n_nodes):
            if i != j:
                model.C6.add(M[i, j] <= (n_nodes - 1)*(1 - x[i, j]))
    # model.C5 = pyo.ConstraintList()
    # for i in range(1, n_nodes):
    #     for j in range(1, n_nodes):
    #         if i != j:
    #             model.C5.add(u[i] - u[j] + 1 <= (n_nodes - 1)*(1 - x[i, j]))
    #====================^^The 5th Constraint has been split up into 2 Constraints in order to Linearise it according to Big-M====================
    # model.pprint()
                
    #====================Solve====================
    opt = SolverFactory('gurobi')
    # opt.options['TimeLimit'] = 400
    # opt.options['MIPGap'] = 0.05
    result = opt.solve(model, tee=True)
    # result = opt.solve(model)

    #====================Process/Return Results====================
    # Form Solution Path Matrix
    solution_path_matrix = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:  
                solution_path_matrix[i, j] = pyo.value(x[i, j])
            else:
                solution_path_matrix[i, j] = 0 

    # print(solution_path_matrix)

    # Calculate Total Route Distance
    distance_travelled = np.sum(solution_path_matrix * distance_matrix)

    # Obtain Route as List of Indexed Nodes
    start_node_index = nodes.index[nodes["Type"] == "Start"][0]
    route = [start_node_index]
    current_node = start_node_index

    while True:
        next_node = np.argmax(solution_path_matrix[current_node])
        if next_node == start_node_index and len(route) == n_nodes:
            route.append(next_node)
            break
        route.append(next_node)
        current_node = next_node

    # Convert Route of Indexed Nodes to Numbered Nodes
    numbered_route = list(nodes.loc[route, "Node"])

    return numbered_route, np.round(distance_travelled, 2), model

#============================================================Run Algorithm============================================================
def run_mtz_generic(file_name, import_node_data_func, display_route=True, name="Integer Programming (MTZ)"):
    nodes = import_node_data_func(file_name)
    distance_matrix = tsp.compute_distance_matrix(nodes)
    solution_with_model = formulate_and_solve_mtz(nodes, distance_matrix)
    solution = solution_with_model[:2]
    if display_route:
        tsp.display_route_as_graph(nodes, solution, name)
    
    return solution_with_model

def run_mtz(file_name, display_route=True):
    return run_mtz_generic(file_name, tsp.import_node_data, display_route)

def run_mtz_tsp_lib(file_name, display_route=True):
    spreadsheet_name = tsp.convert_tsp_lib_instance_to_spreadsheet(file_name)
    return run_mtz_generic(spreadsheet_name, tsp.import_node_data_tsp_lib, display_route)


# print(run_mtz_tsp_lib("berlin52.tsp")[:2])
# print(run_mtz_tsp_lib("ulysses16.tsp")[:2])
# print(run_mtz_tsp_lib("tsp225.tsp")[:2])
# print(run_mtz_tsp_lib("pr107.tsp")[:2])
# print(run_mtz("test_city_1.xlsx")[:2])