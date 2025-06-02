"""This file aims to run all implemented algorithms for the standard VRP against
each other, comparing them for things like runtime, solution quality, etc."""

"""     One key thing to note is that these runtimes include the fully integrated process, 
        from reading the files to displaying the calculated route as a graph        """

import sys, time, numpy as np, plotly.graph_objects as go
sys.path.append("../")
from NearestNeighbour import nearest_neighbour
from IHCV import ihcv
from IntegerProgramming import mtz
from TwoOpt import two_opt
from LargeNeighbourhood import lns
from GeneticAlgorithm import genetic_algorithm as ga

# Compute Average Runtime & CPU Time after "n" Iterations
def compute_average_times(algorithm, dataset, n):
    times = []
    cpu_times = []

    for _ in range(n):
        start_time = time.time()
        start_cpu_time = time.process_time()

        algorithm(*dataset)

        end_time = time.time()
        end_cpu_time = time.process_time()

        time_elapsed = end_time - start_time
        cpu_time_elapsed = end_cpu_time - start_cpu_time

        times.append(time_elapsed)
        cpu_times.append(cpu_time_elapsed)

    average_runtime = np.mean(times)
    average_cpu_times = np.mean(cpu_times)

    return f"{average_runtime}s", f"{average_cpu_times}s",

def compute_percentage_difference(generated_solution, optimal_solution):
    percentage_difference = (abs(generated_solution - optimal_solution) / optimal_solution) * 100

    return percentage_difference


# # Insertion Heuristic w/ Convex Hull
ihcv_tc1_runtime = compute_average_times(ihcv.run_ihcv, ("Instance_For_Presentation.xlsx",), 10)
print(ihcv_tc1_runtime, "IHCV Runtime/CPU Time")

# # GA
ga_tc1_runtime = compute_average_times(ga.run_ga, ("Instance_For_Presentation.xlsx",), 10)
print(ga_tc1_runtime, "GA Runtime/CPU Time")

# # Insertion Heuristic w/ Convex Hull
ihcv_tc1_solution = ihcv.run_ihcv("Instance_For_Presentation.xlsx")[1]
print(ihcv_tc1_solution, "IHCV")

# # Genetic Algorithm
ga_tc1_solution = ga.run_ga("Instance_For_Presentation.xlsx")[1]
print(ga_tc1_solution, "GA")



# #==============================Dataset 1 - Test City 1 - 7 Nodes - Runtime, CPU Time==============================
# # Nearest Neighbour - 0.2912076759338379s, 0.06434819999999997s (10 runs)
# nn_tc1_runtime = compute_average_times(nearest_neighbour.run_nearest_neighbour, ("test_city_1.xlsx",), 10)
# print(nn_tc1_runtime)

# # Insertion Heuristic w/ Convex Hull - 0.31925201416015625s, 0.05864439999999984s (10 runs)
# ihcv_tc1_runtime = compute_average_times(ihcv.run_ihcv, ("test_city_1.xlsx",), 10)
# print(ihcv_tc1_runtime)

# # Integer Programming - 0.48547115325927737s, 0.08146700000000004s (10 runs)
# ip_tc1_runtime = compute_average_times(mtz.run_mtz, ("test_city_1.xlsx",), 10)
# print(ip_tc1_runtime)

# # Nearest Neighbour --> Two Opt - 0.3369459629058838s, 0.058959400000000176s (10 runs)
# nn_two_opt_tc1_runtime = compute_average_times(two_opt.run_two_opt_nearest_neighbour, ("test_city_1.xlsx",), 10)
# print(nn_two_opt_tc1_runtime)

# # IHCV --> Two Opt - 0.398201584815979s, 0.05850529999999994s (10 runs)
# ihcv_two_opt_tc1_runtime = compute_average_times(two_opt.run_two_opt_ihcv, ("test_city_1.xlsx",), 10)
# print(ihcv_two_opt_tc1_runtime)

# # NN --> LNS - 0.593600082397461s, 0.25243659999999996s (10 runs)
# nn_lns_tc1_runtime = compute_average_times(lns.run_lns_nearest_neighbour, ("test_city_1.xlsx",), 10)
# print(nn_lns_tc1_runtime)

# # IHCV --> LNS - 0.6017665624618531s, 0.24799320000000016s (10 runs)
# ihcv_lns_tc1_runtime = compute_average_times(lns.run_lns_ihcv, ("test_city_1.xlsx",), 10)
# print(ihcv_lns_tc1_runtime)

# # NN --> Two Opt --> LNS - 0.6001407957077026s, 0.2650253s (10 runs)
# nn_two_opt_lns_tc1_runtime = compute_average_times(lns.run_lns_two_opt_nearest_neighbour, ("test_city_1.xlsx",), 10)
# print(nn_two_opt_lns_tc1_runtime)

# # IHCV --> Two Opt --> LNS - 0.6019640445709229s, 0.2553482999999999s (10 runs)
# ihcv_two_opt_lns_tc1_runtime = compute_average_times(lns.run_lns_two_opt_ihcv, ("test_city_1.xlsx",), 10)
# print(ihcv_two_opt_lns_tc1_runtime)

# # Genetic Algorithm - 0.46117939949035647s, 0.16895220000000002s (10 runs)
# ga_tc1_runtime = compute_average_times(ga.run_ga, ("test_city_1.xlsx",), 10)
# print(ga_tc1_runtime)

"""
GA PARAMS USED:
    POPULATION_SIZE = 100
    GENERATIONS = 20
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.7
"""

# #==============================Solution Quality============================== - Optimal = 23.63
# # Nearest Neighbour - 29.96
# nn_tc1_solution = nearest_neighbour.run_nearest_neighbour("test_city_1.xlsx")[1]
# print(nn_tc1_solution)

# # Insertion Heuristic w/ Convex Hull - 23.63
# ihcv_tc1_solution = ihcv.run_ihcv("test_city_1.xlsx")[1]
# print(ihcv_tc1_solution)

# # Integer Programming - 23.63
# ip_tc1_solution = mtz.run_mtz("test_city_1.xlsx")[1]
# print(ip_tc1_solution)

# # Nearest Neighbour --> Two Opt - 23.63
# nn_two_opt_tc1_solution = two_opt.run_two_opt_nearest_neighbour("test_city_1.xlsx")[1]
# print(nn_two_opt_tc1_solution)

# # IHCV --> Two Opt - 23.63
# ihcv_two_opt_tc1_solution = two_opt.run_two_opt_ihcv("test_city_1.xlsx")[1]
# print(ihcv_two_opt_tc1_solution)

# # NN --> LNS - 23.63
# nn_lns_tc1_solution = lns.run_lns_nearest_neighbour("test_city_1.xlsx")[1]
# print(nn_lns_tc1_solution)

# # IHCV --> LNS - 23.63
# ihcv_lns_tc1_solution = lns.run_lns_ihcv("test_city_1.xlsx")[1]
# print(ihcv_lns_tc1_solution)

# # NN --> Two Opt --> LNS - 23.63
# nn_two_opt_lns_tc1_solution = lns.run_lns_two_opt_nearest_neighbour("test_city_1.xlsx")[1]
# print(nn_two_opt_lns_tc1_solution)

# # IHCV --> Two Opt --> LNS - 23.63
# ihcv_two_opt_lns_tc1_solution = lns.run_lns_two_opt_ihcv("test_city_1.xlsx")[1]
# print(ihcv_two_opt_lns_tc1_solution)

# # Genetic Algorithm - 23.63
# ga_tc1_solution = ga.run_ga("test_city_1.xlsx")[1]
# print(ga_tc1_solution)

"""
GA PARAMS USED:
    POPULATION_SIZE = 100
    GENERATIONS = 20
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.7
"""



# #==============================Dataset 2 - ulysses16 - 16 Nodes - Runtime==============================
# # Nearest Neighbour - 0.3927781581878662s, 0.06856299999999979s (10 runs)
# nn_ulysses16_runtime = compute_average_times(nearest_neighbour.run_nearest_neighbour_tsp_lib, ("ulysses16.tsp",), 10)
# print(nn_ulysses16_runtime)

# # Insertion Heuristic w/ Convex Hull - 0.4028491497039795s, 0.06983930000000002s (10 runs)
# ihcv_ulysses16_runtime = compute_average_times(ihcv.run_ihcv_tsp_lib, ("ulysses16.tsp",), 10)
# print(ihcv_ulysses16_runtime)

# # Integer Programming - 1.9439756274223328s, 0.0933790000000001s (4 runs)
# ip_ulysses16_runtime = compute_average_times(mtz.run_mtz_tsp_lib, ("ulysses16.tsp",), 4)
# print(ip_ulysses16_runtime)

# # Nearest Neighbour --> Two Opt - 0.4849355220794678s, 0.07274639999999995s (10 runs)
# nn_two_opt_ulysses16_runtime = compute_average_times(two_opt.run_two_opt_nearest_neighbour_tsp_lib, ("ulysses16.tsp",), 10)
# print(nn_two_opt_ulysses16_runtime)

# # IHCV --> Two Opt - 0.4879587411880493s, 0.06585739999999998s (10 runs)
# ihcv_two_opt_ulysses16_runtime = compute_average_times(two_opt.run_two_opt_ihcv_tsp_lib, ("ulysses16.tsp",), 10)
# print(ihcv_two_opt_ulysses16_runtime)

# # NN --> LNS - 0.7752989768981934s, 0.42592280000000005s (10 runs)
# nn_lns_ulysses16_runtime = compute_average_times(lns.run_lns_nearest_neighbour_tsp_lib, ("ulysses16.tsp",), 10)
# print(nn_lns_ulysses16_runtime)

# # IHCV --> LNS - 0.7552976131439209s, 0.41674609999999995s (10 runs)
# ihcv_lns_ulysses16_runtime = compute_average_times(lns.run_lns_ihcv_tsp_lib, ("ulysses16.tsp",), 10)
# print(ihcv_lns_ulysses16_runtime)

# # NN --> Two Opt --> LNS - 0.8133480072021484s, 0.4447246999999999s (10 runs)
# nn_two_opt_lns_ulysses16_runtime = compute_average_times(lns.run_lns_two_opt_nearest_neighbour_tsp_lib, ("ulysses16.tsp",), 10)
# print(nn_two_opt_lns_ulysses16_runtime)

# # IHCV --> Two Opt --> LNS - 0.8137904930114746s, 0.4334836999999997s (10 runs)
# ihcv_two_opt_lns_ulysses16_runtime = compute_average_times(lns.run_lns_two_opt_ihcv_tsp_lib, ("ulysses16.tsp",), 10)
# print(ihcv_two_opt_lns_ulysses16_runtime)

# # Genetic Algorithm - 2.077482557296753s, 1.8290260999999997s (10 runs)
# ga_ulysses16_runtime = compute_average_times(ga.run_ga_tsp_lib, ("ulysses16.tsp",), 10)
# print(ga_ulysses16_runtime)

"""
GA PARAMS USED:
    POPULATION_SIZE = 250
    GENERATIONS = 80
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.7
"""

# #==============================Solution Quality============================== - TSPLIB's Optimal = 74
# # Nearest Neighbour - 104.73
# nn_ulysses16_solution = nearest_neighbour.run_nearest_neighbour_tsp_lib("ulysses16.tsp")[1]
# print(nn_ulysses16_solution)

# # Insertion Heuristic w/ Convex Hull - 75.11
# ihcv_ulysses16_solution = ihcv.run_ihcv_tsp_lib("ulysses16.tsp")[1]
# print(ihcv_ulysses16_solution)

# # Integer Programming - 73.99
# ip_ulysses16_solution = mtz.run_mtz_tsp_lib("ulysses16.tsp")[1]
# print(ip_ulysses16_solution)

# # Nearest Neighbour --> Two Opt - 75.97
# nn_two_opt_ulysses16_solution = two_opt.run_two_opt_nearest_neighbour_tsp_lib("ulysses16.tsp")[1]
# print(nn_two_opt_ulysses16_solution)

# # IHCV --> Two Opt - 73.99
# ihcv_two_opt_ulysses16_solution = two_opt.run_two_opt_ihcv_tsp_lib("ulysses16.tsp")[1]
# print(ihcv_two_opt_ulysses16_solution)

# # NN --> LNS - 73.99
# nn_lns_ulysses16_solution = lns.run_lns_nearest_neighbour_tsp_lib("ulysses16.tsp")[1]
# print(nn_lns_ulysses16_solution)

# # IHCV --> LNS - 73.99
# ihcv_lns_ulysses16_solution = lns.run_lns_ihcv_tsp_lib("ulysses16.tsp")[1]
# print(ihcv_lns_ulysses16_solution)

# # NN --> Two Opt --> LNS - 73.99
# nn_two_opt_lns_ulysses16_solution = lns.run_lns_two_opt_nearest_neighbour_tsp_lib("ulysses16.tsp")[1]
# print(nn_two_opt_lns_ulysses16_solution)

# # IHCV --> Two Opt --> LNS - 73.99
# ihcv_two_opt_lns_ulysses16_solution = lns.run_lns_two_opt_ihcv_tsp_lib("ulysses16.tsp")[1]
# print(ihcv_two_opt_lns_ulysses16_solution)

# # Genetic Algorithm - 73.99
# ga_ulysses16_solution = ga.run_ga_tsp_lib("ulysses16.tsp")[1]
# print(ga_ulysses16_solution)

"""
GA PARAMS USED:
    POPULATION_SIZE = 250
    GENERATIONS = 80
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.7
"""


# #==============================Dataset 3 - berlin52 - 52 Nodes - Runtime, CPU Time==============================
# # Nearest Neighbour - 0.4872169017791748s, 0.1028905s (10 runs)
# nn_berlin52_runtime = compute_average_times(nearest_neighbour.run_nearest_neighbour_tsp_lib, ("berlin52.tsp",), 10)
# print(nn_berlin52_runtime)

# # Insertion Heuristic w/ Convex Hull - 0.5403737020492554s, 0.11625139999999998s (10 runs)
# ihcv_berlin52_runtime = compute_average_times(ihcv.run_ihcv_tsp_lib, ("berlin52.tsp",), 10)
# print(ihcv_berlin52_runtime)

# # Integer Programming - 2.188145935535431s, 0.3090054999999998s (4 runs)
# ip_berlin52_runtime = compute_average_times(mtz.run_mtz_tsp_lib, ("berlin52.tsp",), 4)
# print(ip_berlin52_runtime)

# # Nearest Neighbour --> Two Opt - 0.5839934539794921s, 0.12024610000000022s (10 runs)
# nn_two_opt_berlin52_runtime = compute_average_times(two_opt.run_two_opt_nearest_neighbour_tsp_lib, ("berlin52.tsp",), 10)
# print(nn_two_opt_berlin52_runtime)

# # IHCV --> Two Opt - 0.5570610189437866s, 0.1256982999999999s (10 runs)
# ihcv_two_opt_berlin52_runtime = compute_average_times(two_opt.run_two_opt_ihcv_tsp_lib, ("berlin52.tsp",), 10)
# print(ihcv_two_opt_berlin52_runtime)

# # NN --> LNS - 2.462945890426636s, 2.3432852s (10 runs)
# nn_lns_berlin52_runtime = compute_average_times(lns.run_lns_nearest_neighbour_tsp_lib, ("berlin52.tsp",), 10)
# print(nn_lns_berlin52_runtime)

# # IHCV --> LNS - 2.0279680490493774s, 1.7926807999999999s (10 runs)
# ihcv_lns_berlin52_runtime = compute_average_times(lns.run_lns_ihcv_tsp_lib, ("berlin52.tsp",), 10)
# print(ihcv_lns_berlin52_runtime)

# # NN --> Two Opt --> LNS - 2.0701633214950563s, 1.8235629999999996s (10 runs)
# nn_two_opt_lns_berlin52_runtime = compute_average_times(lns.run_lns_two_opt_nearest_neighbour_tsp_lib, ("berlin52.tsp",), 10)
# print(nn_two_opt_lns_berlin52_runtime)

# # IHCV --> Two Opt --> LNS - 2.3267541885375977s, 1.7669904000000003s (10 runs)
# ihcv_two_opt_lns_berlin52_runtime = compute_average_times(lns.run_lns_two_opt_ihcv_tsp_lib, ("berlin52.tsp",), 10)
# print(ihcv_two_opt_lns_berlin52_runtime)

# # Genetic Algorithm - 11.650958061218262s, 6.796694400000002s (10 runs)
# ga_berlin52_runtime = compute_average_times(ga.run_ga_tsp_lib, ("berlin52.tsp",), 10)
# print(ga_berlin52_runtime)

"""
GA PARAMS USED:
    POPULATION_SIZE = 250
    GENERATIONS = 80
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.7
"""

# #==============================Solution Quality============================== - TSPLIB's Optimal = 7542
# # Nearest Neighbour - 8980.92
# nn_berlin52_solution = nearest_neighbour.run_nearest_neighbour_tsp_lib("berlin52.tsp")[1]
# print(nn_berlin52_solution)

# # Insertion Heuristic w/ Convex Hull - 8105.78
# ihcv_berlin52_solution = ihcv.run_ihcv_tsp_lib("berlin52.tsp")[1]
# print(ihcv_berlin52_solution)

# # Integer Programming - 7544.37
# ip_berlin52_solution = mtz.run_mtz_tsp_lib("berlin52.tsp")[1]
# print(ip_berlin52_solution)

# # Nearest Neighbour --> Two Opt - 8056.83 
# nn_two_opt_berlin52_solution = two_opt.run_two_opt_nearest_neighbour_tsp_lib("berlin52.tsp")[1]
# print(nn_two_opt_berlin52_solution)

# # IHCV --> Two Opt - 8074.56
# ihcv_two_opt_berlin52_solution = two_opt.run_two_opt_ihcv_tsp_lib("berlin52.tsp")[1]
# print(ihcv_two_opt_berlin52_solution)

# # NN --> LNS - 7911.33
# nn_lns_berlin52_solution = lns.run_lns_nearest_neighbour_tsp_lib("berlin52.tsp")[1]
# print(nn_lns_berlin52_solution)

# # IHCV --> LNS - 7911.33
# ihcv_lns_berlin52_solution = lns.run_lns_ihcv_tsp_lib("berlin52.tsp")[1]
# print(ihcv_lns_berlin52_solution)

# # NN --> Two Opt --> LNS - 7911.33
# nn_two_opt_lns_berlin52_solution = lns.run_lns_two_opt_nearest_neighbour_tsp_lib("berlin52.tsp")[1]
# print(nn_two_opt_lns_berlin52_solution)

# # IHCV --> Two Opt --> LNS - 7887.23
# ihcv_two_opt_lns_berlin52_solution = lns.run_lns_two_opt_ihcv_tsp_lib("berlin52.tsp")[1]
# print(ihcv_two_opt_lns_berlin52_solution)

# # Genetic Algorithm - 7544.37
# ga_berlin52_solution = ga.run_ga_tsp_lib("berlin52.tsp")[1]
# print(ga_berlin52_solution)

"""
GA PARAMS USED:
    POPULATION_SIZE = 250
    GENERATIONS = 80
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.7
"""



# #==============================Dataset 4 - pr76 - 76 Nodes - Runtime, CPU Time==============================
# # Nearest Neighbour - 0.5912003755569458s, 0.13564180000000023s (10 runs)
# nn_pr76_runtime = compute_average_times(nearest_neighbour.run_nearest_neighbour_tsp_lib, ("pr76.tsp",), 10)
# print(nn_pr76_runtime)

# # Insertion Heuristic w/ Convex Hull - 0.6079960489273072s, 0.15631350000000008s (10 runs)
# ihcv_pr76_runtime = compute_average_times(ihcv.run_ihcv_tsp_lib, ("pr76.tsp",), 10)
# print(ihcv_pr76_runtime)

# # Integer Programming - 238.9238269329071s, 0.6113710000000001s (2 runs)
# ip_pr76_runtime = compute_average_times(mtz.run_mtz_tsp_lib, ("pr76.tsp",), 2)
# print(ip_pr76_runtime)

# # Nearest Neighbour --> Two Opt - 0.6298728704452514s, 0.16514700000000007s (10 runs)
# nn_two_opt_pr76_runtime = compute_average_times(two_opt.run_two_opt_nearest_neighbour_tsp_lib, ("pr76.tsp",), 10)
# print(nn_two_opt_pr76_runtime)

# # IHCV --> Two Opt - 0.6108887052536011s, 0.15282080000000012s (10 runs)
# ihcv_two_opt_pr76_runtime = compute_average_times(two_opt.run_two_opt_ihcv_tsp_lib, ("pr76.tsp",), 10)
# print(ihcv_two_opt_pr76_runtime)

# # NN --> LNS - 4.3152546882629395s, 3.6238146s (10 runs)
# nn_lns_pr76_runtime = compute_average_times(lns.run_lns_nearest_neighbour_tsp_lib, ("pr76.tsp",), 10)
# print(nn_lns_pr76_runtime)

# # IHCV --> LNS - 5.947743972142537s, 4.7678232000000005s (10 runs)
# ihcv_lns_pr76_runtime = compute_average_times(lns.run_lns_ihcv_tsp_lib, ("pr76.tsp",), 10)
# print(ihcv_lns_pr76_runtime)

# # NN --> Two Opt --> LNS - 4.365378888448079s, 3.8795434s (10 runs)
# nn_two_opt_lns_pr76_runtime = compute_average_times(lns.run_lns_two_opt_nearest_neighbour_tsp_lib, ("pr76.tsp",), 10)
# print(nn_two_opt_lns_pr76_runtime)

# # IHCV --> Two Opt --> LNS - 5.3695143063863116s, 4.6750652s (10 runs)
# ihcv_two_opt_lns_pr76_runtime = compute_average_times(lns.run_lns_two_opt_ihcv_tsp_lib, ("pr76.tsp",), 10)
# print(ihcv_two_opt_lns_pr76_runtime)

# # Genetic Algorithm - 51.395593881607056s, 45.3103905s (10 runs)
# ga_pr76_runtime = compute_average_times(ga.run_ga_tsp_lib, ("pr76.tsp",), 10)
# print(ga_pr76_runtime)

"""
GA PARAMS USED:
    POPULATION_SIZE = 450
    GENERATIONS = 250
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.75
"""

# #==============================Solution Quality============================== - TSPLIB's Optimal = 108159
# # Nearest Neighbour - 153461.92
# nn_pr76_solution = nearest_neighbour.run_nearest_neighbour_tsp_lib("pr76.tsp")[1]
# print(nn_pr76_solution)

# # Insertion Heuristic w/ Convex Hull - 114808.11
# ihcv_pr76_solution = ihcv.run_ihcv_tsp_lib("pr76.tsp")[1]
# print(ihcv_pr76_solution)

# # Integer Programming - 108159.44
# ip_pr76_solution = mtz.run_mtz_tsp_lib("pr76.tsp")[1]
# print(ip_pr76_solution)

# # Nearest Neighbour --> Two Opt - 113635.27
# nn_two_opt_pr76_solution = two_opt.run_two_opt_nearest_neighbour_tsp_lib("pr76.tsp")[1]
# print(nn_two_opt_pr76_solution)

# # IHCV --> Two Opt - 113113.89
# ihcv_two_opt_pr76_solution = two_opt.run_two_opt_ihcv_tsp_lib("pr76.tsp")[1]
# print(ihcv_two_opt_pr76_solution)

# # NN --> LNS - 110666.14
# nn_lns_pr76_solution = lns.run_lns_nearest_neighbour_tsp_lib("pr76.tsp")[1]
# print(nn_lns_pr76_solution)

# # IHCV --> LNS - 109044.07
# ihcv_lns_pr76_solution = lns.run_lns_ihcv_tsp_lib("pr76.tsp")[1]
# print(ihcv_lns_pr76_solution)

# # NN --> Two Opt --> LNS - 109067.89
# nn_two_opt_lns_pr76_solution = lns.run_lns_two_opt_nearest_neighbour_tsp_lib("pr76.tsp")[1]
# print(nn_two_opt_lns_pr76_solution)

# # IHCV --> Two Opt --> LNS - 109044.07
# ihcv_two_opt_lns_pr76_solution = lns.run_lns_two_opt_ihcv_tsp_lib("pr76.tsp")[1]
# print(ihcv_two_opt_lns_pr76_solution)

# # Genetic Algorithm - 108159.44
# ga_pr76_solution = ga.run_ga_tsp_lib("pr76.tsp")[1]
# print(ga_pr76_solution)

"""
GA PARAMS USED:
    POPULATION_SIZE = 450
    GENERATIONS = 250
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.75
"""




# # ==============================Dataset 5 - pr107 - 107 Nodes - Runtime, CPU Time==============================
# # Nearest Neighbour - 0.5175444841384887s, 0.17077940000000016s (10 runs)
# nn_pr107_runtime = compute_average_cpu_time(nearest_neighbour.run_nearest_neighbour_tsp_lib, ("pr107.tsp",), 10)
# print(nn_pr107_runtime)

# # Insertion Heuristic w/ Convex Hull - 0.6871454000473023s, 0.2842870999999999s (10 runs)
# ihcv_pr107_runtime = compute_average_cpu_time(ihcv.run_ihcv_tsp_lib, ("pr107.tsp",), 10)
# print(ihcv_pr107_runtime)

# # Integer Programming - 1 hour execution

# # Nearest Neighbour --> Two Opt - 0.8510305881500244s, 0.24584100000000006s (10 runs)
# nn_two_opt_pr107_runtime = compute_average_cpu_time(two_opt.run_two_opt_nearest_neighbour_tsp_lib, ("pr107.tsp",), 10)
# print(nn_two_opt_pr107_runtime)

# # IHCV --> Two Opt - 0.7930902640024821s, 0.2809012s (10 runs)
# ihcv_two_opt_pr107_runtime = compute_average_cpu_time(two_opt.run_two_opt_ihcv_tsp_lib, ("pr107.tsp",), 10)
# print(ihcv_two_opt_pr107_runtime)

# # NN --> LNS - 6.857412974039714s, 6.4553855s (10 runs)
# nn_lns_pr107_runtime = compute_average_cpu_time(lns.run_lns_nearest_neighbour_tsp_lib, ("pr107.tsp",), 10)
# print(nn_lns_pr107_runtime)

# # IHCV --> LNS - 8.85807736714681s, 7.682524599999998s (10 runs)
# ihcv_lns_pr107_runtime = compute_average_cpu_time(lns.run_lns_ihcv_tsp_lib, ("pr107.tsp",), 10)
# print(ihcv_lns_pr107_runtime)

# # NN --> Two Opt --> LNS - 7.071200450261434s, 6.4950489000000005s (10 runs)
# nn_two_opt_lns_pr107_runtime = compute_average_cpu_time(lns.run_lns_two_opt_nearest_neighbour_tsp_lib, ("pr107.tsp",), 10)
# print(nn_two_opt_lns_pr107_runtime)

# # IHCV --> Two Opt --> LNS - 8.895966529846191s, 8.743738600000002s (10 runs)
# ihcv_two_opt_lns_pr107_runtime = compute_average_cpu_time(lns.run_lns_two_opt_ihcv_tsp_lib, ("pr107.tsp",), 10)
# print(ihcv_two_opt_lns_pr107_runtime)

# # Genetic Algorithm - 94.05615496635437s, 74.549643s (10 runs)
# ga_pr107_runtime = compute_average_cpu_time(ga.run_ga_tsp_lib, ("pr107.tsp",), 10)
# print(ga_pr107_runtime)

"""
GA PARAMS USED:
    POPULATION_SIZE = 400
    GENERATIONS = 200
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.85
"""

# #==============================Solution Quality============================== - TSPLIB's Optimal = 44301
# # Nearest Neighbour - 46678.15
# nn_pr107_solution = nearest_neighbour.run_nearest_neighbour_tsp_lib("pr107.tsp")[1]
# print(nn_pr107_solution)

# # Insertion Heuristic w/ Convex Hull - 45730.01
# ihcv_pr107_solution = ihcv.run_ihcv_tsp_lib("pr107.tsp")[1]
# print(ihcv_pr107_solution)

# # Integer Programming (1 hour) - 54606.7574
# ip_pr107_solution = mtz.run_mtz_tsp_lib("pr107.tsp")[1]
# print(ip_pr107_solution)

# # Nearest Neighbour --> Two Opt - 44767.07
# nn_two_opt_pr107_solution = two_opt.run_two_opt_nearest_neighbour_tsp_lib("pr107.tsp")[1]
# print(nn_two_opt_pr107_solution)

# # IHCV --> Two Opt - 45533.19
# ihcv_two_opt_pr107_solution = two_opt.run_two_opt_ihcv_tsp_lib("pr107.tsp")[1]
# print(ihcv_two_opt_pr107_solution)

# # NN --> LNS - 44436.24
# nn_lns_pr107_solution = lns.run_lns_nearest_neighbour_tsp_lib("pr107.tsp")[1]
# print(nn_lns_pr107_solution)

# # IHCV --> LNS - 44481.17
# ihcv_lns_pr107_solution = lns.run_lns_ihcv_tsp_lib("pr107.tsp")[1]
# print(ihcv_lns_pr107_solution)

# # NN --> Two Opt --> LNS - 44436.24
# nn_two_opt_lns_pr107_solution = lns.run_lns_two_opt_nearest_neighbour_tsp_lib("pr107.tsp")[1]
# print(nn_two_opt_lns_pr107_solution)

# # IHCV --> Two Opt --> LNS - 44301.68
# ihcv_two_opt_lns_pr107_solution = lns.run_lns_two_opt_ihcv_tsp_lib("pr107.tsp")[1]
# print(ihcv_two_opt_lns_pr107_solution)

# # Genetic Algorithm - 44436.24
# ga_pr107_solution = ga.run_ga_tsp_lib("pr107.tsp")[1]
# print(ga_pr107_solution)

"""
GA PARAMS USED:
    POPULATION_SIZE = 400
    GENERATIONS = 200
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.85
"""


# #==============================Dataset 6 - pr136 - 136 Nodes - Runtime, CPU Time==============================
# # Nearest Neighbour - 0.5670787572860718s, 0.2090824999999999s (10 runs)
# nn_pr136_runtime = compute_average_cpu_time(nearest_neighbour.run_nearest_neighbour_tsp_lib, ("pr136.tsp",), 10)
# print(nn_pr136_runtime)

# # Insertion Heuristic w/ Convex Hull - 0.6169691801071167s, 0.34389060000000005s (10 runs)
# ihcv_pr136_runtime = compute_average_cpu_time(ihcv.run_ihcv_tsp_lib, ("pr136.tsp",), 10)
# print(ihcv_pr136_runtime)

# # Integer Programming - 1 hour execution

# # Nearest Neighbour --> Two Opt - 1.1399709383646648s, 0.38065879999999996s (10 runs)
# nn_two_opt_pr136_runtime = compute_average_cpu_time(two_opt.run_two_opt_nearest_neighbour_tsp_lib, ("pr136.tsp",), 10)
# print(nn_two_opt_pr136_runtime)

# # IHCV --> Two Opt - 0.8790188789367676s, 0.45419820000000016s (10 runs)
# ihcv_two_opt_pr136_runtime = compute_average_cpu_time(two_opt.run_two_opt_ihcv_tsp_lib, ("pr136.tsp",), 10)
# print(ihcv_two_opt_pr136_runtime)

# # NN --> LNS - 20.448569536209106s, 18.92006533333333s (10 runs)
# nn_lns_pr136_runtime = compute_average_cpu_time(lns.run_lns_nearest_neighbour_tsp_lib, ("pr136.tsp",), 10)
# print(nn_lns_pr136_runtime)

# # IHCV --> LNS - 15.488086581230164s, 12.037884333333333s (10 runs)
# ihcv_lns_pr136_runtime = compute_average_cpu_time(lns.run_lns_ihcv_tsp_lib, ("pr136.tsp",), 10)
# print(ihcv_lns_pr136_runtime)

# # NN --> Two Opt --> LNS - 19.903863668441772s, 11.2766025s (10 runs)
# nn_two_opt_lns_pr136_runtime = compute_average_cpu_time(lns.run_lns_two_opt_nearest_neighbour_tsp_lib, ("pr136.tsp",), 10)
# print(nn_two_opt_lns_pr136_runtime)

# # IHCV --> Two Opt --> LNS - 18.60973048210144s, 13.697148s (10 runs)
# ihcv_two_opt_lns_pr136_runtime = compute_average_cpu_time(lns.run_lns_two_opt_ihcv_tsp_lib, ("pr136.tsp",), 10)
# print(ihcv_two_opt_lns_pr136_runtime)

# # Genetic Algorithm - 749.3404741287231s, 721.638777333333333s (10 runs)
# ga_pr136_runtime = compute_average_cpu_time(ga.run_ga_tsp_lib, ("pr136.tsp",), 10)
# print(ga_pr136_runtime)

"""
GA PARAMS USED:
    POPULATION_SIZE = 750
    GENERATIONS = 500
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.65
"""

# #==============================Solution Quality============================== - TSPLIB's Optimal = 96772
# # Nearest Neighbour - 120777.86
# nn_pr136_solution = nearest_neighbour.run_nearest_neighbour_tsp_lib("pr136.tsp")[1]
# print(nn_pr136_solution)

# # Insertion Heuristic w/ Convex Hull - 102695.68
# ihcv_pr136_solution = ihcv.run_ihcv_tsp_lib("pr136.tsp")[1]
# print(ihcv_pr136_solution)

# # Integer Programming (1 hour) - 97389.78
# ip_pr136_solution = mtz.run_mtz_tsp_lib("pr136.tsp")[1]
# print(ip_pr136_solution)

# # Nearest Neighbour --> Two Opt - 105114.41
# nn_two_opt_pr136_solution = two_opt.run_two_opt_nearest_neighbour_tsp_lib("pr136.tsp")[1]
# print(nn_two_opt_pr136_solution)

# # IHCV --> Two Opt - 99777.24
# ihcv_two_opt_pr136_solution = two_opt.run_two_opt_ihcv_tsp_lib("pr136.tsp")[1]
# print(ihcv_two_opt_pr136_solution)

# # NN --> LNS - 100906.82
# nn_lns_pr136_solution = lns.run_lns_nearest_neighbour_tsp_lib("pr136.tsp")[1]
# print(nn_lns_pr136_solution)

# # IHCV --> LNS - 98357.67
# ihcv_lns_pr136_solution = lns.run_lns_ihcv_tsp_lib("pr136.tsp")[1]
# print(ihcv_lns_pr136_solution)

# # NN --> Two Opt --> LNS - 100787.66
# nn_two_opt_lns_pr136_solution = lns.run_lns_two_opt_nearest_neighbour_tsp_lib("pr136.tsp")[1]
# print(nn_two_opt_lns_pr136_solution)

# # IHCV --> Two Opt --> LNS - 97319.9
# ihcv_two_opt_lns_pr136_solution = lns.run_lns_two_opt_ihcv_tsp_lib("pr136.tsp")[1]
# print(ihcv_two_opt_lns_pr136_solution)

# # Genetic Algorithm - 96860.88
# ga_pr136_solution = ga.run_ga_tsp_lib("pr136.tsp")[1]
# print(ga_pr136_solution)

"""
GA PARAMS USED:
    POPULATION_SIZE = 750
    GENERATIONS = 500
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.65
"""


# #==============================Dataset 7 - tsp225 - 225 Nodes - Runtime, CPU Time==============================
# # Nearest Neighbour - 0.6898097515106201s, 0.32991720000000013s (10 runs)
# nn_tsp225_runtime = compute_average_cpu_time(nearest_neighbour.run_nearest_neighbour_tsp_lib, ("tsp225.tsp",), 10)
# print(nn_tsp225_runtime)

# # Insertion Heuristic w/ Convex Hull - 1.103521466255188s, 0.8435991999999999s (10 runs)
# ihcv_tsp225_runtime = compute_average_cpu_time(ihcv.run_ihcv_tsp_lib, ("tsp225.tsp",), 10)
# print(ihcv_tsp225_runtime)

# # Integer Programming - 1 hour execution

# # Nearest Neighbour --> Two Opt - 1.5792440176010132s, 0.8629096s (10 runs)
# nn_two_opt_tsp225_runtime = compute_average_cpu_time(two_opt.run_two_opt_nearest_neighbour_tsp_lib, ("tsp225.tsp",), 10)
# print(nn_two_opt_tsp225_runtime)

# # IHCV --> Two Opt - 1.6633020639419556s, 1.1295614999999999s (10 runs)
# ihcv_two_opt_tsp225_runtime = compute_average_cpu_time(two_opt.run_two_opt_ihcv_tsp_lib, ("tsp225.tsp",), 10)
# print(ihcv_two_opt_tsp225_runtime)

# # NN --> LNS - 46.42879557609558s, 36.527747s (10 runs)
# nn_lns_tsp225_runtime = compute_average_cpu_time(lns.run_lns_nearest_neighbour_tsp_lib, ("tsp225.tsp",), 10)
# print(nn_lns_tsp225_runtime)

# # IHCV --> LNS - 36.334240078926086s, 34.378658s (10 runs)
# ihcv_lns_tsp225_runtime = compute_average_cpu_time(lns.run_lns_ihcv_tsp_lib, ("tsp225.tsp",), 10)
# print(ihcv_lns_tsp225_runtime)

# # NN --> Two Opt --> LNS - 39.056431531906128s, 29.161218999999996s (10 runs)
# nn_two_opt_lns_tsp225_runtime = compute_average_cpu_time(lns.run_lns_two_opt_nearest_neighbour_tsp_lib, ("tsp225.tsp",), 10)
# print(nn_two_opt_lns_tsp225_runtime)

# # IHCV --> Two Opt --> LNS - 48.61195933818817s, 37.368492s  (10 runs)
# ihcv_two_opt_lns_tsp225_runtime = compute_average_cpu_time(lns.run_lns_two_opt_ihcv_tsp_lib, ("tsp225.tsp",), 10)
# print(ihcv_two_opt_lns_tsp225_runtime)

# # Genetic Algorithm - 2904.7282733239018s, 2332.163935s (10 runs)
# ga_tsp225_runtime = compute_average_cpu_time(ga.run_ga_tsp_lib, ("tsp225.tsp",), 10)
# print(ga_tsp225_runtime)

"""
GA PARAMS USED:
    POPULATION_SIZE = 750
    GENERATIONS = 500
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.65 
"""

# #==============================Solution Quality============================== - TSPLIB's Optimal = 3916
# # Nearest Neighbour - 4829.0
# nn_tsp225_solution = nearest_neighbour.run_nearest_neighbour_tsp_lib("tsp225.tsp")[1]
# print(nn_tsp225_solution)

# # Insertion Heuristic w/ Convex Hull - 4442.27
# ihcv_tsp225_solution = ihcv.run_ihcv_tsp_lib("tsp225.tsp")[1]
# print(ihcv_tsp225_solution)

# # Integer Programming (1 hour) - 4248.42
# ip_tsp225_solution = mtz.run_mtz_tsp_lib("tsp225.tsp")[1]
# print(ip_tsp225_solution)

# # Nearest Neighbour --> Two Opt - 4123.82
# nn_two_opt_tsp225_solution = two_opt.run_two_opt_nearest_neighbour_tsp_lib("tsp225.tsp")[1]
# print(nn_two_opt_tsp225_solution)

# # IHCV --> Two Opt - 4329.24
# ihcv_two_opt_tsp225_solution = two_opt.run_two_opt_ihcv_tsp_lib("tsp225.tsp")[1]
# print(ihcv_two_opt_tsp225_solution)

# # NN --> LNS - 4166.27
# nn_lns_tsp225_solution = lns.run_lns_nearest_neighbour_tsp_lib("tsp225.tsp")[1]
# print(nn_lns_tsp225_solution)

# # IHCV --> LNS - 3989.86
# ihcv_lns_tsp225_solution = lns.run_lns_ihcv_tsp_lib("tsp225.tsp")[1]
# print(ihcv_lns_tsp225_solution)

# # NN --> Two Opt --> LNS - 3946.93
# nn_two_opt_lns_tsp225_solution = lns.run_lns_two_opt_nearest_neighbour_tsp_lib("tsp225.tsp")[1]
# print(nn_two_opt_lns_tsp225_solution)

# # IHCV --> Two Opt --> LNS - 4000.3
# ihcv_two_opt_lns_tsp225_solution = lns.run_lns_two_opt_ihcv_tsp_lib("tsp225.tsp")[1]
# print(ihcv_two_opt_lns_tsp225_solution)

# # Genetic Algorithm - 3895.66
# ga_tsp225_solution = ga.run_ga_tsp_lib("tsp225.tsp")[1]
# print(ga_tsp225_solution)

"""
GA PARAMS USED:
    POPULATION_SIZE = 750
    GENERATIONS = 500
    MUTATION_RATE = 0.35
    CROSSOVER_RATE = 0.65
"""

# #==============================Plot Runtime Graph - all Algorithms==============================
# nodes = [0, 7, 16, 52, 76, 107, 136, 225]
# algorithms = {
#     'Nearest Neighbour': {'data': [0, 0.2912076759338379, 0.3927781581878662, 0.4872169017791748, 0.5912003755569458, 0.5175444841384887, 0.5670787572860718, 0.6898097515106201], 'color': 'blue'},
#     'Insertion Heuristic w/ Convex Hull': {'data': [0, 0.31925201416015625, 0.4028491497039795, 0.5403737020492554, 0.8179960489273072, 0.6871454000473023, 0.6169691801071167, 1.103521466255188], 'color': 'red'},
#     # 'Integer Programming': {'data': [0, 0.48547115325927737, 1.9439756274223328, 2.188145935535431, 238.9238269329071, 3600, 3600, 3600], 'color': 'green'},
#     'Nearest Neighbour --> Two Opt': {'data': [0, 0.3369459629058838, 0.4849355220794678, 0.5839934539794921, 0.6298728704452514, 0.8510305881500244, 1.1399709383646648, 1.5792440176010132], 'color': 'lime'},
#     'IHCV --> Two Opt': {'data': [0, 0.398201584815979, 0.4879587411880493, 0.5570610189437866, 0.6108887052536011, 0.7930902640024821, 0.8790188789367676, 1.6633020639419556], 'color': 'black'},
#     'NN --> LNS': {'data': [0, 0.593600082397461, 0.7752989768981934, 2.462945890426636, 4.3152546882629395, 6.857412974039714, 20.448569536209106, 46.42879557609558], 'color': 'purple'},
#     'IHCV --> LNS': {'data': [0, 0.6017665624618531, 0.7552976131439209, 2.0279680490493774, 5.947743972142537, 8.85807736714681, 15.488086581230164, 36.334240078926086], 'color': 'magenta'},
#     'NN --> Two Opt --> LNS': {'data': [0, 0.6001407957077026, 0.8133480072021484, 2.0701633214950563, 4.365378888448079, 7.071200450261434, 19.903863668441772, 39.056431531906128], 'color': 'olive'},
#     'IHCV --> Two Opt --> LNS': {'data': [0, 0.6019640445709229, 0.8137904930114746, 2.3267541885375977, 5.3695143063863116, 8.895966529846191, 18.60973048210144, 48.61195933818817], 'color': 'aqua'},
#     'Genetic Algorithm': {'data': [0, 0.46117939949035647, 2.077482557296753, 11.650958061218262, 51.395593881607056, 94.05615496635437, 749.3404741287231, 2904.7282733239018], 'color': 'orange'},
# }

# def plot_runtime_comparison(nodes, algorithms_data, title, all_algorithms):
#     fig = go.Figure()
#     for algorithm_name, details in algorithms_data.items():
#         if algorithm_name in all_algorithms:
#             fig.add_trace(go.Scatter(x=nodes, 
#                                      y=details['data'], 
#                                      mode='lines, markers', 
#                                      name=algorithm_name, 
#                                      line=dict(color=details['color'])))
    
#     fig.update_layout(title=title, 
#                       xaxis_title='Number of Nodes', 
#                       yaxis_title='Runtime', 
#                       legend_title='Algorithm: ')
    
#     fig.show()


# plot_runtime_comparison(nodes, algorithms, 'Algorithm Runtime Comparison - All Algorithms', algorithms.keys())
# plot_runtime_comparison(nodes, algorithms, 'Algorithm Runtime Comparison - Excluding Integer Programming', ['Nearest Neighbour', 'Insertion Heuristic w/ Convex Hull', 'Nearest Neighbour --> Two Opt', 'IHCV --> Two Opt', 'NN --> LNS', 'IHCV --> LNS', 'NN --> Two Opt --> LNS', 'IHCV --> Two Opt', 'Genetic Algorithm'])
# plot_runtime_comparison(nodes, algorithms, 'Algorithm Runtime Comparison - Excluding Integer Programming', ['Nearest Neighbour', 'Insertion Heuristic w/ Convex Hull', 'Nearest Neighbour --> Two Opt', 'IHCV --> Two Opt', 'NN --> LNS', 'IHCV --> LNS', 'NN --> Two Opt --> LNS', 'IHCV --> Two Opt --> LNS'])

# # #==============================Plot CPU Time Graph - all Algorithms==============================
# nodes = [0, 7, 16, 52, 76, 107, 136, 225]
# algorithms = {
#     'Nearest Neighbour': {'data': [0, 0.06434819999999997, 0.06856299999999979, 0.1028905, 0.13564180000000023, 0.17077940000000016, 0.2090824999999999, 0.32991720000000013], 'color': 'blue'},
#     'Insertion Heuristic w/ Convex Hull': {'data': [0, 0.05864439999999984, 0.06983930000000002, 0.11625139999999998, 0.15631350000000008, 0.2842870999999999, 0.34389060000000005, 0.8435991999999999], 'color': 'red'},
#     # 'Integer Programming': {'data': [0, 0.08146700000000004, 0.0933790000000001, 0.3090054999999998, 0.6113710000000001, 3600, 3600, 3600], 'color': 'green'},
#     'Nearest Neighbour --> Two Opt': {'data': [0, 0.058959400000000176, 0.07274639999999995, 0.12024610000000022, 0.16514700000000007, 0.24584100000000006, 0.38065879999999996, 0.8629096], 'color': 'lime'},
#     'IHCV --> Two Opt': {'data': [0, 0.05850529999999994, 0.06585739999999998, 0.1256982999999999, 0.15282080000000012, 0.2809012, 0.45419820000000016, 1.1295614999999999], 'color': 'black'},
#     'NN --> LNS': {'data': [0, 0.25243659999999996, 0.42592280000000005, 2.3432852, 3.6238146, 6.4553855, 18.92006533333333, 36.527747], 'color': 'purple'},
#     'IHCV --> LNS': {'data': [0, 0.24799320000000016, 0.41674609999999995, 1.7926807999999999, 4.7678232000000005, 7.682524599999998, 12.037884333333333, 34.378658], 'color': 'magenta'},
#     'NN --> Two Opt --> LNS': {'data': [0, 0.2650253, 0.4447246999999999, 1.8235629999999996, 3.8795434, 6.4950489000000005, 11.2766025, 29.161218999999996], 'color': 'olive'},
#     'IHCV --> Two Opt --> LNS': {'data': [0, 0.2553482999999999, 0.4334836999999997, 1.7669904000000003, 4.6750652, 8.743738600000002, 13.697148, 37.368492], 'color': 'aqua'},
#     'Genetic Algorithm': {'data': [0, 0.16895220000000002, 1.8290260999999997, 6.796694400000002, 45.3103905, 74.549643, 721.638777333333333, 2332.163935], 'color': 'orange'},
# }

# def plot_cpu_time_comparison(nodes, algorithms_data, title, all_algorithms):
#     fig = go.Figure()
#     for algorithm_name, details in algorithms_data.items():
#         if algorithm_name in all_algorithms:
#             fig.add_trace(go.Scatter(x=nodes, 
#                                      y=details['data'], 
#                                      mode='lines, markers', 
#                                      name=algorithm_name, 
#                                      line=dict(color=details['color'])))
    
#     fig.update_layout(title=title, 
#                       xaxis_title='Number of Nodes', 
#                       yaxis_title='CPU Time', 
#                       legend_title='Algorithm: ')
    
#     fig.show()


# plot_cpu_time_comparison(nodes, algorithms, 'Algorithm CPU Time Comparison - All Algorithms', algorithms.keys())
# plot_cpu_time_comparison(nodes, algorithms, 'Algorithm CPU Time Comparison - Excluding Integer Programming', ['Nearest Neighbour', 'Insertion Heuristic w/ Convex Hull', 'Nearest Neighbour --> Two Opt', 'IHCV --> Two Opt', 'NN --> LNS', 'IHCV --> LNS', 'NN --> Two Opt --> LNS', 'IHCV --> Two Opt --> LNS'])




# #==============================Plot Solution Quality Graph - all Algorithms==============================
# nodes = [0, 7, 16, 52, 76, 107, 136, 225]
# algorithms = {
#     'Nearest Neighbour': {'data': [0, 26.79, 41.5, 19.08, 41.89, 5.36, 24.81, 23.31], 'color': 'blue'},
#     'Insertion Heuristic w/ Convex Hull': {'data': [0, 0, 1.5, 7.48, 6.15, 3.22, 6.12, 14.44], 'color': 'red'},
#     # 'Integer Programming': {'data': [0, 0, 0, 0.03, 0, 23.26, 0.64, 8.49], 'color': 'green'},
#     'Nearest Neighbour --> Two Opt': {'data': [0, 0, 2.67, 6.83, 5.06, 1.05, 8.62, 5.31], 'color': 'lime'},
#     'IHCV --> Two Opt': {'data': [0, 0, 0, 7.06, 4.58, 2.78, 3.11, 10.55], 'color': 'black'},
#     'NN --> LNS': {'data': [0, 0, 0, 4.90, 2.31, 0.31, 4.27, 6.39], 'color': 'purple'},
#     'IHCV --> LNS': {'data': [0, 0, 0, 4.90, 0.82, 0.41, 1.64, 1.89], 'color': 'magenta'},
#     'NN --> Two Opt --> LNS': {'data': [0, 0, 0, 4.90, 0.84, 0.31, 4.15, 0.79], 'color': 'olive'},
#     'IHCV --> Two Opt --> LNS': {'data': [0, 0, 0, 4.58, 0.82, 0, 0.57, 2.15], 'color': 'aqua'},
#     'Genetic Algorithm': {'data': [0, 0, 0, 0.03, 0, 0.31, 0.09, -0.52], 'color': 'orange'},
# }

# def plot_runtime_comparison(nodes, algorithms_data, title, all_algorithms):
#     fig = go.Figure()
#     for algorithm_name, details in algorithms_data.items():
#         if algorithm_name in all_algorithms:
#             fig.add_trace(go.Scatter(x=nodes, 
#                                      y=details['data'], 
#                                      mode='lines, markers', 
#                                      name=algorithm_name, 
#                                      line=dict(color=details['color'])))
    
#     fig.update_layout(title=title, 
#                       xaxis_title='Number of Nodes', 
#                       yaxis_title='Solution Quality - Percentage Difference from Global Optimal (%)', 
#                       legend_title='Algorithm: ')
    
#     fig.show()


# plot_runtime_comparison(nodes, algorithms, 'Algorithm Runtime Comparison - All Algorithms', algorithms.keys())
# plot_runtime_comparison(nodes, algorithms, 'Algorithm Solution Quality Comparison - Excluding Integer Programming', ['Nearest Neighbour', 'Insertion Heuristic w/ Convex Hull', 'Nearest Neighbour --> Two Opt', 'IHCV --> Two Opt', 'NN --> LNS', 'IHCV --> LNS', 'NN --> Two Opt --> LNS', 'IHCV --> Two Opt --> LNS'])

# print(compute_percentage_difference(3895.66, 3916))




# #==============================Plot Runtime vs CPU Time (Bubble Graph - bigger bubble = more nodes in the input dataset)==============================
# nodes = [7, 16, 52, 76, 107, 136, 225]
# algorithms_runtime = {
#     'Nearest Neighbour': {'data': [0.4112076759338379, 0.3927781581878662, 0.4872169017791748, 0.5912003755569458, 0.5175444841384887, 0.5670787572860718, 0.6898097515106201], 'color': 'blue'},
#     'Insertion Heuristic w/ Convex Hull': {'data': [0.31925201416015625, 0.4028491497039795, 0.5403737020492554, 0.8179960489273072, 0.6871454000473023, 0.6169691801071167, 1.103521466255188], 'color': 'red'},
#     # 'Integer Programming': {'data': [0.48547115325927737, 1.9439756274223328, 2.188145935535431, 238.9238269329071, 3600, 3600, 3600], 'color': 'green'},
#     'Nearest Neighbour --> Two Opt': {'data': [0.3369459629058838, 0.4849355220794678, 0.5839934539794921, 0.6298728704452514, 0.8510305881500244, 1.1399709383646648, 1.5792440176010132], 'color': 'lime'},
#     'IHCV --> Two Opt': {'data': [0.398201584815979, 0.4879587411880493, 0.5570610189437866, 0.6108887052536011, 0.7930902640024821, 0.8790188789367676, 1.6633020639419556], 'color': 'black'},
#     'NN --> LNS': {'data': [0.593600082397461, 0.7752989768981934, 2.462945890426636, 4.3152546882629395, 6.857412974039714, 20.448569536209106, 46.42879557609558], 'color': 'purple'},
#     'IHCV --> LNS': {'data': [0.6017665624618531, 0.7552976131439209, 2.0279680490493774, 5.947743972142537, 8.85807736714681, 15.488086581230164, 36.334240078926086], 'color': 'magenta'},
#     'NN --> Two Opt --> LNS': {'data': [0.6001407957077026, 0.8133480072021484, 2.0701633214950563, 4.365378888448079, 7.071200450261434, 19.903863668441772, 39.056431531906128], 'color': 'olive'},
#     'IHCV --> Two Opt --> LNS': {'data': [0.6019640445709229, 0.8137904930114746, 2.3267541885375977, 5.3695143063863116, 8.895966529846191, 18.60973048210144, 48.61195933818817], 'color': 'aqua'},
#     'Genetic Algorithm': {'data': [0.46117939949035647, 2.077482557296753, 11.650958061218262, 51.395593881607056, 94.05615496635437, 749.3404741287231, 2904.7282733239018], 'color': 'orange'},
# }

# algorithms_cpu_time = {
#     'Nearest Neighbour': {'data': [0.06434819999999997, 0.06856299999999979, 0.1028905, 0.13564180000000023, 0.17077940000000016, 0.2090824999999999, 0.32991720000000013], 'color': 'blue'},
#     'Insertion Heuristic w/ Convex Hull': {'data': [0.05864439999999984, 0.06983930000000002, 0.11625139999999998, 0.15631350000000008, 0.2842870999999999, 0.34389060000000005, 0.8435991999999999], 'color': 'red'},
#     # 'Integer Programming': {'data': [0.08146700000000004, 0.0933790000000001, 0.3090054999999998, 0.6113710000000001, 3600, 3600, 3600], 'color': 'green'},
#     'Nearest Neighbour --> Two Opt': {'data': [0.058959400000000176, 0.07274639999999995, 0.12024610000000022, 0.16514700000000007, 0.24584100000000006, 0.38065879999999996, 0.8629096], 'color': 'lime'},
#     'IHCV --> Two Opt': {'data': [0.05850529999999994, 0.06585739999999998, 0.1256982999999999, 0.15282080000000012, 0.2809012, 0.45419820000000016, 1.1295614999999999], 'color': 'black'},
#     'NN --> LNS': {'data': [0.25243659999999996, 0.42592280000000005, 2.3432852, 3.6238146, 6.4553855, 18.92006533333333, 36.527747], 'color': 'purple'},
#     'IHCV --> LNS': {'data': [0.24799320000000016, 0.41674609999999995, 1.7926807999999999, 4.7678232000000005, 7.682524599999998, 12.037884333333333, 34.378658], 'color': 'magenta'},
#     'NN --> Two Opt --> LNS': {'data': [0.2650253, 0.4447246999999999, 1.8235629999999996, 3.8795434, 6.4950489000000005, 11.2766025, 29.161218999999996], 'color': 'olive'},
#     'IHCV --> Two Opt --> LNS': {'data': [0.2553482999999999, 0.4334836999999997, 1.7669904000000003, 4.6750652, 8.743738600000002, 13.697148, 37.368492], 'color': 'aqua'},
#     'Genetic Algorithm': {'data': [0.16895220000000002, 1.8290260999999997, 6.796694400000002, 45.3103905, 74.549643, 721.638777333333333, 2332.163935], 'color': 'orange'},
# }

# def plot_performance_comparison(nodes, algorithms_runtime, algorithms_cpu_time, title):
#     fig = go.Figure()
#     for algorithm_name, algorithm_runtime_details in algorithms_runtime.items():
#         if algorithm_name in algorithms_cpu_time:  
#             algorithm_quality_details = algorithms_cpu_time[algorithm_name]
#             sizes = [10 + 2*np.log(node)**2 for node in nodes]  

#             runtime_values = algorithm_runtime_details['data']
#             quality_values = algorithm_quality_details['data']

#             hover_text = [ f"Algorithm Name: {algorithm_name}<br>No. Nodes: {node}<br>Algorithm Runtime: {np.round(runtime, 2)}s<br>Solution Quality: {quality}%" for (node, runtime, quality) in zip(nodes, runtime_values, quality_values) ]

#             fig.add_trace(go.Scatter(x=runtime_values, 
#                                      y=quality_values, 
#                                      mode='markers', 
#                                      marker=dict(size=sizes, 
#                                                  color=algorithm_quality_details['color'],
#                                                  opacity=0.65),
#                                     text=hover_text, hoverinfo="text",
#                                     name=algorithm_name))

#             fig.update_layout(
#                 title=title, 
#                 xaxis_title='Runtime (s)', 
#                 yaxis_title='CPU Time (s)',
#                 legend_title='Algorithm: '
#             )
    
#     fig.show()

# plot_performance_comparison(nodes, algorithms_runtime, algorithms_cpu_time, 'Holistic Algorithm Performance Comparison: Runtime vs CPU Time')
# #==============================Plot Runtime vs CPU Time (Bubble Graph - bigger bubble = more nodes in the input dataset)==============================
# nodes = [7, 16, 52, 76, 107, 136, 225]
# algorithms_quality = {
#     'Nearest Neighbour': {'data': [26.79, 41.5, 19.08, 41.89, 5.36, 24.81, 23.31], 'color': 'blue'},
#     'Insertion Heuristic w/ Convex Hull': {'data': [0, 1.5, 7.48, 6.15, 3.22, 6.12, 14.44], 'color': 'red'},
#     # 'Integer Programming': {'data': [0, 0, 0.03, 0, 23.26, 0.64, 8.49], 'color': 'green'},
#     'Nearest Neighbour --> Two Opt': {'data': [0, 2.67, 6.83, 5.06, 1.05, 8.62, 5.31], 'color': 'lime'},
#     'IHCV --> Two Opt': {'data': [0, 0, 7.06, 4.58, 2.78, 3.11, 10.55], 'color': 'black'},
#     'NN --> LNS': {'data': [0, 0, 4.90, 2.31, 0.31, 4.27, 6.39], 'color': 'purple'},
#     'IHCV --> LNS': {'data': [0, 0, 4.90, 0.82, 0.41, 1.64, 1.89], 'color': 'magenta'},
#     'NN --> Two Opt --> LNS': {'data': [0, 0, 4.90, 0.84, 0.31, 4.15, 0.79], 'color': 'olive'},
#     'IHCV --> Two Opt --> LNS': {'data': [0, 0, 4.58, 0.82, 0, 0.57, 2.15], 'color': 'aqua'},
#     'Genetic Algorithm': {'data': [0, 0, 0.03, 0, 0.31, 0.09, -0.52], 'color': 'orange'},
# }

# algorithms_cpu_time = {
#     'Nearest Neighbour': {'data': [0.06434819999999997, 0.06856299999999979, 0.1028905, 0.13564180000000023, 0.17077940000000016, 0.2090824999999999, 0.32991720000000013], 'color': 'blue'},
#     'Insertion Heuristic w/ Convex Hull': {'data': [0.05864439999999984, 0.06983930000000002, 0.11625139999999998, 0.15631350000000008, 0.2842870999999999, 0.34389060000000005, 0.8435991999999999], 'color': 'red'},
#     # 'Integer Programming': {'data': [0.08146700000000004, 0.0933790000000001, 0.3090054999999998, 0.6113710000000001, 3600, 3600, 3600], 'color': 'green'},
#     'Nearest Neighbour --> Two Opt': {'data': [0.058959400000000176, 0.07274639999999995, 0.12024610000000022, 0.16514700000000007, 0.24584100000000006, 0.38065879999999996, 0.8629096], 'color': 'lime'},
#     'IHCV --> Two Opt': {'data': [0.05850529999999994, 0.06585739999999998, 0.1256982999999999, 0.15282080000000012, 0.2809012, 0.45419820000000016, 1.1295614999999999], 'color': 'black'},
#     'NN --> LNS': {'data': [0.25243659999999996, 0.42592280000000005, 2.3432852, 3.6238146, 6.4553855, 18.92006533333333, 36.527747], 'color': 'purple'},
#     'IHCV --> LNS': {'data': [0.24799320000000016, 0.41674609999999995, 1.7926807999999999, 4.7678232000000005, 7.682524599999998, 12.037884333333333, 34.378658], 'color': 'magenta'},
#     'NN --> Two Opt --> LNS': {'data': [0.2650253, 0.4447246999999999, 1.8235629999999996, 3.8795434, 6.4950489000000005, 11.2766025, 29.161218999999996], 'color': 'olive'},
#     'IHCV --> Two Opt --> LNS': {'data': [0.2553482999999999, 0.4334836999999997, 1.7669904000000003, 4.6750652, 8.743738600000002, 13.697148, 37.368492], 'color': 'aqua'},
#     'Genetic Algorithm': {'data': [0.16895220000000002, 1.8290260999999997, 6.796694400000002, 45.3103905, 74.549643, 721.638777333333333, 2332.163935], 'color': 'orange'},
# }

# def plot_performance_comparison(nodes, algorithms_quality, algorithms_cpu_time, title):
#     fig = go.Figure()
#     for algorithm_name, algorithm_runtime_details in algorithms_quality.items():
#         if algorithm_name in algorithms_cpu_time:  
#             algorithm_quality_details = algorithms_cpu_time[algorithm_name]
#             sizes = [10 + 2*np.log(node)**2 for node in nodes]  

#             runtime_values = algorithm_runtime_details['data']
#             quality_values = algorithm_quality_details['data']

#             hover_text = [ f"Algorithm Name: {algorithm_name}<br>No. Nodes: {node}<br>Algorithm Runtime: {np.round(runtime, 2)}s<br>Solution Quality: {quality}%" for (node, runtime, quality) in zip(nodes, runtime_values, quality_values) ]

#             fig.add_trace(go.Scatter(x=runtime_values, 
#                                      y=quality_values, 
#                                      mode='markers', 
#                                      marker=dict(size=sizes, 
#                                                  color=algorithm_quality_details['color'],
#                                                  opacity=0.65),
#                                     text=hover_text, hoverinfo="text",
#                                     name=algorithm_name))

#             fig.update_layout(
#                 title=title, 
#                 xaxis_title='Solution Quality from Optimal (%)', 
#                 yaxis_title='CPU Time (s)',
#                 legend_title='Algorithm: '
#             )
    
#     fig.show()

# plot_performance_comparison(nodes, algorithms_quality, algorithms_cpu_time, 'Holistic Algorithm Performance Comparison: Solution Quality vs CPU Time')



# #==============================Plot Runtime vs Solution Quality Graph (Bubble Graph - bigger bubble = more nodes in the input dataset)==============================
# nodes = [7, 16, 52, 76, 107, 136, 225]
# algorithms_runtime = {
#     'Nearest Neighbour': {'data': [0.4112076759338379, 0.3927781581878662, 0.4872169017791748, 0.5912003755569458, 0.5175444841384887, 0.5670787572860718, 0.6898097515106201], 'color': 'blue'},
#     'Insertion Heuristic w/ Convex Hull': {'data': [0.31925201416015625, 0.4028491497039795, 0.5403737020492554, 0.8179960489273072, 0.6871454000473023, 0.6169691801071167, 1.103521466255188], 'color': 'red'},
#     # 'Integer Programming': {'data': [0.48547115325927737, 1.9439756274223328, 2.188145935535431, 238.9238269329071, 3600, 3600, 3600], 'color': 'green'},
#     'Nearest Neighbour --> Two Opt': {'data': [0.3369459629058838, 0.4849355220794678, 0.5839934539794921, 0.6298728704452514, 0.8510305881500244, 1.1399709383646648, 1.5792440176010132], 'color': 'lime'},
#     'IHCV --> Two Opt': {'data': [0.398201584815979, 0.4879587411880493, 0.5570610189437866, 0.6108887052536011, 0.7930902640024821, 0.8790188789367676, 1.6633020639419556], 'color': 'black'},
#     'NN --> LNS': {'data': [0.593600082397461, 0.7752989768981934, 2.462945890426636, 4.3152546882629395, 6.857412974039714, 20.448569536209106, 46.42879557609558], 'color': 'purple'},
#     'IHCV --> LNS': {'data': [0.6017665624618531, 0.7552976131439209, 2.0279680490493774, 5.947743972142537, 8.85807736714681, 15.488086581230164, 36.334240078926086], 'color': 'magenta'},
#     'NN --> Two Opt --> LNS': {'data': [0.6001407957077026, 0.8133480072021484, 2.0701633214950563, 4.365378888448079, 7.071200450261434, 19.903863668441772, 39.056431531906128], 'color': 'olive'},
#     'IHCV --> Two Opt --> LNS': {'data': [0.6019640445709229, 0.8137904930114746, 2.3267541885375977, 5.3695143063863116, 8.895966529846191, 18.60973048210144, 48.61195933818817], 'color': 'aqua'},
#     'Genetic Algorithm': {'data': [0.46117939949035647, 2.077482557296753, 11.650958061218262, 51.395593881607056, 94.05615496635437, 749.3404741287231, 2904.7282733239018], 'color': 'orange'},
# }

# algorithms_quality = {
#     'Nearest Neighbour': {'data': [26.79, 41.5, 19.08, 41.89, 5.36, 24.81, 23.31], 'color': 'blue'},
#     'Insertion Heuristic w/ Convex Hull': {'data': [0, 1.5, 7.48, 6.15, 3.22, 6.12, 14.44], 'color': 'red'},
#     # 'Integer Programming': {'data': [0, 0, 0.03, 0, 23.26, 0.64, 8.49], 'color': 'green'},
#     'Nearest Neighbour --> Two Opt': {'data': [0, 2.67, 6.83, 5.06, 1.05, 8.62, 5.31], 'color': 'lime'},
#     'IHCV --> Two Opt': {'data': [0, 0, 7.06, 4.58, 2.78, 3.11, 10.55], 'color': 'black'},
#     'NN --> LNS': {'data': [0, 0, 4.90, 2.31, 0.31, 4.27, 6.39], 'color': 'purple'},
#     'IHCV --> LNS': {'data': [0, 0, 4.90, 0.82, 0.41, 1.64, 1.89], 'color': 'magenta'},
#     'NN --> Two Opt --> LNS': {'data': [0, 0, 4.90, 0.84, 0.31, 4.15, 0.79], 'color': 'olive'},
#     'IHCV --> Two Opt --> LNS': {'data': [0, 0, 4.58, 0.82, 0, 0.57, 2.15], 'color': 'aqua'},
#     'Genetic Algorithm': {'data': [0, 0, 0.03, 0, 0.31, 0.09, -0.52], 'color': 'orange'},
# }

# def plot_performance_comparison(nodes, algorithms_runtime, algorithms_quality, title):
#     fig = go.Figure()
#     for algorithm_name, algorithm_runtime_details in algorithms_runtime.items():
#         if algorithm_name in algorithms_quality:  
#             algorithm_quality_details = algorithms_quality[algorithm_name]
#             sizes = [10 + 2*np.log(node)**2 for node in nodes]  

#             runtime_values = algorithm_runtime_details['data']
#             quality_values = algorithm_quality_details['data']

#             hover_text = [ f"Algorithm Name: {algorithm_name}<br>No. Nodes: {node}<br>Algorithm Runtime: {np.round(runtime, 2)}s<br>Solution Quality: {quality}%" for (node, runtime, quality) in zip(nodes, runtime_values, quality_values) ]

#             fig.add_trace(go.Scatter(x=runtime_values, 
#                                      y=quality_values, 
#                                      mode='markers', 
#                                      marker=dict(size=sizes, 
#                                                  color=algorithm_quality_details['color'],
#                                                  opacity=0.65),
#                                     text=hover_text, hoverinfo="text",
#                                     name=algorithm_name))

#             fig.update_layout(
#                 title=title, 
#                 xaxis_title='Runtime (s)', 
#                 yaxis_title='Solution Quality - Percentage Difference from Global Optimal (%)',
#                 legend_title='Algorithm: '
#             )
    
#     fig.show()

# plot_performance_comparison(nodes, algorithms_runtime, algorithms_quality, 'Holistic Algorithm Performance Comparison: Runtime vs Solution Quality - Excluding Integer Programming')