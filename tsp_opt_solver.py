# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys

sys.path.append(".")
import numpy as np
import math

import tsplib95
import lkh


def tsp2concorde(sample):
    pos_list = []
    node_list = ["output", "1"]
    pos_list.append(str(sample["depot"][0]))
    pos_list.append(str(sample["depot"][1]))
    for i, cus in enumerate(sample["customers"]):
        pos_list.append(str(cus["position"][0]))
        pos_list.append(str(cus["position"][1]))
        node_list.append(str(i + 2))
    node_list.append("1")
    return " ".join(pos_list + node_list)


def concorde2tsp(concorde, num_node):
    sample = {}
    derivate = concorde.split(" ")
    buff = np.zeros(shape=(num_node, 2), dtype=np.float64)
    buff[:, 0] = np.array(derivate[0 : 2 * num_node : 2], dtype=np.float64)
    buff[:, 1] = np.array(derivate[1 : 2 * num_node : 2], dtype=np.float64)
    sample["depot"] = (buff[0, 0], buff[0, 1])
    sample["capacity"] = num_node * 2
    sample["customers"] = []
    for i in range(1, num_node):
        sample["customers"].append({"position": (buff[i, 0], buff[i, 1]), "demand": 1})
    return sample


def tsplib2sample(problem):
    if problem.edge_weight_type != "EUC_2D":
        raise Exception("Weight type not implemented", problem.edge_weight_type)

    dimension = problem.dimension
    sample = {}
    pos_x_list, pos_y_list = [], []
    for n in range(1, dimension + 1):
        pos_x_list.append(problem.node_coords[n][0])
        pos_y_list.append(problem.node_coords[n][1])
    max_x = np.max(pos_x_list)
    max_y = np.max(pos_y_list)
    min_x = np.min(pos_x_list)
    min_y = np.min(pos_y_list)
    pos_x_list = [(x - min_x) / (max_x - min_x) for x in pos_x_list]
    pos_y_list = [(y - min_y) / (max_y - min_y) for y in pos_y_list]
    sample["depot"] = (pos_x_list[0], pos_y_list[0])
    sample["capacity"] = dimension * 2
    sample["customers"] = []
    for n in range(1, len(pos_x_list)):
        sample["customers"].append(
            {"position": (pos_x_list[n], pos_y_list[n]), "demand": 1}
        )
    sample["max_pos"] = (max_x, max_y)
    sample["min_pos"] = (min_x, min_y)
    return sample


def node_distance(left, right):
    return math.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)


def sample2tsplib_explict(sample):
    dimension = len(sample["customers"]) + 1
    problem_str = f"""NAME : tsp
COMMENT : tsp_comm
TYPE : ATSP
DIMENSION : {dimension}
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX 
EDGE_WEIGHT_SECTION"""
    node_pos = [sample["depot"]]
    for customer in sample["customers"]:
        node_pos.append(customer["position"])
    for i in range(len(node_pos)):
        dist_list = []
        for j in range(len(node_pos)):
            if i == j or (i == len(node_pos) - 1 and j > 0):
                dist_list.append("9999")
            else:
                dist_list.append(str(node_distance(node_pos[i], node_pos[j]) * 1000))
        dist_str = "    ".join(dist_list)
        problem_str += f"\n {dist_str}"
    problem_str += "\n EOF\n"
    return problem_str


def sample2tsplib(sample):
    dimension = len(sample["customers"]) + 1
    problem_str = f"""NAME : tsp
COMMENT : tsp_comm
TYPE : TSP
DIMENSION : {dimension}
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION"""
    problem_str += f"\n1 {sample['depot'][0]*1000} {sample['depot'][1]*1000}"
    for i, customer in enumerate(sample["customers"]):
        problem_str += (
            f"\n{i+2} {customer['position'][0]*1000} {customer['position'][1]*1000}"
        )
    problem_str += "\nEOF\n"
    return problem_str


def lkh_solver(sample, atsp=False, runs=-1):
    if not atsp:
        problem = tsplib95.parse(sample2tsplib(sample))
    else:
        problem = tsplib95.parse(sample2tsplib_explict(sample))

    if runs == -1:
        route = lkh.solve(
            "/usr/local/bin/LKH", problem=problem, max_trials=problem.dimension, runs=1
        )  # , max_trials=problem.dimension, runs=10)
    else:
        route = lkh.solve(
            "/usr/local/bin/LKH",
            problem=problem,
            max_trials=problem.dimension,
            runs=runs,
        )
    route = [r - 1 for r in route[0]]
    customer_pos_list = []
    if "max_pos" in sample:
        depot_pos = (
            sample["depot"][0] * (sample["max_pos"][0] - sample["min_pos"][0])
            + sample["min_pos"][0],
            sample["depot"][1] * (sample["max_pos"][1] - sample["min_pos"][1])
            + sample["min_pos"][1],
        )
        for cus in sample["customers"]:
            cus_pos = (
                cus["position"][0] * (sample["max_pos"][0] - sample["min_pos"][0])
                + sample["min_pos"][0],
                cus["position"][1] * (sample["max_pos"][1] - sample["min_pos"][1])
                + sample["min_pos"][1],
            )
            customer_pos_list.append(cus_pos)
    else:
        depot_pos = sample["depot"]
        for cus in sample["customers"]:
            cus_pos = cus["position"]
            customer_pos_list.append(cus_pos)

    route.append(0)
    if not atsp:
        dist = node_distance(depot_pos, customer_pos_list[route[1] - 1])
        for i in range(1, len(route) - 2):
            dist += node_distance(
                customer_pos_list[route[i] - 1], customer_pos_list[route[i + 1] - 1]
            )
        dist += node_distance(customer_pos_list[route[-2] - 1], depot_pos)
    else:
        dist = node_distance(depot_pos, customer_pos_list[route[1] - 1])
        for i in range(1, len(route) - 2):
            dist += node_distance(
                customer_pos_list[route[i] - 1], customer_pos_list[route[i + 1] - 1]
            )
    return route, dist
