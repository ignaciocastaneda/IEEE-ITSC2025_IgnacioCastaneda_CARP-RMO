# -*- coding: utf-8 -*-
"""
Created on Thu May  1 23:53:26 2025

@author: p04497
"""

import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import time
from matplotlib.lines import Line2D
import pandas as pd


### IMPORT DATA

excel_path = 'routes_c.xlsx'
nodes_df = pd.read_excel(excel_path, sheet_name='nodes')
arcs_df = pd.read_excel(excel_path, sheet_name='arcs')

x_nodes = nodes_df['x']
y_nodes = nodes_df['y']
id_nodes = nodes_df['id']

start_n = arcs_df['start_n']
end_n = arcs_df['end_n']
len_km = arcs_df['len_km']


## GRAPH GENERATION

G = nx.DiGraph()

##### Graph position

pos = {id_node: (x, y) for id_node, x, y in zip(id_nodes, x_nodes, y_nodes)}

G.add_nodes_from(id_nodes)


##### EDGES CREATION

edges = [(x, y) for x, y in zip(start_n, end_n)]

edges2 = edges + [(b, a) for (a, b) in edges]
arcs = sorted(edges2, key=lambda x: (x[0], x[1]))

G.add_edges_from(arcs)

###### DISTANCES

distance1 = [(x, y, d) for x, y, d in zip(start_n, end_n, len_km)]
distance2 = distance1 + [(y, x, d) for x, y, d in distance1]
distance3 = sorted(distance2, key=lambda x: (x[0], x[1]))

distances = {(x, y): d for x, y, d in distance3}


#types of roadsides

roadsides = [0,1,2]

roadside_list = [((i, j),r) for i, j in arcs for r in roadsides]


#### DEMAND Q

demand = [
            ((12, 9), 0), ((9, 11), 0), ((11, 9), 0), ((9, 10), 0), ((10, 9), 0), ((6, 5), 0), ((5, 3), 0), ((3, 4), 0), ((5, 13), 0), ((9, 6), 0),
            ((6, 9), 0), ((6, 7), 0), ((7, 6), 0), ((6, 8), 0), ((8, 6), 0), ((5, 6), 0), ((4, 3), 0), ((3, 5), 0), ((13, 5), 0), ((9, 12), 0), 
            ((13, 14), 0), ((14, 13), 0), ((14, 15), 0), ((15, 14), 0), ((15, 18), 0), ((18, 15), 0), ((18, 19), 0), ((19, 18), 0), ((15, 19), 0),
            ((19, 15), 0), ((19, 24), 0), ((24, 19), 0), ((19, 20), 0), ((20, 19), 0),
            
            ((28, 29), 0), ((29, 31), 0), ((31, 30), 0), ((30, 25), 0), ((25, 30), 0), ((31, 23), 0), ((23, 31), 0), ((60, 57), 0), ((57, 56), 0),
            ((56, 57), 0), ((29, 28), 0), ((28, 26), 0), ((26, 28), 0), ((25, 24), 0), ((24, 25), 0), ((64, 62), 0), ((62, 64), 0), ((64, 65), 0), 
            ((65, 64), 0), ((69, 71), 0), ((71, 69), 0), ((69, 70), 0), ((70, 69), 0), ((66, 67), 0), ((67, 66), 0), ((66, 65), 0), ((65, 66), 0),
            ((28, 32), 0), ((32, 28), 0), ((32, 56), 0), ((56, 32), 0), ((32, 55), 0), ((55, 32), 0),
    
            ((43, 42), 0), ((42, 41), 0), ((41, 40), 0), ((40, 41), 0), ((41, 43), 0), ((43, 41), 0), ((41, 42), 0), ((42, 43), 0), ((51, 53), 0),
            ((53, 54), 0), ((54, 53), 0), ((43, 44), 0), ((44, 43), 0), ((44, 45), 0), ((45, 44), 0), ((55, 56), 0), ((56, 55), 0), ((48, 50), 0),
            ((50, 48), 0), ((53, 51), 0), ((65, 64), 0), ((34, 55), 0), ((55, 34), 0),
    
            ((14, 17), 1), ((16, 14), 1),
            ((20, 21), 1), ((21, 22), 1), ((22, 21), 1), ((21, 20), 1),
            ((36, 37), 1), ((35, 33), 1), ((33, 35), 1),
            ((62, 63), 1), ((63, 62), 1),
    
            ((1, 2), 2), ((2, 1), 2), ((1, 3), 2), 
            ((48, 49), 2), ((49, 48), 2), ((33, 34), 2),
            ((59, 61), 2), ((61, 59), 2), ((59, 60), 2)
        ]


# Demand

demands = {}

#Assuming a regular roadside width of 1mt and 0.5kg of biomass per m2
width_r0 = 1/4
width_r1 = 3/8
width_r2 = 3/8
ton_km = 0.8 


for (i, j), r in demand:
        if (i, j) in distances:
            if r == 0:
                biomass = distances[(i,j)] * width_r0 * ton_km
                demands[(i, j), r] = round(biomass,2)
            if r == 1:
                biomass = distances[(i,j)] * width_r1 * ton_km
                demands[(i, j), r] = round(biomass,2)
            if r == 2:
                biomass = distances[(i,j)] * width_r2 * ton_km
                demands[(i, j), r] = round(biomass,2)
                
                

#TIMES

travel_speed = 40 #(Km/h)
mowing_1_speed = 6 #(Km/h)
mowing_2_3_speed = 3 #(Km/h)

travel_times = {arc: round(distances[arc] / travel_speed ,2) for arc in arcs}

serving_times = {(arc, r): round(-distances[arc] / travel_speed + 
                                 distances[arc] / (mowing_1_speed if r == 0 else mowing_2_3_speed), 2) for arc, r in demands
                }





################# OPTIMIZATION MODEL #####################

#########  IMPORTING GUROBIPY  ##############

import gurobipy
from gurobipy import Model, GRB, quicksum


######## SET AND PARAMETERS ##########

## SETS
# Graph
N = list(G.nodes())  # Set of nodes
A = list(G.edges())  # Set of arcs
A_r = list(demands)  # Subset of required arcs
D = [N[72], N[73], N[74]] #Depots
N_minus_D = [elem for i, elem in enumerate(N) if i not in [72, 73, 74]] #Nodes without depots

#Type of roadside
R = roadsides #type of roadsides

#Periods
num_periods = 6
P = list(range(num_periods)) # Set of periods

#Set of Vehicles
num_vehicles = 6
V = list(range(num_vehicles))  # Set of vehicles
V_1 = V[:len(V) // 2] #Subset of vehicles type 1: can mow only r = 0
V_2 = V[len(V) // 2:] #Subset of vehicles type 2: can mow r = 1 and r = 2


## PARAMETRES
c = distances #Distance
q = demands #Demand
W = {w: 200 for w in V} # Capacity of vehicle type 
t_max = 10 #(hrs) Total working hours per day
t_travel = travel_times # Time of travelling from i to j
t_serve = serving_times # Time of serving from i to j


##################  MODEL ############################
model = Model('CARP_RMO')


####### VARIABLES #############

x = model.addVars(A, V, P, vtype=GRB.INTEGER, name="x") #Visit arc (i,j)
l = model.addVars(A_r, V, P, vtype=GRB.BINARY, name="l") #Serve arc (i,j)
f = model.addVars(A, V, P, vtype=GRB.CONTINUOUS, name="f") #Flow variable: it's positive if x==1
a = model.addVars(V, D, vtype=GRB.BINARY, name="a") #If vehicle p is allocated to depot d. 


####### OBJECTIVE FUNCTION ############

model.setObjective(quicksum(c[(i, j)] * x[i, j, v, p] for i, j in A for v in V for p in P), GRB.MINIMIZE)


# Constraints

startCons = time.time()

# Ensures route continuity
for i in N:
    for p in P:
        for v in V:
            model.addConstr(quicksum(x[i, k, v, p] for k in N if (i, k) in A) - 
                            quicksum(x[k, i, v, p] for k in N if (k, i) in A) == 0, name = f"flow_{i}_{p}_{v}")


# Each arc with demand is served just once
for (i, j), r in A_r:
    model.addConstr(quicksum(l[(i, j), r, v, p] for v in V for p in P) == 1 , name = f"service_{i}_{j}")
                    #(q[(i, j), r] + W - 1) // W, name = f"service_{i}_{j}")


#Start from the depot
for p in P:
    for d in D:
        for v in V:
            model.addConstr(quicksum(x[d, j, v, p] for j in N if (d, j) in A) <= a[v, d], name=f"start_from_depot_{p}_{d}")
            


#model.addConstr(quicksum(a[v, d] for v in V for d in D) <= num_vehicles, name=f"depot_{d}")


#A vehicle is assigned to only one depot
for v in V:
    model.addConstr(quicksum(a[v, d] for d in D) <= 1, name=f"depot_{d}")


#Allocate only vehicles that are going to be used
for v in V: 
    model.addConstr(quicksum(l[(i, j), r, v, p] for (i, j), r in A_r for p in P) >= quicksum(a[v, d] for d in D), name=f"depot2_{d}")


# Vehicle type 1 must serve roadside type 1 but cannot serve types 2 and 3
for (i, j), r in A_r:
    if r == 1 or r ==2:
        model.addConstr(quicksum(l[(i, j), r, v, p] for v in V_1 for p in P) == 0 , name = f"service_{i}_{j}")
    

# If the vehicle doesn't pass to the arc, the arc can't be served
for (i, j), r in A_r:
    for p in P:
        for v in V:
                model.addConstr(x[i, j, v, p] >= l[(i, j), r, v, p], name = f"traverse_{i}_{j}_{p}_{v}")


# Capacity constraints for vehicles
for p in P:
    for v in V:
        model.addConstr(quicksum(l[(i, j), r, v, p] * q[(i, j), r] for (i, j), r in A_r) <= W[v], name = f"capacity_{v}_{p}")


# Time constraints
for p in P:
    for v in V:
        model.addConstr(quicksum(t_travel[(i, j)] * x[i, j, v, p] for i, j in A) +
                        quicksum(t_serve[(i, j), r] * l[(i, j), r, v, p] for (i, j), r in A_r) <= t_max, 
                        name = f"time_{v}_{p}")


# Subtour elimination
for i in N_minus_D: #N - depot
    for p in P:
        for v in V:
            model.addConstr(quicksum(f[i, k, v, p] for k in N if (i, k) in A) -
                            quicksum(f[k, i, v, p] for k in N if (k, i) in A) == 
                            quicksum(x[i, j, v, p] for j in N if (i, j) in A), name = f"subtour_e_{i}_{v}_{p}")

for i, j in A:
    for p in P:
        for v in V:
            model.addConstr(f[i, j, v, p] <= len(N) * x[i, j, v, p], name = f"subtour_ub_{i}_{j}_{v}_{p}")
            model.addConstr(f[i, j, v, p] >= 0, f"subtour_lb_{i}_{j}_{p}")


###### OPTIMIZE ######

#STOP CONDITIONS
#model.Params.MIPGap = 0.1
#mdl.Params.TImeLimit = 72000

model.optimize()