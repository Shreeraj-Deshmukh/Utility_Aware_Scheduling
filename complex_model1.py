from gurobipy import Model, GRB, quicksum


import numpy as np
from math import gcd
from functools import reduce

# Helper function to compute LCM of a list of numbers
def lcm(a, b):
    return a * b // gcd(a, b)

def calculate_lcm(numbers):
    return reduce(lcm, numbers)

# Sample data (you should replace these with your actual values)
NT = 3  # Number of tasks
NP = 2  # Number of processors

# Task data: (period, mandatory time, optional segment times, utility, max optional segments)
task_data = [
    (10, [5, 2], [2, 1], 10, 1),  # Task 1: Period 10, mandatory time 5, optional times [2,1], utility 10, max optional segments 1
    (15, [6, 3], [3, 2], 8, 1),   # Task 2: Period 15, mandatory time 6, optional times [3,2], utility 8, max optional segments 1
    (20, [7, 4], [4, 3], 12, 2)   # Task 3: Period 20, mandatory time 7, optional times [4,3], utility 12, max optional segments 2
]

# Processor data: (voltage, frequency) pairs for each processor
processor_data = [
    [(1.0, 2.0), (1.2, 2.5)],  # Processor 1: voltage-frequency pairs
    [(1.1, 2.2), (1.3, 2.6)]   # Processor 2: voltage-frequency pairs
]

E_max = 1000  # Maximum energy budget

# Gurobi Model setup
m = Model('Task_Scheduling')

# Decision Variables
x = {}
y = {}
t = {}
for i in range(NT):
    for j in range(NP):
        x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
        for t_idx in range(len(task_data[i][1]) + 1):  # mandatory + optional segments
            y[i, t_idx, j] = m.addVar(vtype=GRB.BINARY, name=f'y_{i}_{t_idx}_{j}')
            t[i, t_idx, j] = m.addVar(vtype=GRB.CONTINUOUS, name=f't_{i}_{t_idx}_{j}')

# Objective: Maximize utility
m.setObjective(
    quicksum(task_data[i][3] * task_data[i][1][t_idx] * y[i, t_idx, j] 
                 for i in range(NT) for j in range(NP) for t_idx in range(len(task_data[i][1]) + 1)),
    GRB.GRB.MAXIMIZE
)

# Constraints

# 1. Each task should be assigned to exactly one processor
for i in range(NT):
    m.addConstr(quicksum(x[i, j] for j in range(NP)) == 1)

# 2. Segment execution constraint
for i in range(NT):
    for j in range(NP):
        # The task must execute the mandatory segment
        m.addConstr(y[i, 0, j] == 1)  # Mandatory segment must be executed

        # Task can execute up to k_i optional segments (0 to k_i)
        m.addConstr(quicksum(y[i, t_idx, j] for t_idx in range(1, task_data[i][4] + 1)) <= task_data[i][4])

        # Task must execute at least one segment (mandatory or optional)
        m.addConstr(quicksum(y[i, t_idx, j] for t_idx in range(0, task_data[i][4] + 1)) >= 1)

# 3. Energy Consumption Constraint
total_energy = quicksum(
    (processor_data[j][0][0] ** 2 * processor_data[j][0][1] * t[i, t_idx, j])
    for i in range(NT) for j in range(NP) for t_idx in range(len(task_data[i][1]) + 1)
)
m.addConstr(total_energy <= E_max)

# 4. Time Calculation (adjusted for the max number of optional segments)
for i in range(NT):
    for t_idx in range(len(task_data[i][1]) + 1):
        for j in range(NP):
            m.addConstr(t[i, t_idx, j] == task_data[i][1][t_idx] * (processor_data[j][0][1] / processor_data[0][0][1]))

# Solve the optimization problem
m.optimize()

# Print the solution
if m.status == GRB.GRB.OPTIMAL:
    print("\nOptimal Solution:")
    for i in range(NT):
        for j in range(NP):
            if x[i, j].x > 0.5:
                print(f"Task {i} assigned to Processor {j}")
        for t_idx in range(len(task_data[i][1]) + 1):
            if y[i, t_idx, j].x > 0.5:
                print(f"Task {i} uses segment {t_idx} on Processor {j}")
else:
    print("No optimal solution found.")
