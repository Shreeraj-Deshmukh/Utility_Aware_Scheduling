import gurobipy as gp
from gurobipy import GRB

# Data
weights = [1,2,3]
values = [2,3,5]
max_qty = [10,10,1]   # maximum units available
capacity = 31
n = len(weights)

# Model
m = gp.Model("bounded_knapsack")

# Variables: integer since multiple quantities allowed
x = m.addVars(n, vtype=GRB.INTEGER, name="x")

# Objective
m.setObjective(gp.quicksum(values[i]*x[i] for i in range(n)), GRB.MAXIMIZE)

# Capacity constraint
m.addConstr(gp.quicksum(weights[i]*x[i] for i in range(n)) <= capacity)

# Quantity limits
for i in range(n):
    m.addConstr(x[i] <= max_qty[i])

# Solve
m.optimize()

# Output solution
print("\nOptimal solution:")
for i in range(n):
    if x[i].x > 0:
        print(f"Take {int(x[i].x)} units of item {i+1}")
print("Total value:", m.objVal)
