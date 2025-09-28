import gurobipy as gp
from gurobipy import GRB

# Data
projects = ["A", "B", "C", "D", "E"]
c = [20, 15, 40, 30, 10]   # contribution of each project
a = [                     # resource usage matrix
    [3, 2, 7, 5, 1],      # resource 1 usage
    [4, 3, 8, 6, 2]       # resource 2 usage
]
b = [10, 12]  # resource capacities

# Model
m = gp.Model("CapitalBudgeting")

# Decision variables (0 or 1)
x = m.addVars(len(projects), vtype=GRB.BINARY, name="x")

# Objective: maximize contribution
m.setObjective(gp.quicksum(c[j]*x[j] for j in range(len(projects))), GRB.MAXIMIZE)

# Resource constraints
for i in range(len(b)):
    m.addConstr(gp.quicksum(a[i][j]*x[j] for j in range(len(projects))) <= b[i])

# Logical constraints
m.addConstr(x[2] >= x[0])   # If A is chosen, C must also be chosen
m.addConstr(x[1] + x[3] <= 1)  # At most one of B or D can be chosen

# Solve
m.optimize()

# Print solution
if m.status == GRB.OPTIMAL:
    print("\nOptimal Project Selection:")
    for j in range(len(projects)):
        if x[j].x > 0.5:
            print(f"Choose project {projects[j]}")
    print(f"Total contribution = {m.objVal}")
