import gurobipy as gp
from gurobipy import GRB

# Example: 4 cities with distances
n = 4
dist = [[0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]]

# Create model
m = gp.Model("TSP")

# Decision variables: x[i,j] = 1 if we travel from i to j
x = m.addVars(n, n, vtype=GRB.BINARY, name="x")

# Objective: minimize total travel cost
m.setObjective(gp.quicksum(dist[i][j]*x[i,j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

# Constraints:
# 1. No self-loops
for i in range(n):
    m.addConstr(x[i,i] == 0)

# 2. Leave each city exactly once
for i in range(n):
    m.addConstr(gp.quicksum(x[i,j] for j in range(n)) == 1)

# 3. Enter each city exactly once
for j in range(n):
    m.addConstr(gp.quicksum(x[i,j] for i in range(n)) == 1)

# --- Subtour elimination constraints (Miller–Tucker–Zemlin formulation) ---
u = m.addVars(n, vtype=GRB.INTEGER, name="u")
for i in range(1, n):
    for j in range(1, n):
        if i != j:
            m.addConstr(u[i] - u[j] + n*x[i,j] <= n-1)

# Solve
m.optimize()

# Reconstruct ordered tour from x[i,j]
succ = {i: j for i in range(n) for j in range(n) if x[i, j].X > 0.5}
start = 0
tour = [start]
for _ in range(n - 1):
    tour.append(succ[tour[-1]])

# Print nice tour and total distance
print("Ordered tour:", " -> ".join(map(str, tour + [start])))
total = sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))
print("Total distance:", total)
