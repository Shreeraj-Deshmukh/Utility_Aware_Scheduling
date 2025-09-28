"""
Simplified Real-Time Scheduler
Gurobi implementation for scheduling real-time tasks on heterogeneous processors
with energy and timing constraints to maximize total utility.
"""

import gurobipy as gp
from gurobipy import GRB
from math import lcm
from functools import reduce

class SimpleScheduler:
    def __init__(self, processors, tasks, energy_budget):
        """
        Initialize the Simplified Scheduler.
        
        Parameters:
        -----------
        processors : list
            A list of processor identifiers (e.g., [0, 1, 2])
        
        tasks : list of dict
            Each task dict contains:
            - 'id': task identifier
            - 'period': task period
            - 'exec_time': list of execution times on each processor
            - 'energy_cost': list of energy costs on each processor
            - 'utility': fixed utility gained upon job completion
        
        energy_budget : float
            Total energy budget B
        """
        self.processors = processors
        self.tasks = tasks
        self.energy_budget = energy_budget
        
        # ## Derived parameters
        self.N_prc = len(processors)
        self.N_tsk = len(tasks)
        self.periods = [t['period'] for t in tasks]
        self.hyperperiod = reduce(lcm, self.periods) if self.periods else 1
        
        self.N_jobs = [self.hyperperiod // t['period'] for t in tasks]
        self.total_jobs = sum(self.N_jobs)
        self.max_positions = self.total_jobs
        
        print(f"System Configuration:")
        print(f"  Processors: {self.N_prc}, Tasks: {self.N_tsk}")
        print(f"  Hyperperiod: {self.hyperperiod}, Total Jobs: {self.total_jobs}")

    def build_model(self):
        """Build the Gurobi optimization model."""
        self.model = gp.Model("SimpleScheduler")
        
        # ## 1. Create decision variables
        # V[i,j,x,y] = 1 if job j of task i is at position y on processor x
        self.V = self.model.addVars(
            [(i, j, x, y) 
             for i in range(self.N_tsk)
             for j in range(self.N_jobs[i])
             for x in range(self.N_prc)
             for y in range(self.max_positions)],
            vtype=GRB.BINARY, name="V"
        )
        
        # Auxiliary variables for timing
        self.est = self.model.addVars(self.N_prc, self.max_positions, name="est")
        self.eft = self.model.addVars(self.N_prc, self.max_positions, name="eft")

        # ## 2. Add constraints
        # C1: Each job is executed exactly once
        for i in range(self.N_tsk):
            for j in range(self.N_jobs[i]):
                self.model.addConstr(
                    self.V.sum(i, j, '*', '*') == 1, 
                    name=f"C1_job_{i}_{j}"
                )

        # C2: At most one job at any position on a processor
        for x in range(self.N_prc):
            for y in range(self.max_positions):
                self.model.addConstr(
                    self.V.sum('*', '*', x, y) <= 1, 
                    name=f"C2_pos_{x}_{y}"
                )

        # C3: Timing constraints
        for x in range(self.N_prc):
            for y in range(self.max_positions):
                release_time = gp.quicksum(self.V[i,j,x,y] * (j * self.tasks[i]['period']) 
                                           for i in range(self.N_tsk) for j in range(self.N_jobs[i]))
                exec_time = gp.quicksum(self.V[i,j,x,y] * self.tasks[i]['exec_time'][x] 
                                        for i in range(self.N_tsk) for j in range(self.N_jobs[i]))
                deadline = gp.quicksum(self.V[i,j,x,y] * ((j + 1) * self.tasks[i]['period']) 
                                       for i in range(self.N_tsk) for j in range(self.N_jobs[i]))

                # Link start, finish, and execution times
                self.model.addConstr(self.eft[x,y] == self.est[x,y] + exec_time, name=f"eft_{x}_{y}")
                
                # Start time logic
                if y == 0:
                    self.model.addConstr(self.est[x,y] >= release_time, name=f"est_{x}_{y}")
                else:
                    self.model.addConstr(self.est[x,y] >= self.eft[x,y-1], name=f"est_prec_{x}_{y}")
                    self.model.addConstr(self.est[x,y] >= release_time, name=f"est_release_{x}_{y}")

                # Deadline constraint (only if position is used)
                position_used = self.V.sum('*', '*', x, y)
                M = self.hyperperiod * 2
                self.model.addConstr(self.eft[x,y] <= deadline + M * (1 - position_used), name=f"deadline_{x}_{y}")

        # C4: Energy budget constraint
        total_energy = gp.quicksum(self.V[i,j,x,y] * self.tasks[i]['energy_cost'][x]
                                   for i,j,x,y in self.V)
        self.model.addConstr(total_energy <= self.energy_budget, name="energy_budget")

        # ## 3. Set objective: Maximize total utility
        total_utility = gp.quicksum(self.V[i,j,x,y] * self.tasks[i]['utility'] 
                                    for i,j,x,y in self.V)
        self.model.setObjective(total_utility, GRB.MAXIMIZE)

    def solve(self, time_limit=60):
        """Solve the model and print the solution."""
        self.model.setParam('TimeLimit', time_limit)
        self.model.optimize()

        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            print("\n" + "="*80)
            print("OPTIMIZATION RESULTS")
            print("="*80)
            print(f">>> Objective (Total Utility): {self.model.objVal:.2f}\n")

            schedule = []
            for (i,j,x,y), var in self.V.items():
                if var.x > 0.5:
                    schedule.append({
                        'task': i, 'job': j, 'proc': x, 'pos': y,
                        'start': self.est[x,y].x, 'finish': self.eft[x,y].x
                    })
            
            schedule.sort(key=lambda item: (item['proc'], item['start']))
            
            for proc_id in range(self.N_prc):
                print(f"\n--- Processor P{proc_id} Schedule ---")
                print(f"{'Pos':<5} {'Task':<6} {'Job':<5} {'Start':<8} {'End':<8} {'Deadline':<8}")
                print("-" * 50)
                proc_schedule = [s for s in schedule if s['proc'] == proc_id]
                for job in proc_schedule:
                    deadline = (job['job'] + 1) * self.tasks[job['task']]['period']
                    print(f"{job['pos']:<5} T{job['task']:<5} {job['job']:<5} {job['start']:<8.2f} {job['finish']:<8.2f} {deadline:<8.0f}")

        elif self.model.status == GRB.INFEASIBLE:
            print("\nModel is infeasible. Check constraints and data.")
        else:
            print(f"\nOptimization ended with status code: {self.model.status}")

# ## Example Usage
def main():
    processors = [0, 1]  # Two processors
    
    tasks = [
        {
            'id': 0, 'period': 20, 'utility': 10,
            'exec_time': [5, 6],    # [on P0, on P1]
            'energy_cost': [8, 10]  # [on P0, on P1]
        },
        {
            'id': 1, 'period': 40, 'utility': 15,
            'exec_time': [8, 7],
            'energy_cost': [12, 11]
        }
    ]
    
    energy_budget = 50

    # Initialize and solve
    scheduler = SimpleScheduler(processors, tasks, energy_budget)
    scheduler.build_model()
    scheduler.solve()

if __name__ == "__main__":
    main()