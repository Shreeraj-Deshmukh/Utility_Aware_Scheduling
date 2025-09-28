import gurobipy as gp
from gurobipy import GRB
import numpy as np
from math import lcm
from functools import reduce
from testcasev2 import testcase 

class UAS:
    def __init__(self, processors, tasks, B, alpha=0.1, beta=0.001):
        self.processors = processors
        self.tasks = tasks
        self.B = B
        self.alpha = alpha
        self.beta = beta
        
        self.N_prc = len(processors)
        self.N_tsk = len(tasks)
        self.p_is = [t['p_i'] for t in tasks]
        self.hyperp_i = reduce(lcm, self.p_is)
        
        self.N_jobs = [self.hyperp_i // t['p_i'] for t in tasks]
        self.total_jobs = sum(self.N_jobs)
        
        self.max_positions = self.total_jobs
        
        print(f"System Configuration:")
        print(f"  Processors: {self.N_prc}")
        print(f"  Tasks: {self.N_tsk}")
        print(f"  Hyperp_i: {self.hyperp_i}")
        print(f"  Total jobs: {self.total_jobs}")
        print(f"  Energy budget: {B}")
    
    def exe_time_calc(self, T_i, seg_k, P_x, f_z):
        task = self.tasks[T_i]
        proc = self.processors[P_x]
        freq = proc['frequencies'][f_z]
        
        exec_time = task['e_m'] / freq
        
        for seg in range(min(seg_k, len(task['e_o_k']))):
            exec_time += task['e_o_k'][seg] / freq
        
        return exec_time
    
    def E(self, T_i, seg_k, P_x, f_z):
        task = self.tasks[T_i]
        proc = self.processors[P_x]
        freq = proc['frequencies'][f_z]
        
        exec_time_max = task['e_m']
        for seg in range(min(seg_k, len(task['e_o_k']))):
            exec_time_max += task['e_o_k'][seg]
        
        exec_time_scaled = exec_time_max / freq
        energy = self.alpha * exec_time_scaled * (self.beta * freq**2 + exec_time_max)
        return energy
    
    def build_model(self):
        self.model = gp.Model("USRT_Scheduling")
        
        self.V_est_eft()
        
        self.C1_C2_C3_C4()
        
        self.goal_phi()
        
        return self.model
    
    def V_est_eft(self):
        self.V = {}
        
        for i in range(self.N_tsk):
            task = self.tasks[i]
            n_segments = len(task['e_o_k'])
            
            for j in range(self.N_jobs[i]):
                for k in range(n_segments + 1):
                    for x in range(self.N_prc):
                        proc = self.processors[x]
                        n_freqs = len(proc['frequencies'])
                        
                        for y in range(self.max_positions):
                            for z in range(n_freqs):
                                var_name = f"V_{i}_{j}_{k}_{x}_{y}_{z}"
                                self.V[i,j,k,x,y,z] = self.model.addVar(
                                    vtype=GRB.BINARY,
                                    name=var_name
                                )
        
        self.est = {}
        self.eft = {}
        
        for x in range(self.N_prc):
            for y in range(self.max_positions):
                self.est[x,y] = self.model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0,
                    name=f"est_x_y"
                )
                self.eft[x,y] = self.model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0,
                    name=f"eft_x_y"
                )
        
        self.model.update()
    
    def C1_C2_C3_C4(self):
        for i in range(self.N_tsk):
            for j in range(self.N_jobs[i]):
                self.model.addConstr(
                    gp.quicksum(
                        self.V[i,j,k,x,y,z]
                        for k in range(len(self.tasks[i]['e_o_k']) + 1)
                        for x in range(self.N_prc)
                        for y in range(self.max_positions)
                        for z in range(len(self.processors[x]['frequencies']))
                    ) == 1,
                    name=f"C1_T_i_j"
                )
        
        for x in range(self.N_prc):
            for y in range(self.max_positions):
                self.model.addConstr(
                    gp.quicksum(
                        self.V[i,j,k,x,y,z]
                        for i in range(self.N_tsk)
                        for j in range(self.N_jobs[i])
                        for k in range(len(self.tasks[i]['e_o_k']) + 1)
                        for z in range(len(self.processors[x]['frequencies']))
                    ) <= 1,
                    name=f"C2_proc{x}_pos{y}"
                )
        
        self.C3_timeconstr()
        
        self.C4_Econstr()
    
    def C3_timeconstr(self):
        for x in range(self.N_prc):
            for y in range(self.max_positions):
                release_time_expr = gp.quicksum(
                    self.V[i,j,k,x,y,z] * (j * self.tasks[i]['p_i'])
                    for i in range(self.N_tsk)
                    for j in range(self.N_jobs[i])
                    for k in range(len(self.tasks[i]['e_o_k']) + 1)
                    for z in range(len(self.processors[x]['frequencies']))
                )
                
                exec_time_expr = gp.quicksum(
                    self.V[i,j,k,x,y,z] * self.exe_time_calc(i, k, x, z)
                    for i in range(self.N_tsk)
                    for j in range(self.N_jobs[i])
                    for k in range(len(self.tasks[i]['e_o_k']) + 1)
                    for z in range(len(self.processors[x]['frequencies']))
                )
                
                deadline_expr = gp.quicksum(
                    self.V[i,j,k,x,y,z] * ((j + 1) * self.tasks[i]['p_i'])
                    for i in range(self.N_tsk)
                    for j in range(self.N_jobs[i])
                    for k in range(len(self.tasks[i]['e_o_k']) + 1)
                    for z in range(len(self.processors[x]['frequencies']))
                )
                
                if y == 0:
                    self.model.addConstr(
                        self.est[x,0] == release_time_expr,
                        name=f"est_proc{x}_pos{y}"
                    )
                else:
                    M = self.hyperp_i * 2
                    
                    delta = self.model.addVar(vtype=GRB.BINARY, name=f"delta_{x}_{y}")
                    
                    self.model.addConstr(
                        self.est[x,y] >= release_time_expr,
                        name=f"est_release_{x}_{y}"
                    )
                    self.model.addConstr(
                        self.est[x,y] >= self.eft[x,y-1],
                        name=f"est_prev_eft_{x}_{y}"
                    )
                    self.model.addConstr(
                        self.est[x,y] <= release_time_expr + M * delta,
                        name=f"est_upper1_{x}_{y}"
                    )
                    self.model.addConstr(
                        self.est[x,y] <= self.eft[x,y-1] + M * (1 - delta),
                        name=f"est_upper2_{x}_{y}"
                    )
                
                self.model.addConstr(
                    self.eft[x,y] == self.est[x,y] + exec_time_expr,
                    name=f"eft_proc{x}_pos{y}"
                )
                
                position_used = gp.quicksum(
                    self.V[i,j,k,x,y,z]
                    for i in range(self.N_tsk)
                    for j in range(self.N_jobs[i])
                    for k in range(len(self.tasks[i]['e_o_k']) + 1)
                    for z in range(len(self.processors[x]['frequencies']))
                )
                
                M_deadline = self.hyperp_i * 2
                self.model.addConstr(
                    self.eft[x,y] <= deadline_expr + M_deadline * (1 - position_used),
                    name=f"deadline_proc{x}_pos{y}"
                )
    
    def C4_Econstr(self):
        E_net = gp.quicksum(
            self.V[i,j,k,x,y,z] * self.E(i, k, x, z)
            for i in range(self.N_tsk)
            for j in range(self.N_jobs[i])
            for k in range(len(self.tasks[i]['e_o_k']) + 1)
            for x in range(self.N_prc)
            for y in range(self.max_positions)
            for z in range(len(self.processors[x]['frequencies']))
        )
        
        self.model.addConstr(
            E_net <= self.B,
            name="B"
        )
    
    def goal_phi(self):
        total_utility = 0
        
        for i in range(self.N_tsk):
            task = self.tasks[i]
            task_utility = 0
            
            for j in range(self.N_jobs[i]):
                for k in range(1, len(task['e_o_k']) + 1):
                    optional_exec_time = sum(task['e_o_k'][:k])
                    
                    job_utility = gp.quicksum(
                        self.V[i,j,k,x,y,z] * task['u_i'] * optional_exec_time
                        for x in range(self.N_prc)
                        for y in range(self.max_positions)
                        for z in range(len(self.processors[x]['frequencies']))
                    )
                    
                    task_utility += job_utility
            
            total_utility += task_utility
        
        avg_utility = total_utility / self.N_tsk
        
        self.model.setObjective(avg_utility, GRB.MAXIMIZE)
    
    def solve(self, time_limit=3600, mip_gap=0.5):
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', mip_gap)
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL:
            print("\nOptimal solution exists")
        elif self.model.status == GRB.TIME_LIMIT:
            print("\nTLE. Best solution uptil now:")
        elif self.model.status == GRB.INFEASIBLE:
            print("\nModel is infeasible. Possibly error testcases, period less than e_m, energy budget too low.")
            return None
        else:
            print(f"\nFinal end status {self.model.status}")
        
        return self.extract_solution()
    
    def extract_solution(self):
        if self.model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        
        solution = {
            'objective': self.model.objVal,
            'schedule': [],
            'energy_usage': 0
        }
        
        for i in range(self.N_tsk):
            for j in range(self.N_jobs[i]):
                for k in range(len(self.tasks[i]['e_o_k']) + 1):
                    for x in range(self.N_prc):
                        for y in range(self.max_positions):
                            for z in range(len(self.processors[x]['frequencies'])):
                                if self.V[i,j,k,x,y,z].X > 0.5:
                                    job_info = {
                                        'task': i,
                                        'job': j,
                                        'processor': x,
                                        'position': y,
                                        'frequency': self.processors[x]['frequencies'][z],
                                        'frequency_index': z,
                                        'e_o_k': k,
                                        'start_time': self.est[x,y].X if (x,y) in self.est else 0,
                                        'finish_time': self.eft[x,y].X if (x,y) in self.eft else 0,
                                        'energy': self.E(i, k, x, z)
                                    }
                                    solution['schedule'].append(job_info)
                                    solution['energy_usage'] += job_info['energy']
        
        solution['schedule'].sort(key=lambda x: (x['processor'], x['position']))
        
        return solution
    
    def print_solution(self, solution):
        if solution is None:
            print("\nNo solution available. Possible error testcases, period less than e_m, energy budget too low.")
            return
        
        print(f"\n{'=+'*30}")
        print(f"OPTIMIZATION RESULTS - OPTIMAL SOLUTION")
        print(f"{'-+'*28}")
        
        print("\n>>> Information:")
        print(f"  phi: {solution['objective']:.4f}")
        print(f"  E total: {solution['energy_usage']:.4f}")
        print(f"  B budget: {self.B:.4f}")
        print(f"  percent energy used from B: {solution['energy_usage']/self.B*100:.1f}%")
        print(f"  Unused energy B: {self.B - solution['energy_usage']:.4f}")
        
       
        print(f"\n{'+*='*30}")
        print(f"Mapping allocation")
        print(f"{'-*='*7}")
        
        for P_x in range(self.N_prc):
            proc_schedule = [s for s in solution['schedule'] if s['processor'] == P_x]
            
            if proc_schedule:
                print(f"\n>>> Processor P{P_x}:")
                print(f"{'Pos':<4} {'Task':<6} {'Job':<5} {'Freq':<6} {'OptSeg':<7} {'Start':<8} {'End':<8} {'Energy':<8} {'Deadline':<8}")
                print("-" * 60)
                
                for job in proc_schedule:
                    T_i = job['task']
                    job_id = job['job']
                    deadline = (job_id + 1) * self.tasks[T_i]['p_i']
                    
                    deadline_status = "Yes" if job['finish_time'] <= deadline else "✗"
                    
                    print(f"{job['position']:<4} "
                          f"T{job['task']:<5} "
                          f"{job['job']:<5} "
                          f"{job['frequency']:<6.2f} "
                          f"{job['e_o_k']:<7} "
                          f"{job['start_time']:<8.2f} "
                          f"{job['finish_time']:<8.2f} "
                          f"{job['energy']:<8.3f} "
                          f"{deadline:<7.0f}{deadline_status}")
        
        print("\n" + "+"*80)

def main():
    processors, tasks, B = testcase()
    
    scheduler = UAS(
        processors=processors,
        tasks=tasks,
        B=B,
        alpha=0.1,
        beta=0.001
    )
    
    print("\nBuilding optimization model...")
    model = scheduler.build_model()
    
    print("\nSolving optimization model...")
    solution = scheduler.solve(time_limit=300, mip_gap=0.5)
    
    scheduler.print_solution(solution)
    
    return scheduler, solution

if __name__ == "__main__":
    scheduler, solution = main()