"""
Ant Colony Optimization para Job-Shop Scheduling Problem
Autor: Andres Felipe Morales Mejia
Código: 1004754257
Profesor: Angel Augusto Agudelo Z
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class JobShopACO:
    def __init__(self, jobs_data: List[List[Tuple[int, int]]], 
                 n_ants: int = 20, 
                 n_iterations: int = 100,
                 alpha: float = 1.0, 
                 beta: float = 2.0, 
                 rho: float = 0.5,
                 Q: float = 100.0):
        """
        Inicializa ACO para JSP
        
        jobs_data: Lista de trabajos, cada trabajo es lista de (máquina, tiempo)
        Ejemplo: [[(0, 3), (1, 2), (2, 2)], [(1, 2), (2, 1), (0, 4)]]
        """
        self.jobs_data = jobs_data
        self.n_jobs = len(jobs_data)
        self.n_machines = max(max(op[0] for op in job) for job in jobs_data) + 1
        self.n_operations = sum(len(job) for job in jobs_data)
        
        # Parámetros ACO
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # Inicializar matriz de feromonas
        self.pheromone = np.ones((self.n_operations, self.n_operations)) * 0.1
        
        # Calcular heurística (inverso del tiempo)
        self.heuristic = self._calculate_heuristic()
        
        # Mejores resultados
        self.best_makespan = float('inf')
        self.best_schedule = None
        self.history = []
    
    def _calculate_heuristic(self) -> np.ndarray:
        """Calcula matriz heurística basada en tiempos de operación"""
        heuristic = np.zeros((self.n_operations, self.n_operations))
        operations = []
        
        for job_idx, job in enumerate(self.jobs_data):
            for op_idx, (machine, time) in enumerate(job):
                operations.append((job_idx, op_idx, machine, time))
        
        for i in range(self.n_operations):
            for j in range(self.n_operations):
                if i != j:
                    time = operations[j][3]
                    heuristic[i][j] = 1.0 / (time + 1e-10)
        
        return heuristic
    
    def _construct_solution(self) -> List[int]:
        """Una hormiga construye una solución (secuencia de operaciones)"""
        # Seguimiento de operaciones completadas por trabajo
        job_operation_index = [0] * self.n_jobs
        solution = []
        available_operations = []
        
        # Inicializar operaciones disponibles (primera de cada trabajo)
        for job_idx in range(self.n_jobs):
            available_operations.append(job_idx)
        
        while len(solution) < self.n_operations:
            # Calcular operación global actual
            current_op = len(solution) if len(solution) > 0 else 0
            
            # Calcular probabilidades para operaciones disponibles
            probabilities = []
            operation_indices = []
            
            for job_idx in available_operations:
                op_idx = job_operation_index[job_idx]
                # Índice global de operación
                global_op_idx = sum(len(self.jobs_data[j]) for j in range(job_idx)) + op_idx
                operation_indices.append(global_op_idx)
                
                if current_op == 0:
                    prob = 1.0 / len(available_operations)
                else:
                    tau = self.pheromone[current_op - 1][global_op_idx]
                    eta = self.heuristic[current_op - 1][global_op_idx]
                    prob = (tau ** self.alpha) * (eta ** self.beta)
                
                probabilities.append(prob)
            
            # Normalizar probabilidades
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
            else:
                probabilities = [1.0 / len(probabilities)] * len(probabilities)
            
            # Seleccionar operación
            selected_idx = np.random.choice(len(available_operations), p=probabilities)
            selected_job = available_operations[selected_idx]
            
            solution.append(selected_job)
            job_operation_index[selected_job] += 1
            
            # Actualizar operaciones disponibles
            if job_operation_index[selected_job] >= len(self.jobs_data[selected_job]):
                available_operations.remove(selected_job)
        
        return solution
    
    def _calculate_makespan(self, solution: List[int]) -> Tuple[int, Dict]:
        """Calcula el makespan de una solución"""
        job_operation_index = [0] * self.n_jobs
        job_end_time = [0] * self.n_jobs
        machine_end_time = [0] * self.n_machines
        schedule = {m: [] for m in range(self.n_machines)}
        
        for job_idx in solution:
            op_idx = job_operation_index[job_idx]
            machine, duration = self.jobs_data[job_idx][op_idx]
            
            # Tiempo de inicio: máximo entre fin del trabajo y fin de la máquina
            start_time = max(job_end_time[job_idx], machine_end_time[machine])
            end_time = start_time + duration
            
            # Actualizar tiempos
            job_end_time[job_idx] = end_time
            machine_end_time[machine] = end_time
            
            # Guardar en cronograma
            schedule[machine].append((job_idx, op_idx, start_time, end_time))
            
            job_operation_index[job_idx] += 1
        
        makespan = max(machine_end_time)
        return makespan, schedule
    
    def _update_pheromones(self, solutions: List[List[int]], makespans: List[int]):
        """Actualiza matriz de feromonas"""
        # Evaporación
        self.pheromone *= (1 - self.rho)
        
        # Depósito de feromonas
        for solution, makespan in zip(solutions, makespans):
            delta = self.Q / makespan
            
            for i in range(len(solution) - 1):
                job_from = solution[i]
                job_to = solution[i + 1]
                
                # Calcular índices globales
                op_from = i
                op_to = i + 1
                
                if op_from < self.n_operations and op_to < self.n_operations:
                    self.pheromone[op_from][op_to] += delta
        
        # Mantener feromonas en rango razonable
        self.pheromone = np.clip(self.pheromone, 0.01, 10.0)
    
    def optimize(self) -> Tuple[int, List[int], Dict]:
        """Ejecuta el algoritmo ACO"""
        print(f"🐜 Iniciando ACO con {self.n_ants} hormigas y {self.n_iterations} iteraciones")
        print(f"📊 Problema: {self.n_jobs} trabajos, {self.n_machines} máquinas, {self.n_operations} operaciones")
        print("-" * 60)
        
        for iteration in range(self.n_iterations):
            solutions = []
            makespans = []
            
            # Cada hormiga construye una solución
            for ant in range(self.n_ants):
                solution = self._construct_solution()
                makespan, schedule = self._calculate_makespan(solution)
                
                solutions.append(solution)
                makespans.append(makespan)
                
                # Actualizar mejor solución
                if makespan < self.best_makespan:
                    self.best_makespan = makespan
                    self.best_schedule = schedule
                    self.best_solution = solution
            
            # Actualizar feromonas
            self._update_pheromones(solutions, makespans)
            
            # Guardar historial
            avg_makespan = np.mean(makespans)
            self.history.append((self.best_makespan, avg_makespan))
            
            # Imprimir progreso cada 10 iteraciones
            if (iteration + 1) % 10 == 0:
                print(f"Iteración {iteration + 1:3d} | Mejor: {self.best_makespan:3d} | "
                      f"Promedio: {avg_makespan:.2f}")
        
        print("-" * 60)
        print(f"✅ Optimización completada!")
        print(f"🏆 Mejor makespan: {self.best_makespan}")
        
        return self.best_makespan, self.best_solution, self.best_schedule
    
    def print_schedule(self):
        """Imprime el cronograma detallado"""
        if self.best_schedule is None:
            print("No hay cronograma disponible")
            return
        
        print("\n📅 CRONOGRAMA ÓPTIMO")
        print("=" * 60)
        
        for machine in range(self.n_machines):
            print(f"\n🔧 Máquina {machine}:")
            if self.best_schedule[machine]:
                for job, op, start, end in self.best_schedule[machine]:
                    duration = end - start
                    print(f"  J{job}-Op{op}: [{start:3d} → {end:3d}] ({duration} min)")
            else:
                print("  Sin operaciones")
        
        print("\n" + "=" * 60)
    
    def plot_convergence(self):
        """Grafica la convergencia del algoritmo"""
        if not self.history:
            print("No hay historial disponible")
            return
        
        best_history = [h[0] for h in self.history]
        avg_history = [h[1] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(best_history, 'b-', linewidth=2, label='Mejor makespan')
        plt.plot(avg_history, 'r--', linewidth=1.5, label='Makespan promedio')
        plt.xlabel('Iteración', fontsize=12)
        plt.ylabel('Makespan', fontsize=12)
        plt.title('Convergencia del ACO', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('aco_convergence.png', dpi=150)
        print("\n📊 Gráfico guardado como 'aco_convergence.png'")
        plt.show()


def main():
    """Ejemplo de uso con el problema del taller"""
    
    # Datos del problema JSP del taller
    # Formato: [(máquina, tiempo), ...]
    jobs = [
        [(0, 3), (1, 2), (2, 2)],  # J1: M1(3) → M2(2) → M3(2)
        [(1, 2), (2, 1), (0, 4)],  # J2: M2(2) → M3(1) → M1(4)
        [(2, 2), (0, 1), (1, 3)]   # J3: M3(2) → M1(1) → M2(3)
    ]
    
    print("\n" + "="*60)
    print("🏭 JOB-SHOP SCHEDULING PROBLEM con ACO")
    print("="*60)
    
    # Crear instancia de ACO
    aco = JobShopACO(
        jobs_data=jobs,
        n_ants=20,
        n_iterations=100,
        alpha=1.0,    # Influencia de feromonas
        beta=2.0,     # Influencia heurística
        rho=0.5,      # Tasa de evaporación
        Q=100.0       # Constante de refuerzo
    )
    
    # Optimizar
    best_makespan, best_solution, best_schedule = aco.optimize()
    
    # Mostrar resultados
    aco.print_schedule()
    
    print(f"\n🔢 Secuencia óptima de trabajos:")
    print(f"   {' → '.join([f'J{j}' for j in best_solution])}")
    
    # Graficar convergencia
    aco.plot_convergence()
    
    print("\n✅ ¡Taller completado exitosamente!")


if __name__ == "__main__":
    main()

