# 🐜 Taller ACO - Job-Shop Scheduling Problem

## Descripción
Implementación de Ant Colony Optimization (ACO) para resolver el Job-Shop Scheduling Problem (JSP).

## Problema JSP
- **Objetivo**: Minimizar el tiempo total de procesamiento (makespan)
- **Restricciones**: 
  - Cada trabajo tiene operaciones en orden específico
  - Cada operación requiere una máquina específica
  - Una máquina procesa solo una operación a la vez

## Uso
```bash
python aco_jsp.py
```

## Parámetros principales
- `n_ants`: Número de hormigas (default: 20)
- `n_iterations`: Número de iteraciones (default: 100)
- `alpha`: Influencia de feromonas (default: 1.0)
- `beta`: Influencia heurística (default: 2.0)
- `rho`: Tasa de evaporación (default: 0.5)

## Ejemplo de salida
El programa muestra:
- Mejor makespan encontrado
- Secuencia de operaciones óptima
- Cronograma detallado por máquina
- Gráfico de convergencia

## Componentes ACO
- **Feromonas**: Preferencia histórica entre operaciones
- **Heurística**: Inverso del tiempo de procesamiento
- **Probabilidad**: Combina feromonas y heurística con α y β
- **Evaporación**: Permite exploración evitando convergencia prematura

## Autor
Andres Felipe Morales Mejia — Código: 1004754257

## Profesor
Angel Augusto Agudelo Z

