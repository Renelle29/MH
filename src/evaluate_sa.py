import statistics as stats
import sys
import time

import numpy as np

from UWL import UWL
from CWL import CWL

def get_heuristic_sols(uwl):
    sols = []
    sols.append(uwl.heuristic_glutton_opening())
    sols.append(uwl.heuristic_nearest_warehouse())
    sols.append(uwl.heuristic_cover(1000))
    return sols

def evaluate_simulated_annealing(filename, base_path="instances/", min_instance=0, max_instance=1000):

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    instances, opt_values = [], []
    for line in lines[:]:
        vals = line.split()
        instances.append(str(vals[0]))
        opt_values.append(float(vals[1]))

    output_tab = """| Instance |                 |      RC      | RC     | RC   |
| -------- | :-------------: | :-----------: | :-----: | :---: | :------------: | :-----: | :---: | :------------: | :-----: | :---: |
|          | Valeur Optimale |    Valeur     | Temps   | Gap   |"""

    durations = []
    errors = []

    for inst, opt in zip(instances[min_instance:max_instance], opt_values[min_instance:max_instance]):
        filepath = f"{base_path}{inst}.txt"
        new_line = f"| {inst} | {opt:.3f}"
        uwl = UWL(filepath, opt)

        heuristic_sols = get_heuristic_sols(uwl)
        final_cost = 10**15

        for open_warehouses, assignated_warehouses, best_cost in heuristic_sols:
            
            start = time.time()

            _, _, new_cost = uwl.simulated_annealing(open_warehouses,assignated_warehouses,best_cost,alpha=0.999,max_time=5)

            if new_cost < final_cost:
                duration = time.time() - start
                final_cost = new_cost
        
        error = ((final_cost - opt) / opt) * 100

        new_line += f" | {final_cost:.3f} | {duration:.2f} | {error:.1f}"
        durations.append(duration)
        errors.append(error)

        output_tab += "\n" + new_line

    print("--------- Results per instance --------")
    print(output_tab)
    
    general_tab = """| Statistiques d'évaluation de la métaheuristique | RC |
| ----------------------------- | --- |"""

    general_tab += f"\n| **Durée d'exécution moyenne (s)** | {stats.mean(durations):.2f} |"
    general_tab += f"\n| **Erreur moyenne (%)** | {stats.mean(errors):.2f} |"
    general_tab += f"\n| **Ecart-type sur l'erreur (%)** | {stats.stdev(errors):.2f} |"
    general_tab += f"\n| **Erreur médiane (%)** | {stats.median(errors):.2f} |"
    general_tab += f"\n| **Erreur minimum (%)** | {min(errors):.2f} |"
    general_tab += f"\n| **Erreur maximum (%)** | {max(errors):.2f} |"

    print("--------- Aggregated results --------")
    print(general_tab)

evaluate_simulated_annealing("instances/optima.txt")