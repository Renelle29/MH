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

def evaluate_descents(filename, base_path="instances/", min_instance=0, max_instance=1000):

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    instances, opt_values = [], []
    for line in lines[:]:
        vals = line.split()
        instances.append(str(vals[0]))
        opt_values.append(float(vals[1]))

    output_tab = """| Instance |                 |      C1-H      | C1-H     | C1-H   | CP-H             | CP-H      | CP-H    | AP-H            | AP-H     | AP-H   |
| -------- | :-------------: | :-----------: | :-----: | :---: | :------------: | :-----: | :---: | :------------: | :-----: | :---: |
|          | Valeur Optimale |    Valeur     | Temps   | Gap   | Valeur         | Temps   | Gap   | Valeur         | Temps   | Gap   |"""

    stats_data = {
        'C1-H': {'durations': [], 'errors': [], 'method': 'descent', 'k_max':1},
        'CP-H': {'durations': [], 'errors': [], 'method': 'descent', 'k_max':3},
        'AP-H': {'durations': [], 'errors': [], 'method': 'random_descent', 'max_time':5}
    }

    for inst, opt in zip(instances[min_instance:max_instance], opt_values[min_instance:max_instance]):
        filepath = f"{base_path}{inst}.txt"
        new_line = f"| {inst} | {opt:.3f}"
        uwl = UWL(filepath, opt)

        heuristic_sols = get_heuristic_sols(uwl)

        for key, data in stats_data.items():
            method = getattr(uwl, data['method'])
            final_cost = 10**15

            for open_warehouses, assignated_warehouses, best_cost in heuristic_sols:
                start = time.time()

                if 'k_max' in data:
                    _, _, new_cost = method(data['k_max'], open_warehouses, assignated_warehouses, best_cost)

                else:
                    _, _, new_cost = method(None, open_warehouses, assignated_warehouses, best_cost, max_time=data['max_time'])
                
                if new_cost < final_cost:
                    duration = time.time() - start
                    final_cost = new_cost

            error = ((final_cost - opt) / opt) * 100

            new_line += f" | {final_cost:.3f} | {duration:.2f} | {error:.1f}"
            data['durations'].append(duration)
            data['errors'].append(error)

        output_tab += "\n" + new_line

    print("--------- Results per instance --------")
    print(output_tab)

    general_tab = """| Statistiques d'évaluation des heuristiques | C1-H | CP-H  | AP-H |
| ----------------------------- | --- | --- | --- |"""

    general_tab += f"\n| **Durée d'exécution moyenne (s)** | {stats.mean(stats_data['C1-H']['durations']):.2f} | {stats.mean(stats_data['CP-H']['durations']):.2f} | {stats.mean(stats_data['AP-H']['durations']):.2f} |"
    general_tab += f"\n| **Erreur moyenne (%)** | {stats.mean(stats_data['C1-H']['errors']):.2f} | {stats.mean(stats_data['CP-H']['errors']):.2f} | {stats.mean(stats_data['AP-H']['errors']):.2f} |"
    general_tab += f"\n| **Ecart-type sur l'erreur (%)** | {stats.stdev(stats_data['C1-H']['errors']):.2f} | {stats.stdev(stats_data['CP-H']['errors']):.2f} | {stats.stdev(stats_data['AP-H']['errors']):.2f} |"
    general_tab += f"\n| **Erreur médiane (%)** | {stats.median(stats_data['C1-H']['errors']):.2f} | {stats.median(stats_data['CP-H']['errors']):.2f} | {stats.median(stats_data['AP-H']['errors']):.2f} |"
    general_tab += f"\n| **Erreur minimum (%)** | {min(stats_data['C1-H']['errors']):.2f} | {min(stats_data['CP-H']['errors']):.2f} | {min(stats_data['AP-H']['errors']):.2f} |"
    general_tab += f"\n| **Erreur maximum (%)** | {max(stats_data['C1-H']['errors']):.2f} | {max(stats_data['CP-H']['errors']):.2f} | {max(stats_data['AP-H']['errors']):.2f} |"

    print("--------- Aggregated results --------")
    print(general_tab)

evaluate_descents("instances/optima.txt")