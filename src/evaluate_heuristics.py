import statistics as stats
import sys
import time

import numpy as np

from UWL import UWL
from CWL import CWL

def evaluate_heuristics(filename, base_path="instances/"):

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    instances, opt_values = [], []
    for line in lines:
        vals = line.split()
        instances.append(str(vals[0]))
        opt_values.append(float(vals[1]))

    output_tab = """| Instance |                 |      PPU      | PPU     | PPU   | CG             | CG      | CG    | CMI            | CMI     | CMI   |
| -------- | :-------------: | :-----------: | :-----: | :---: | :------------: | :-----: | :---: | :------------: | :-----: | :---: |
|          | Valeur Optimale |    Valeur     | Temps   | Gap   | Valeur         | Temps   | Gap   | Valeur         | Temps   | Gap   |"""

    stats_data = {
        'PPU': {'durations': [], 'errors': [], 'method': 'heuristic_nearest_warehouse'},
        'CG':  {'durations': [], 'errors': [], 'method': 'heuristic_glutton_opening'},
        'CMI': {'durations': [], 'errors': [], 'method': 'heuristic_cover', 'arg': 1000}
    }

    for inst, opt in zip(instances, opt_values):
        filepath = f"{base_path}{inst}.txt"
        new_line = f"| {inst} | {opt:.3f}"
        uwl = UWL(filepath)

        for key, data in stats_data.items():
            method = getattr(uwl, data['method'])
            start = time.time()
            if 'arg' in data:
                _, _, best_cost = method(data['arg'])
            else:
                _, _, best_cost = method()
            duration = time.time() - start
            error = ((best_cost - opt) / opt) * 100

            new_line += f" | {best_cost:.3f} | {duration:.2f} | {error:.1f}"
            data['durations'].append(duration)
            data['errors'].append(error)

        output_tab += "\n" + new_line

    print("--------- Results per instance --------")
    print(output_tab)

    general_tab = """| Statistiques d'évaluation des heuristiques | PPU | CG  | CMI |
| ----------------------------- | --- | --- | --- |"""

    general_tab += f"\n| **Durée d'exécution moyenne (s)** | {stats.mean(stats_data['PPU']['durations']):.2f} | {stats.mean(stats_data['CG']['durations']):.2f} | {stats.mean(stats_data['CMI']['durations']):.2f} |"
    general_tab += f"\n| **Erreur moyenne (%)** | {stats.mean(stats_data['PPU']['errors']):.2f} | {stats.mean(stats_data['CG']['errors']):.2f} | {stats.mean(stats_data['CMI']['errors']):.2f} |"
    general_tab += f"\n| **Ecart-type sur l'erreur (%)** | {stats.stdev(stats_data['PPU']['errors']):.2f} | {stats.stdev(stats_data['CG']['errors']):.2f} | {stats.stdev(stats_data['CMI']['errors']):.2f} |"
    general_tab += f"\n| **Erreur médiane (%)** | {stats.median(stats_data['PPU']['errors']):.2f} | {stats.median(stats_data['CG']['errors']):.2f} | {stats.median(stats_data['CMI']['errors']):.2f} |"
    general_tab += f"\n| **Erreur minimum (%)** | {min(stats_data['PPU']['errors']):.2f} | {min(stats_data['CG']['errors']):.2f} | {min(stats_data['CMI']['errors']):.2f} |"
    general_tab += f"\n| **Erreur maximum (%)** | {max(stats_data['PPU']['errors']):.2f} | {max(stats_data['CG']['errors']):.2f} | {max(stats_data['CMI']['errors']):.2f} |"

    print("--------- Aggregated results --------")
    print(general_tab)

evaluate_heuristics("instances/optima.txt")