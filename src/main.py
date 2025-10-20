import numpy as np
import statistics as stats
import sys
import time

from UWL import UWL
from CWL import CWL

def main():

    filename = "./instances/cap71.txt"

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    uwl = UWL(filename)

    open_warehouses, assignated_warehouses, best_cost = uwl.heuristic_glutton_opening()
    #uwl.descent(3, open_warehouses, assignated_warehouses, best_cost)
    open_warehouses, assignated_warehouses, best_cost = uwl.heuristic_nearest_warehouse()
    #uwl.descent(3, open_warehouses, assignated_warehouses, best_cost)
    open_warehouses, assignated_warehouses, best_cost = uwl.heuristic_cover(1000)
    #uwl.descent(3, open_warehouses, assignated_warehouses, best_cost)
    uwl.print()

def evaluate_heuristics(filename, base_path="instances/"):

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    instances = []
    opt_values = []

    for line in lines:
        vals = line.split()
        instances.append(str(vals[0]))
        opt_values.append(float(vals[1]))

    output_tab = """| Instance |                 |      PPU      | PPU     | PPU   | CG             | CG      | CG    | CMI            | CMI     | CMI   |
| -------- | :-------------: | :-----------: | :-----: | :---: | :------------: | :-----: | :---: | :------------: | :-----: | :---: |
|          | Valeur Optimale |    Valeur     | Temps   | Gap   | Valeur         | Temps   | Gap   | Valeur         | Temps   | Gap   |"""

    ppu_durations = []
    ppu_errors = []
    cg_durations = []
    cg_errors = []
    cmi_durations = []
    cmi_errors = []

    for i in range(len(instances)):
        filename = base_path + instances[i] + ".txt"
        opt = opt_values[i]
        new_line = f"| {instances[i]} | {opt:.3f}"

        uwl = UWL(filename)

        start = time.time()
        open_warehouses, assignated_warehouses, best_cost = uwl.heuristic_nearest_warehouse()
        duration = time.time()-start
        error = ((best_cost - opt)/opt)*100
        new_line += f" | {best_cost:.3f} | {duration:.2f} | {error:.1f}"
        ppu_durations.append(duration)
        ppu_errors.append(error)

        start = time.time()
        open_warehouses, assignated_warehouses, best_cost = uwl.heuristic_glutton_opening()
        duration = time.time()-start
        error = ((best_cost - opt)/opt)*100
        new_line += f" | {best_cost:.3f} | {duration:.2f} | {error:.1f}"
        cg_durations.append(duration)
        cg_errors.append(error)

        start = time.time()
        open_warehouses, assignated_warehouses, best_cost = uwl.heuristic_cover(1000)
        duration = time.time()-start
        error = ((best_cost - opt)/opt)*100
        new_line += f" | {best_cost:.3f} | {duration:.2f} | {error:.1f}"
        cmi_durations.append(duration)
        cmi_errors.append(error)

        output_tab += "\n" + new_line
        
    print(output_tab)

    general_tab = """| Statistiques d'évaluation des heuristiques | PPU | CG  | CMI |
| ----------------------------- | --- | --- | --- |"""

    general_tab += f"\n| **Durée d'exécution moyenne (s)** | {stats.mean(ppu_durations):.2f} | {stats.mean(cg_durations):.2f} | {stats.mean(cmi_durations):.2f} |"
    general_tab += f"\n| **Erreur moyenne (%)** | {stats.mean(ppu_errors):.2f} | {stats.mean(cg_errors):.2f} | {stats.mean(cmi_errors):.2f} |"
    general_tab += f"\n| **Ecart-type sur l'erreur (%)** | {stats.stdev(ppu_errors):.2f} | {stats.stdev(cg_errors):.2f} | {stats.stdev(cmi_errors):.2f} |"
    general_tab += f"\n| **Erreur médiane (%)** | {stats.median(ppu_errors):.2f} | {stats.median(cg_errors):.2f} | {stats.median(cmi_errors):.2f} |"
    general_tab += f"\n| **Erreur minimum (%)** | {min(ppu_errors):.2f} | {min(cg_errors):.2f} | {min(cmi_errors):.2f} |"
    general_tab += f"\n| **Erreur maximum (%)** | {max(ppu_errors):.2f} | {max(cg_errors):.2f} | {max(cmi_errors):.2f} |"

    print(general_tab)

def random_tests():
    sets = np.array([
        [1,1,0,0],
        [1,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
    sets = sets.astype(bool)
    costs = np.array([10,1,1,1])

    print(uwl.hn_heuristic_set_cover(sets,costs))
    uwl.heuristic_cover_K(203364)

    uwl.heuristic_cover()
    uwl.print()

    neighborhood = uwl.k_hamming_neighborhood(np.zeros(5),5)
    print(neighborhood)

if __name__ == '__main__':
    evaluate_heuristics("instances/optima.txt")