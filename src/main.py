import statistics as stats
import sys
import time

import numpy as np

from UWL import UWL
from CWL import CWL

def read_heuristics(heuristics : str) -> list[str]:
    """Return a list of heuristic names from a text formatted as [HR1,HR2,...,HRN].
       Available heuristics PPU, CG, CMI.

    Args:
        heuristics (str): Input list of (meta)-heuristics.
    """

    return heuristics[1:-1].split(",")

def main():

    start = time.time()

    filename = "./instances/cap131.txt"
    opt = None

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    try:
        uwl = UWL(filename, opt)
    except:
        print(f"[ERR] - Invalid file name {filename}")
        return

    heuristics_d = {
        "PPU" : ("heuristic_nearest_warehouse", {}),
        "CG" :("heuristic_glutton_opening", {}),
        "CMI" : ("heuristic_cover", {"lim": 1000}),
    }

    input_heuristics = ["PPU", "CG", "CMI"]

    if len(sys.argv) > 2:
        input_heuristics = read_heuristics(sys.argv[2])

    try:
        heuristics = [heuristics_d[h] for h in input_heuristics]
    except:
        print(f"[ERR] - Invalid heuristics {input_heuristics} - only possible choices among, [PPU,CG,CMI].")
        return

    meta_d = {
        "C1-H" : ("descent", {"k_max": 1}),
        "CP-H" : ("descent", {"k_max": 3}),
        "AP-H" : ("random_descent", {"max_time": 5}),
        "RC" : ("simulated_annealing", {"alpha": 0.999,"max_time": 5})
    }

    input_meta = ["CP-H"]

    if len(sys.argv) > 3:
        input_meta = read_heuristics(sys.argv[3])
    
    try:
        meta = [meta_d[m] for m in input_meta]
    except:
        print(f"[ERR] - Invalid metaheuristics {input_meta} - only possible choices among, [C1-H,CP-H,AP-H,RC].")
        return

    for func_name, params in heuristics:

        func = getattr(uwl, func_name)

        if params:
            open_warehouses, assignated_warehouses, best_cost = func(**params)
        else:
            open_warehouses, assignated_warehouses, best_cost = func()
        
        for meta_name, meta_params in meta:
            
            new_meta_params = meta_params.copy()
            new_meta_params["open_warehouses"] = open_warehouses
            new_meta_params["assignated_warehouses"] = assignated_warehouses
            new_meta_params["best_cost"] = best_cost

            meta_func = getattr(uwl, meta_name)
            
            open_warehouses, assignated_warehouses, best_cost = meta_func(**new_meta_params)

    uwl.print()
    print(f"Temps total d'exécution : {(time.time() - start):.2f} secondes.")
    return

###############################################
##### SOME UTILS FUNCTIONS FOR EVALUATION #####
###############################################

def get_heuristic_sols(uwl):
    sols = []
    sols.append(uwl.heuristic_glutton_opening())
    sols.append(uwl.heuristic_nearest_warehouse())
    sols.append(uwl.heuristic_cover(1000))
    return sols

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

    print(output_tab)

    general_tab = """| Statistiques d'évaluation des heuristiques | PPU | CG  | CMI |
| ----------------------------- | --- | --- | --- |"""

    general_tab += f"\n| **Durée d'exécution moyenne (s)** | {stats.mean(stats_data['PPU']['durations']):.2f} | {stats.mean(stats_data['CG']['durations']):.2f} | {stats.mean(stats_data['CMI']['durations']):.2f} |"
    general_tab += f"\n| **Erreur moyenne (%)** | {stats.mean(stats_data['PPU']['errors']):.2f} | {stats.mean(stats_data['CG']['errors']):.2f} | {stats.mean(stats_data['CMI']['errors']):.2f} |"
    general_tab += f"\n| **Ecart-type sur l'erreur (%)** | {stats.stdev(stats_data['PPU']['errors']):.2f} | {stats.stdev(stats_data['CG']['errors']):.2f} | {stats.stdev(stats_data['CMI']['errors']):.2f} |"
    general_tab += f"\n| **Erreur médiane (%)** | {stats.median(stats_data['PPU']['errors']):.2f} | {stats.median(stats_data['CG']['errors']):.2f} | {stats.median(stats_data['CMI']['errors']):.2f} |"
    general_tab += f"\n| **Erreur minimum (%)** | {min(stats_data['PPU']['errors']):.2f} | {min(stats_data['CG']['errors']):.2f} | {min(stats_data['CMI']['errors']):.2f} |"
    general_tab += f"\n| **Erreur maximum (%)** | {max(stats_data['PPU']['errors']):.2f} | {max(stats_data['CG']['errors']):.2f} | {max(stats_data['CMI']['errors']):.2f} |"

    print(general_tab)

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

    print(output_tab)

    general_tab = """| Statistiques d'évaluation des heuristiques | C1-H | CP-H  | AP-H |
| ----------------------------- | --- | --- | --- |"""

    general_tab += f"\n| **Durée d'exécution moyenne (s)** | {stats.mean(stats_data['C1-H']['durations']):.2f} | {stats.mean(stats_data['CP-H']['durations']):.2f} | {stats.mean(stats_data['AP-H']['durations']):.2f} |"
    general_tab += f"\n| **Erreur moyenne (%)** | {stats.mean(stats_data['C1-H']['errors']):.2f} | {stats.mean(stats_data['CP-H']['errors']):.2f} | {stats.mean(stats_data['AP-H']['errors']):.2f} |"
    general_tab += f"\n| **Ecart-type sur l'erreur (%)** | {stats.stdev(stats_data['C1-H']['errors']):.2f} | {stats.stdev(stats_data['CP-H']['errors']):.2f} | {stats.stdev(stats_data['AP-H']['errors']):.2f} |"
    general_tab += f"\n| **Erreur médiane (%)** | {stats.median(stats_data['C1-H']['errors']):.2f} | {stats.median(stats_data['CP-H']['errors']):.2f} | {stats.median(stats_data['AP-H']['errors']):.2f} |"
    general_tab += f"\n| **Erreur minimum (%)** | {min(stats_data['C1-H']['errors']):.2f} | {min(stats_data['CP-H']['errors']):.2f} | {min(stats_data['AP-H']['errors']):.2f} |"
    general_tab += f"\n| **Erreur maximum (%)** | {max(stats_data['C1-H']['errors']):.2f} | {max(stats_data['CP-H']['errors']):.2f} | {max(stats_data['AP-H']['errors']):.2f} |"

    print(general_tab)

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

    print(output_tab)
    
    general_tab = """| Statistiques d'évaluation de la métaheuristique | RC |
| ----------------------------- | --- |"""

    general_tab += f"\n| **Durée d'exécution moyenne (s)** | {stats.mean(durations):.2f} |"
    general_tab += f"\n| **Erreur moyenne (%)** | {stats.mean(errors):.2f} |"
    general_tab += f"\n| **Ecart-type sur l'erreur (%)** | {stats.stdev(errors):.2f} |"
    general_tab += f"\n| **Erreur médiane (%)** | {stats.median(errors):.2f} |"
    general_tab += f"\n| **Erreur minimum (%)** | {min(errors):.2f} |"
    general_tab += f"\n| **Erreur maximum (%)** | {max(errors):.2f} |"

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
    main()
    #evaluate_simulated_annealing("instances/optima.txt")
    #evaluate_descents("instances/optima.txt")