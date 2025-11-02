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

if __name__ == '__main__':
    main()