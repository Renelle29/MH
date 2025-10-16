import numpy as np
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