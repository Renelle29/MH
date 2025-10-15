import numpy as np
import sys

from UWL import UWL

def main():
    filename = "./instances/cap71.txt"

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    uwl = UWL(filename)
    uwl.heuristic_one_warehouse()
    uwl.heuristic_nearest_warehouse()
    uwl.heuristic_cover(1000)
    uwl.descent(3) # Works fine up to kmax = 3
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