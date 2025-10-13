import numpy as np

from UWL import UWL

def main():
    uwl = UWL("./instances/capc.txt")
    uwl.heuristic_one_warehouse()
    uwl.heuristic_nearest_warehouse()
    uwl.heuristic_cover()
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

if __name__ == '__main__':
    main()