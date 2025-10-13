from UWL import UWL

def main():
    uwl = UWL("./instances/capc.txt")
    uwl.heuristic_one_warehouse()
    uwl.heuristic_nearest_warehouse()
    uwl.print()

if __name__ == '__main__':
    main()