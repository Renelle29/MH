from UWL import UWL

def main():
    uwl = UWL("./instances/cap71.txt")
    uwl.heuristic_one_warehouse()
    uwl.print()

if __name__ == '__main__':
    main()