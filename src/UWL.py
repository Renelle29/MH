import numpy as np

class UWL:

    def __init__(self, filename):
        
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        m, n = map(int, lines[0].split())

        self.m = m
        self.n = n

        self.build_costs = np.zeros(m)

        for i in range(1,m+1):
            line = lines[i]
            _, cost = map(float, line.split())
            self.build_costs[i-1] = cost

        self.distance_matrix = np.zeros((n,m))
        
        all_vals = []
        for line in lines[m+1:]:
            all_vals += list(map(float, line.split()))
            
        for i in range(n):
            for j in range(m):
                self.distance_matrix[i,j] = all_vals[1 + i + m * i + j]

        self.open_warehouses = np.zeros(m)
        self.open_warehouses[0] = 1
        self.assignated_warehouses = np.zeros(n)
        self.best_cost = self.compute_best_cost()

    def compute_best_cost(self):
        return np.sum(self.build_costs * self.open_warehouses) + np.sum(self.distance_matrix[np.arange(self.n), self.assignated_warehouses.astype(int)])

    def compute_cost(self, open_warehouses, assignated_warehouses):
        return np.sum(self.build_costs * open_warehouses) + np.sum(self.distance_matrix[np.arange(self.n), assignated_warehouses.astype(int)])

    def heuristic_one_warehouse(self):
        
        open_warehouses = np.zeros(self.m)
        assignated_warehouses = np.zeros(self.n) - 1

        for i in range(self.m):
            open_warehouses[i-1] = 0
            open_warehouses[i] = 1
            assignated_warehouses = assignated_warehouses + 1
            
            new_cost = self.compute_cost(open_warehouses,assignated_warehouses)

            if new_cost < self.best_cost:
                print(f"Found a better solution with One_Warehouse heuristic. New cost: {new_cost}")
                self.open_warehouses = open_warehouses
                self.assignated_warehouses = assignated_warehouses
                self.best_cost = new_cost

    def heuristic_nearest_warehouse(self):

        open_warehouses = np.zeros(self.m)
        assignated_warehouses = np.zeros(self.n)

        for i in range(self.n):
            closest_warehouse = np.argmin(self.distance_matrix[i])
            assignated_warehouses[i] = closest_warehouse
            open_warehouses[closest_warehouse] = 1

        new_cost = self.compute_cost(open_warehouses,assignated_warehouses)
        
        if new_cost < self.best_cost:
            print(f"Found a better solution with Nearest_Warehouse heuristic. New cost: {new_cost}")
            self.open_warehouses = open_warehouses
            self.assignated_warehouses = assignated_warehouses
            self.best_cost = new_cost

    def print(self):
        print(f"""---------------------------
Uncapacitated warehouse location problem:
Number of warehouses: {self.m}
Number of clients: {self.n}
Current best warehouse assignation: {self.assignated_warehouses}
Current best cost: {self.best_cost}
---------------------------""")