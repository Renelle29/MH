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

    def compute_best_cost(self):
        return np.sum(self.build_costs * self.open_warehouses) * np.sum(self.distance_matrix[np.arange(self.n), self.assignated_warehouses.astype(int)])

    def print(self):
        print(f"""---------------------------
Uncapacitated warehouse location problem:
Number of warehouses: {self.m}
Number of clients: {self.n}
Current best warehouse assignation: {self.assignated_warehouses}
Current best cost: {self.compute_best_cost()}
---------------------------""")