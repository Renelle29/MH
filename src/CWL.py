import numpy as np
import random as rd
import time
from itertools import combinations

class CWL:
    """Capacitated Facility Location"""

    def __init__(self, filename):
        
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        m, n = map(int, lines[0].split())

        self.m = m
        self.n = n

        self.build_costs = np.zeros(m)
        self.capacity = np.zeros(m)

        for i in range(1,m+1):
            line = lines[i]
            capacity, cost = map(float, line.split())
            self.build_costs[i-1] = cost
            self.capacity[i-1] = capacity
        
        self.distance_matrix = np.zeros((n,m))
        
        all_vals = []
        for line in lines[m+1:]:
            all_vals += list(map(float, line.split()))

        for i in range(n):
            for j in range(m):
                self.distance_matrix[i,j] = all_vals[1 + i + m * i + j]

        self.demand = np.zeros(n)

        for i in range(n):
            self.demand[i] = all_vals[i + m * i]
        print(self.demand)

        self.open_warehouses = np.zeros(m)
        self.open_warehouses[0] = 1
        self.assignated_warehouses = np.zeros(n)
        self.best_cost = self.compute_best_cost()

    def compute_best_cost(self):
        return np.sum(self.build_costs * self.open_warehouses) + np.sum(self.distance_matrix[np.arange(self.n), self.assignated_warehouses.astype(int)])
