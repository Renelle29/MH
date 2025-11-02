import numpy as np
import random as rd
import time
from itertools import combinations

class UWL:
    """Uncapacitated Facility Location"""

    def __init__(self, filename, opt=None):
        
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

        self.opt = opt

    def initialize_bad_solution(self):
        open_warehouses = np.zeros(self.m)
        open_warehouses[0] = 1
        assignated_warehouses = np.zeros(self.n)
        best_cost = self.compute_cost(open_warehouses,assignated_warehouses)

        return open_warehouses, assignated_warehouses, best_cost

    def compute_best_cost(self):
        return np.sum(self.build_costs * self.open_warehouses) + np.sum(self.distance_matrix[np.arange(self.n), self.assignated_warehouses.astype(int)])

    def compute_cost(self, open_warehouses, assignated_warehouses):
        return np.sum(self.build_costs * open_warehouses) + np.sum(self.distance_matrix[np.arange(self.n), assignated_warehouses.astype(int)])

    def heuristic_one_warehouse(self):
        
        best_open_warehouses, best_assignated_warehouses, best_cost = self.initialize_bad_solution()
        print(f"-------- Starting One_Warehouse heuristic --------")

        open_warehouses = np.zeros(self.m)
        assignated_warehouses = np.zeros(self.n) - 1

        for i in range(self.m):
            open_warehouses[i-1] = 0
            open_warehouses[i] = 1
            assignated_warehouses = assignated_warehouses + 1
            
            new_cost = self.compute_cost(open_warehouses,assignated_warehouses)

            if new_cost < best_cost:
                print(f"Found a better solution with One_Warehouse heuristic. New cost: {new_cost}")
                best_open_warehouses = open_warehouses
                best_assignated_warehouses = assignated_warehouses
                best_cost = new_cost

        if best_cost < self.best_cost:
            self.open_warehouses = best_open_warehouses
            self.assignated_warehouses = best_assignated_warehouses
            self.best_cost = best_cost

        return best_open_warehouses, best_assignated_warehouses, best_cost

    def heuristic_nearest_warehouse(self):
        
        print(f"-------- Starting Nearest_Warehouse heuristic --------")

        open_warehouses = np.zeros(self.m)
        assignated_warehouses = np.zeros(self.n)
        best_cost = np.sum(self.distance_matrix)

        for i in range(self.n):
            closest_warehouse = np.argmin(self.distance_matrix[i])
            assignated_warehouses[i] = closest_warehouse
            open_warehouses[closest_warehouse] = 1

        new_cost = self.compute_cost(open_warehouses,assignated_warehouses)

        if new_cost < best_cost:
            print(f"Found a better solution with Nearest_Warehouse heuristic. New cost: {new_cost}")
        
        if new_cost < self.best_cost:
            self.open_warehouses = open_warehouses
            self.assignated_warehouses = assignated_warehouses
            self.best_cost = new_cost
        
        return open_warehouses, assignated_warehouses, new_cost

    def heuristic_glutton_opening(self):
        best_open_warehouses, best_assignated_warehouses, best_cost = self.heuristic_one_warehouse()

        for i in range(len(best_open_warehouses)):
            
            if best_open_warehouses[i] != 1:
                best_open_warehouses[i] = 1

                open_idx = np.where(best_open_warehouses == 1)[0]
                if len(open_idx) == 0:
                    continue
                local_min_idx = np.argmin(self.distance_matrix[:, open_idx], axis=1)
                assignated_warehouses = open_idx[local_min_idx]
                    
                new_cost = self.compute_cost(best_open_warehouses,assignated_warehouses)

                if new_cost > best_cost:
                    best_open_warehouses[i] = 0
                else:
                    print(f"Found a better solution with Glutton_Opening heuristic. New cost: {new_cost}")
                    best_cost = new_cost

        open_idx = np.where(best_open_warehouses == 1)[0]
        local_min_idx = np.argmin(self.distance_matrix[:, open_idx], axis=1)
        assignated_warehouses = open_idx[local_min_idx]
        new_cost = self.compute_cost(best_open_warehouses,assignated_warehouses)

        if new_cost < self.best_cost:
            self.open_warehouses = best_open_warehouses
            self.assignated_warehouses = assignated_warehouses
            self.best_cost = new_cost

        return best_open_warehouses, assignated_warehouses, new_cost

    def hn_heuristic_set_cover(self, sets, costs):
        
        selected_sets = []
    
        uncovered_mask = np.ones(sets.shape[1], dtype=bool)

        epsilon = 1e-8
        safe_costs = np.where(costs == 0, epsilon, costs)
        
        while np.any(uncovered_mask):
            
            gains = np.sum(sets & uncovered_mask, axis=1) / safe_costs
            
            if (np.max(gains) <= 0):
                return None

            best_set_idx = np.argmax(gains)
            selected_sets.append(int(best_set_idx))
            
            uncovered_mask &= ~sets[best_set_idx]
        
        return selected_sets

    def heuristic_cover_K(self, max_dist):
        bool_matrix = self.distance_matrix.T <= max_dist
        selected_sets = self.hn_heuristic_set_cover(bool_matrix, self.build_costs)

        if selected_sets == None:
            return None

        selected_sets.sort()
        selected_sets = np.array(selected_sets, dtype=int)

        local_min_idx = np.argmin(self.distance_matrix[:, selected_sets], axis=1)
        assignated_warehouses = selected_sets[local_min_idx]

        open_warehouses = np.zeros(self.m)
        open_warehouses[selected_sets] = 1

        new_cost = self.compute_cost(open_warehouses, assignated_warehouses)
        
        return open_warehouses, assignated_warehouses, new_cost

    def heuristic_cover(self, lim=10000):
        best_open_warehouses, best_assignated_warehouses, best_cost = self.initialize_bad_solution()
        print(f"-------- Starting the Cover heuristic --------")

        max_min_dist = np.max(np.min(self.distance_matrix, axis=1))

        flat_dist = self.distance_matrix.flatten()
        sorted_dist = flat_dist[flat_dist > max_min_dist]
        np.random.shuffle(sorted_dist) 
        sorted_dist = sorted_dist[:lim]

        for dist in sorted_dist:
            open_warehouses, assignated_warehouses, new_cost = self.heuristic_cover_K(dist)

            if new_cost < best_cost:
                print(f"Found a better solution with the Cover heuristic. New cost: {new_cost}")
                best_open_warehouses = open_warehouses
                best_assignated_warehouses = assignated_warehouses
                best_cost = new_cost
        
        if best_cost < self.best_cost:
            self.open_warehouses = best_open_warehouses
            self.assignated_warehouses = best_assignated_warehouses
            self.best_cost = best_cost

        return best_open_warehouses, best_assignated_warehouses, best_cost

    def k_hamming_neighborhood(self, x, k=1):
        n = len(x)
        neighbors = []

        for idxs in combinations(range(n), k):
            neighbor = x.copy()
            neighbor[list(idxs)] = 1 - neighbor[list(idxs)]
            neighbors.append(neighbor)
        
        return neighbors

    def descent(self, k_max=1, open_warehouses=None, assignated_warehouses=None, best_cost=None):

        best_open_warehouses = open_warehouses
        best_assignated_warehouses = assignated_warehouses

        if open_warehouses is None:
            best_open_warehouses = self.open_warehouses
        if assignated_warehouses is None:
            best_assignated_warehouses = self.assignated_warehouses
        if best_cost is None:
            best_cost = self.best_cost
        
        print(f"-------- Starting the Descent from a solution of cost: {best_cost} --------")

        k = 1

        while k <= k_max:
            improved = False

            neighborhood = self.k_hamming_neighborhood(best_open_warehouses,k)

            for open_warehouses in neighborhood:
                
                open_idx = np.where(open_warehouses == 1)[0]
                if len(open_idx) == 0:
                    continue
                local_min_idx = np.argmin(self.distance_matrix[:, open_idx], axis=1)
                assignated_warehouses = open_idx[local_min_idx]
                    
                new_cost = self.compute_cost(open_warehouses,assignated_warehouses)

                if new_cost < best_cost:
                    print(f"Found a better solution during the Descent - {k}-Hamming neighborhood. New cost: {new_cost}")
                    best_open_warehouses = open_warehouses
                    best_assignated_warehouses = assignated_warehouses
                    best_cost = new_cost
                    improved = True
                    
                    if self.opt is not None and round(best_cost,3) == round(self.opt,3):
                        break
            
            if self.opt is not None and round(best_cost,3) == round(self.opt,3):
                break

            if not improved:
                k += 1

        if best_cost < self.best_cost:
            self.open_warehouses = best_open_warehouses
            self.assignated_warehouses = best_assignated_warehouses
            self.best_cost = best_cost

        return best_open_warehouses, best_assignated_warehouses, best_cost
    
    def k_hamming_random_neighborhood(self, x, k=1, neighborhood_size=200):
        n = len(x)
        neighbors = []

        for i in range(neighborhood_size):
            idxs = list(sorted(rd.sample(range(n), k)))
            neighbor = x.copy()
            neighbor[idxs] = 1 - neighbor[list(idxs)]
            neighbors.append(neighbor)
        
        return neighbors

    def random_descent(self, k_max=None, open_warehouses=None, assignated_warehouses=None, best_cost=None, neighborhood_size=200, max_time=1):
        
        best_open_warehouses = open_warehouses
        best_assignated_warehouses = assignated_warehouses

        if open_warehouses is None:
            best_open_warehouses = self.open_warehouses
        if assignated_warehouses is None:
            best_assignated_warehouses = self.assignated_warehouses
        if best_cost is None:
            best_cost = self.best_cost
        if k_max is None:
            k_max = len(best_open_warehouses)
        
        print(f"-------- Starting the Random Descent from a solution of cost: {best_cost} --------")

        start = time.time()

        while (time.time() - start) < max_time:
            
            k = 1

            while k <= k_max:
                improved = False

                neighborhood = self.k_hamming_random_neighborhood(best_open_warehouses,k,neighborhood_size)

                for open_warehouses in neighborhood:
                
                    open_idx = np.where(open_warehouses == 1)[0]
                    if len(open_idx) == 0:
                        continue
                    local_min_idx = np.argmin(self.distance_matrix[:, open_idx], axis=1)
                    assignated_warehouses = open_idx[local_min_idx]
                        
                    new_cost = self.compute_cost(open_warehouses,assignated_warehouses)

                    if new_cost < best_cost:
                        print(f"Found a better solution during the Random Descent - {k}-Hamming neighborhood. New cost: {new_cost}")
                        best_open_warehouses = open_warehouses
                        best_assignated_warehouses = assignated_warehouses
                        best_cost = new_cost
                        start = time.time()
                        
                if self.opt is not None and round(best_cost,3) == round(self.opt,3):
                    break

                k += 1
            
            if self.opt is not None and round(best_cost,3) == round(self.opt,3):
                break

        if best_cost < self.best_cost:
            self.open_warehouses = best_open_warehouses
            self.assignated_warehouses = best_assignated_warehouses
            self.best_cost = best_cost

        return best_open_warehouses, best_assignated_warehouses, best_cost

    def simulated_annealing(self, open_warehouses=None, assignated_warehouses=None, best_cost=None, temp=None, alpha=0.95, max_time=1):
        
        print(f"-------- Starting the Simulated Annealing from a solution of cost: {best_cost} --------")

        if temp is None:
            temp = max(np.max(self.distance_matrix),np.max(self.build_costs))/1000
            print(f"Starting a simulated annealing with initial temp of {temp}")

        best_open_warehouses = open_warehouses
        best_assignated_warehouses = assignated_warehouses

        if open_warehouses is None:
            best_open_warehouses = self.open_warehouses
        if assignated_warehouses is None:
            best_assignated_warehouses = self.assignated_warehouses
        if best_cost is None:
            best_cost = self.best_cost

        current_open_warehouses = best_open_warehouses
        current_assignated_warehouses = best_assignated_warehouses
        current_cost = best_cost

        start = time.time()

        while (time.time() - start) < max_time:
            
            neighbor = self.k_hamming_random_neighborhood(current_open_warehouses,neighborhood_size=1)[0]
            open_idx = np.where(neighbor == 1)[0]

            try:
                
                local_min_idx = np.argmin(self.distance_matrix[:, open_idx], axis=1)
                current_assignated_warehouses = open_idx[local_min_idx]
                        
                new_cost = self.compute_cost(neighbor,current_assignated_warehouses)

                delta_cost = new_cost - best_cost
                compare_value = np.exp((-1 * delta_cost) / temp)
                if rd.random() < compare_value:
                    #print(f"Switching current solution, comp value: {compare_value}")
                    if new_cost < best_cost:
                        print(f"Found a better solution during the Simulated Annealing. New cost: {new_cost}")
                        best_open_warehouses = neighbor
                        best_assignated_warehouses = current_assignated_warehouses
                        best_cost = new_cost
                        if self.opt is not None and round(best_cost,3) == round(self.opt,3):
                            break
                    current_open_warehouses = neighbor
                    current_cost = new_cost
                    start = time.time()

            except:
                if len(open_idx) == 0:
                    print("Reached a solution where all warehouses were closed.")
                else:
                    print("Unknown error - to be investigated.")
                pass

            temp *= alpha

        if best_cost < self.best_cost:
            self.open_warehouses = best_open_warehouses
            self.assignated_warehouses = best_assignated_warehouses
            self.best_cost = best_cost

        return best_open_warehouses, best_assignated_warehouses, best_cost


    def print(self):
        print(f"""---------------------------
Uncapacitated warehouse location problem:
Number of warehouses: {self.m}
Number of clients: {self.n}
Current best warehouse assignation: {self.assignated_warehouses}
Current best cost: {self.best_cost}
---------------------------""")