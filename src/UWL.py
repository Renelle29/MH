"""This module allow to find a solution to a Uncapacitated Warehouse Location problem
by running various heuristics, neighborhood descents, and metaheuristics."""

from typing import Optional
import random as rd
import time
from itertools import combinations

import numpy as np

UWL_Solution = tuple[np.ndarray, np.ndarray, float]

class UWL:
    """Uncapacitated Warehouse Location [UWL]"""

    def __init__(self, file_path : str, opt : float = None) -> None:
        """Initialize an object representing a UWL instance problem from a defined file format

        Args:
            file_path (str): Path to the file reprensenting the instance.
                             see here for a description of the format:
                             https://people.brunel.ac.uk/~mastjjb/jeb/orlib/uncapinfo.html
            opt (float, optional): Optimal value of the instance. Defaults to None.
        """

        self.file_path = file_path

        with open(file_path, 'r') as f:
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

    def initialize_bad_solution(self) -> UWL_Solution:
        """Return a valid initial bad solution.

        Returns:
            UWL_Solution: Initial solution.
        """

        open_warehouses = np.zeros(self.m)
        open_warehouses[0] = 1
        assignated_warehouses = np.zeros(self.n)
        best_cost = self.compute_cost(open_warehouses,assignated_warehouses)

        return open_warehouses, assignated_warehouses, best_cost

    def compute_best_cost(self) -> float:
        """Return the cost of the current best solution.

        Returns:
            float: Cost of the current best solution.
        """

        return np.sum(self.build_costs * self.open_warehouses) + np.sum(self.distance_matrix[np.arange(self.n), self.assignated_warehouses.astype(int)])

    def compute_cost(self, open_warehouses : np.ndarray, assignated_warehouses: np.ndarray) -> float:
        """Compute the cost of a given solution.

        Args:
            open_warehouses (nd.ndarray): Binary array of size m of open warehouses.
            assignated_warehouses (nd.ndarray): Integer array of size n of each client designated warehouse.

        Returns:
            float: Cost of the given solution.
        """

        return np.sum(self.build_costs * open_warehouses) + np.sum(self.distance_matrix[np.arange(self.n), assignated_warehouses.astype(int)])

    def heuristic_one_warehouse(self) -> UWL_Solution:
        """Heuristic that returns the best solution
           when we restrict ourselves to solutions that contain ONLY ONE open warehouse.

        Returns:
            UWL_Solution: Solution found by the heuristic.
        """
        
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

    def heuristic_nearest_warehouse(self) -> UWL_Solution:
        """Heuristic that assign each client to its nearest warehouse.

        Returns:
            UWL_Solution: Solution found by the heuristic.
        """
        
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

    def heuristic_glutton_opening(self) -> UWL_Solution:
        """Heuristic that starts from the nearest_warehouse solution,
           and checks iteratively on each open warehouse, if it's best to close it.

        Returns:
            UWL_Solution: Solution found by the heuristic.
        """

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

    def hn_heuristic_set_cover(self, sets : np.ndarray, costs : np.ndarray) -> Optional[list[int]]:
        """Glutton heuristic solution for the problem of set covering.

        Args:
            sets (np.ndarray): (m,n) boolean matrix, listing the elements
                               that can be covered by each set.
            costs (np.ndarray): Cost of each set.

        Returns:
            Optional[list[int]]: List of the sets to pick in the solution,
                                 or None, if no such solution exists.
        """

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

    def heuristic_cover_K(self, max_dist : float) -> UWL_Solution:
        """Transform a UWL problem into a SetCover problem,
           by considering only clients within a given radius for each warehouse,
           and return the solution found by the SetCover heuristic.

        Args:
            max_dist (float): Each warehouse can serve clients located
                              less or equal than max_dist away.

        Returns:
            UWL_Solution: Solution found for this particular instance.
        """

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

    def heuristic_cover(self, lim : int = 10000) -> UWL_Solution:
        """Heuristic that try to find a good solution to the initial problem
           by considering successive SetCover instances, and finding good solutions for each.
           NB : if lim is lower than a number of different distance values, then the heuristic
           result becomes non-deterministic.

        Args:
            lim (int, optional): Maximum number of calls to heuristic_cover_K. Defaults to 10000.

        Returns:
            UWL_Solution: Solution found by the heuristic.
        """

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

    def k_hamming_neighborhood(self, x : np.ndarray, k : int = 1) -> list[np.ndarray]:
        """Returns the full k-hamming neighborhood for a given boolean array.

        Args:
            x (np.ndarray): Boolean array of the current solution.
            k (int, optional): Number of bit(s) to flip. Must be lower than 4. Defaults to 1.

        Returns:
            list[np.ndarray]: List of all neighbors of the current solution.
        """

        n = len(x)
        neighbors = []

        for idxs in combinations(range(n), k):
            neighbor = x.copy()
            neighbor[list(idxs)] = 1 - neighbor[list(idxs)]
            neighbors.append(neighbor)
        
        return neighbors

    def descent(self, k_max : int = 1, open_warehouses : np.ndarray = None, 
        assignated_warehouses : np.ndarray = None, best_cost : float = None) -> UWL_Solution:
        """Execute from a given solution a full descent method, by fully exploring the
           1,...,k_max-hamming neighborhoods untill no further improvement is found.

        Args:
            k_max (int, optional): Maximum size of the hamming-neighborhood.
                                   Must be lower than 4. Defaults to 1.
            open_warehouses (np.ndarray, optional): Open warehouses in the initial solution. Defaults to None.
            assignated_warehouses (np.ndarray, optional): Assignated warehouse in the initial solution. Defaults to None.
            best_cost (float, optional): Cost of the initial solution. Defaults to None.

        Returns:
            UWL_Solution: Best solution found during the descent.
        """

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
    
    def k_hamming_random_neighborhood(self, x : np.ndarray, k : int = 1, neighborhood_size : int = 200) -> list[np.ndarray]:
        """Return a list of randomly picked neighbors in a k-Hamming neighborhood.

        Args:
            x (np.ndarray): 
            k (int, optional): Boolean array of the current solution. Defaults to 1.
            neighborhood_size (int, optional): Number of neighbors to pick. Defaults to 200.

        Returns:
            list[np.ndarray]: List of maximum neighborhood_size neighbors found in the neighborhood of x.
        """
        
        n = len(x)
        neighbors = []

        for i in range(neighborhood_size):
            idxs = list(sorted(rd.sample(range(n), k)))
            neighbor = x.copy()
            neighbor[idxs] = 1 - neighbor[list(idxs)]
            neighbors.append(neighbor)
        
        return neighbors

    def random_descent(self, k_max : int = None, open_warehouses : np.ndarray = None,
        assignated_warehouses : np.ndarray = None, best_cost : float = None, 
        neighborhood_size : int = 200, max_time : int = 1) -> UWL_Solution:
        """Execute from a given solution a random descent method, by randomly exploring the
           1,...,k_max-hamming neighborhoods untill no more time is left.

        Args:
            k_max (int, optional): Maximum size of the hamming-neighborhood. Defaults to None.
            open_warehouses (np.ndarray, optional): Open warehouses in the initial solution. Defaults to None.
            assignated_warehouses (np.ndarray, optional): Assignated warehouse in the initial solution. Defaults to None.
            best_cost (float, optional): Cost of the initial solution. Defaults to None.
            neighborhood_size (int, optional): Maximum number of neighbors to pick in each neighborhood. Defaults to 200.
            max_time (int, optional): Maximum duration in seconds to wait from the last improvement before stopping.
                                      Defaults to 1.

        Returns:
            UWL_Solution: Best solution found during the descent.
        """
        
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

    def simulated_annealing(self, open_warehouses : np.ndarray = None, assignated_warehouses : np.ndarray = None,
        best_cost : float = None, temp : int = None, alpha : float = 0.999,
        max_time : int = 1) -> UWL_Solution:
        """Run a simulated annealing method from a given initial solution,
           and randomly picking neighbors in the 1-Hamming neighborhood.

        Args:
            open_warehouses (np.ndarray, optional): Open warehouses in the initial solution. Defaults to None.
            assignated_warehouses (np.ndarray, optional): Assignated warehouse in the initial solution. Defaults to None.
            best_cost (float, optional): Cost of the initial solution. Defaults to None.
            temp (int, optional): Initial temperature of the method. Defaults to None.
            alpha (float, optional): Linear coefficient to decrease temp every iterations. Defaults to 0.999.
            max_time (int, optional): Maximum duration in seconds to wait from the last improvement before stopping.
                                      Defaults to 1.

        Returns:
            UWL_Solution: Best solution found during the metaheuristic.
        """

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

                delta_cost = new_cost - current_cost
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
        """Print the best solution found to the problem so far."""

        print(f"""---------------------------
Uncapacitated warehouse location problem:
Name: {self.file_path}
Number of warehouses: {self.m}
Number of clients: {self.n}
Current best warehouse assignation: {self.assignated_warehouses}
Current best cost: {self.best_cost}
---------------------------""")