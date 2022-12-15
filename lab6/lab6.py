#!/usr/bin/env python
import sys
import time
import copy

import numpy as np

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

# config
G_MIN_COST = 5
G_MAX_COST = 20

# init
if len(sys.argv) != 3:
    raise ValueError("Wrong number of arguments. Proper usage: python lab6.py N d")
N = int(sys.argv[1])
d = int(sys.argv[2])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Generate the graph
np.random.seed(42)
G = np.random.randint(low=G_MIN_COST, high=G_MAX_COST + 1, size=(N, N))
G = np.triu(G, k=1) + np.triu(G, k=1).T

class Path:
    def __init__(self, u_start: int = 0) -> None:
        self.visited = set([u_start])
        self.path: list[int] = [u_start]
        self.cum_cost = 0
        
    def append(self, v: int, c: int):
        if v in self.visited:
            raise ValueError(f'Vertex {v} has already been visited')
        self.visited.add(v)
        self.path.append(v)
        self.cum_cost += c
        return self
    
    def last(self) -> int:
        return self.path[-1]
    
    def __contains__(self, u: int) -> bool:
        return u in self.visited
    
    def __len__(self) -> int:
        return len(self.visited)

def generate_init_paths(G: np.ndarray, d: int) -> list[Path]:
    if d > G.shape[0]:
        raise ValueError(f'd ({d}) cannot be bigger than N ({N})')
    
    current_min = 0
    paths = [Path()]
    if d == 1:
        return paths
    
    final_paths = []
    
    while paths:
        path = paths.pop()
        u = path.last()
        for v in range(N):
            if v not in path:
                new_path = copy.deepcopy(path)
                new_path.append(v, G[u, v])
                if new_path.cum_cost >= current_min: # pruning
                    continue
                
                if len(new_path) < d:
                    paths.append(new_path)
                else:
                    final_paths.append(new_path)
                    current_min = new_path.cum_cost

    return final_paths

if __name__ == '__main__':
    test = []
    with MPIPoolExecutor() as executor:
        print(rank)
        test.append('a')
    print(test)
