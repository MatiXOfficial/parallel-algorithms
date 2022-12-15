#!/usr/bin/env python
import sys
import time
import copy
import math

import numpy as np

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

# config
G_MIN_COST = 5
G_MAX_COST = 20

# init
if len(sys.argv) not in [3, 4]:
    raise ValueError("Wrong number of arguments. Proper usage: python lab6.py N d")
N = int(sys.argv[len(sys.argv) - 2])
d = int(sys.argv[len(sys.argv) - 1])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Generate the graph
CIRCLE_R = 5

def generate_graph():
    G = np.zeros((N, N))
    points = np.empty((N, 2))
    np.random.seed(42)
    alpha = sorted(2 * np.pi * np.random.random(N))
    points[:, 0] = CIRCLE_R * np.cos(alpha)
    points[:, 1] = CIRCLE_R * np.sin(alpha)

    for u in range(N - 1):
        for v in range(u + 1, N):
            dist = math.dist(points[u], points[v])
            G[u][v] = G[v][u] = dist
            
    return G

#####
class Path:
    def __init__(self, u_start: int = 0) -> None:
        self.visited = set([u_start])
        self.path: list[int] = [u_start]
        self.cum_cost: float = 0
        
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
    
    def __repr__(self) -> str:
        result = ''
        result += f' visited: {self.visited}\n'
        result += f'    path: {self.path}\n'
        result += f'cum_cost: {self.cum_cost:0.4f}\n'
        return result
    
def choose_optimal_path(paths: list[Path]) -> Path:
    final_path = paths[0]
    for path in paths[1:]:
        if path.cum_cost < final_path.cum_cost:
            final_path = path
    return final_path
    
def generate_shortest_path(G: np.ndarray, path: Path, d: int = N, N: int = N, return_all: bool = False) -> Path|list[Path]:
    paths = [path]
    current_min = float('inf')
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
                    if len(new_path) == N:
                        current_min = min(current_min, new_path.cum_cost)

    if return_all:
        return final_paths
    else:
        return choose_optimal_path(final_paths)

def generate_init_paths(G: np.ndarray, d: int = d) -> list[Path]:
    if d > G.shape[0]:
        raise ValueError(f'd ({d}) cannot be bigger than N ({N})')

    path = Path()
    if d == 1:
        return [path]
    
    return generate_shortest_path(G, path, d, return_all=True)

def fun(x):
    return 'a'

if __name__ == '__main__':
    G = generate_graph()
    # print(G)
    start = time.time()
    init_paths = generate_init_paths(G)
    # print(init_paths)
    if d == N:
        paths = init_paths
    else:
        with MPIPoolExecutor() as executor:
            paths = list(executor.map(generate_shortest_path, [G] * len(init_paths), init_paths))
    end = time.time()
    print(f'{N}, {d}, {end - start:0.8f}') # N, d, time
    print(choose_optimal_path(paths))
