#!/usr/bin/env python

import time
import sys
import math

from mpi4py import MPI

import numpy as np
from matplotlib import pyplot as plt

# params
if len(sys.argv) != 3:
    raise ValueError("Wrong number of arguments. Proper usage: python lab4_solver.py [map size] [number of iterations]")
N = int(sys.argv[1])
N_ITER = int(sys.argv[2])

# constants
RESULTS_PATH = './results.csv'

# init the table
Q = np.zeros((N + 2, N + 2))

# edges
Q[0] = 0                                              # φ(0, y)
Q[N + 1] = 1                                          # φ(x, N + 1)
Q[:, 0] = np.arange(0, 1.000000001, 1 / (N + 1))      # φ(x, 0)
Q[:, N + 1] = np.arange(0, 1.000000001, 1 / (N + 1))  # φ(N + 1, y) 

# thread data init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('Rank:', rank)

interval = N // size

x_start = 1 + interval * rank
x_end = x_start + interval if rank < size - 1 else N + 1 # the last process covers everything left
print(f'Calculating: [{x_start}..{x_end}]')

# single field
def calculate_field(x, y):
    Q[x, y] = (Q[x - 1, y] + Q[x, y + 1] + Q[x + 1, y] + Q[x, y - 1]) / 4

# calculations of the whole table chunk
def calculate_red():
    for x in range(x_start, x_end, 1):
        for y in range(1 + x % 2, N + 1, 2):
            calculate_field(x, y)

def calculate_black():
    for x in range(x_start, x_end, 1):
        for y in range(1 + (x + 1) % 2, N + 1, 2):
            calculate_field(x, y)

# communication
def send_message(rank, data):
    comm.send(data, dest=rank)

def send_red():
    # left
    if rank > 0:
        data = {}
        for y in range(1 + x_start % 2, N + 1, 2):
            data[(x_start, y)] = Q[x_start, y]
        send_message(rank - 1, data)
    
    # right
    if rank < size - 1:
        data = {}
        for y in range(1 + (x_end - 1) % 2, N + 1, 2):
            data[(x_end - 1, y)] = Q[x_end - 1, y]
        send_message(rank + 1, data)
    
def send_black():
    # left
    if rank > 0:
        data = {}
        for y in range(1 + (x_start + 1) % 2, N + 1, 2):
            data[(x_start, y)] = Q[x_start, y]
        send_message(rank - 1, data)
    
    # right
    if rank < size - 1:
        data = {}
        for y in range(1 + x_end % 2, N + 1, 2):
            data[(x_end - 1, y)] = Q[x_end - 1, y]
        send_message(rank + 1, data)
        
def recv_messages():
    for source_rank in [rank - 1, rank + 1]:
        if 0 <= source_rank < size:
            data = comm.recv(source=source_rank)
            for (x, y), val in data.items():
                Q[x][y] = val
           
def run_iteration():
    calculate_red()
    send_red()
    recv_messages()
    
    calculate_black()
    send_black()
    recv_messages()


if __name__ == '__main__':
    start = time.time()
    for _ in range(N_ITER):
        run_iteration()
        
    if rank != 0: # send the results to the 0th thread
        comm.send((Q[x_start:x_end], x_start, x_end), dest=0)
    else:         # collect the results from other threads
        for source_rank in range(1, size):
            (data, x_start, x_end) = comm.recv(source=source_rank)
            Q[x_start:x_end] = data

    end = time.time()
    
    # plt.imshow(Q)
    # plt.colorbar()
    # plt.savefig('results.png')
    
    if rank == 0:
        with open(RESULTS_PATH, 'a') as file:
            file.write(f'{N},{size},{N_ITER},{end - start}\n')
