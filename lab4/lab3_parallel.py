#!/usr/bin/env python

import time
import sys

from mpi4py import MPI

import numpy as np
from matplotlib import pyplot as plt

# constants
RESULTS_PATH = './results.csv'
N = int(sys.argv[1])
N_ITER = int(sys.argv[2])

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

size_sqrt = int(np.sqrt(size))
n_field = N // size_sqrt

x_start = 1 + (rank % size_sqrt) * n_field
x_end = x_start + n_field
y_start = 1 + (rank // size_sqrt) * n_field
y_end = y_start + n_field
print(f'Calculating: [{x_start}..{x_end}] x [{y_start}..{y_end}]')

# neighbour threads
left_rank = rank - 1 if rank % size_sqrt != 0 else None
up_rank = rank - size_sqrt if rank >= size_sqrt else None
right_rank = rank + 1 if rank % size_sqrt != size_sqrt - 1 else None
down_rank = rank + size_sqrt if rank < size - size_sqrt else None

# single field
def calculate_field(x, y):
    Q[x, y] = (Q[x - 1, y] + Q[x, y + 1] + Q[x + 1, y] + Q[x, y - 1]) / 4

# calculations of the whole table chunk
def calculate_red():
    for x in range(x_start, x_end, 1):
        for y in range(y_start + x % 2, y_end, 2):
            calculate_field(x, y)

def calculate_black():
    for x in range(x_start, x_end, 1):
        for y in range(y_start + (x + 1) % 2, y_end, 2):
            calculate_field(x, y)

# communication
def send_message(rank, data):
    comm.send(data, dest=rank)

def send_red():
    # left
    if left_rank is not None:
        data = {}
        for y in range(y_start + x_start % 2, y_end, 2):
            data[(x_start, y)] = Q[x_start, y]
        send_message(left_rank, data)

    # up
    if up_rank is not None:
        data = {}
        for x in range(x_start + x_start % 2, x_end, 2):
            data[(x, y_start)] = Q[x, y_start]
        send_message(up_rank, data)
    
    # right
    if right_rank is not None:
        data = {}
        for y in range(y_start + (x_start + 1) % 2, y_end, 2):
            data[(x_end - 1, y)] = Q[x_end - 1, y]
        send_message(right_rank, data)
    
    # down
    if down_rank is not None:
        data = {}
        for x in range(x_start + (x_start + 1) % 2, x_end, 2):
            data[(x, y_end - 1)] = Q[x, y_end - 1]
        send_message(down_rank, data)
    
def send_black():
    # left
    if left_rank is not None:
        data = {}
        for y in range(y_start + (x_start + 1) % 2, y_end, 2):
            data[(x_start, y)] = Q[x_start, y]
        send_message(left_rank, data)

    # up
    if up_rank is not None:
        data = {}
        for x in range(x_start + (x_start + 1) % 2, x_end, 2):
            data[(x, y_start)] = Q[x, y_start]
        send_message(up_rank, data)
    
    # right
    if right_rank is not None:
        data = {}
        for y in range(y_start + x_start % 2, y_end, 2):
            data[(x_end - 1, y)] = Q[x_end - 1, y]
        send_message(right_rank, data)
    
    # down
    if down_rank is not None:
        data = {}
        for x in range(x_start + x_start % 2, x_end, 2):
            data[(x, y_end - 1)] = Q[x, y_end - 1]
        send_message(down_rank, data)
        
def recv_messages():
    for source_rank in [left_rank, up_rank, right_rank, down_rank]:
        if source_rank is not None:
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
        data = (Q[x_start:x_end, y_start:y_end], x_start, x_end, y_start, y_end)
        comm.send(data, dest=0)
    else:         # collect the results from other threads
        for source_rank in range(1, size):
            (data, x_start, x_end, y_start, y_end) = comm.recv(source=source_rank)
            Q[x_start:x_end, y_start:y_end] = data

    end = time.time()
    
    with open(RESULTS_PATH, 'a') as file:
        file.write(f'{N},{size},{N_ITER},{end - start}\n')
