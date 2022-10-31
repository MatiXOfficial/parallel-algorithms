from mpi4py import MPI

import numpy as np
from matplotlib import pyplot as plt


N = 20
N_JOB = 4

N_ITER = 100

# init the table
Q = np.zeros((N + 2, N + 2))

# edges
Q[0] = 0                                              # φ(0, y)
Q[N + 1] = 1                                          # φ(x, N + 1)
Q[:, 0] = np.arange(0, 1.000000001, 1 / (N + 1))      # φ(x, 0)
Q[:, N + 1] = np.arange(0, 1.000000001, 1 / (N + 1))  # φ(N + 1, y) 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def calculate_field(Q, x, y):
    Q[x, y] = (Q[x - 1, y] + Q[x, y + 1] + Q[x + 1, y] + Q[x, y - 1]) / 4

def calculate_red(Q):
    for x in range(1, N + 1, 1):
        for y in range(1 + x % 2, N + 1, 2):
            calculate_field(Q, x, y)

def calculate_black(Q):
    for x in range(1, N + 1, 1):
        for y in range(1 + (x + 1) % 2, N + 1, 2):
            calculate_field(Q, x, y)
            
for _ in range(100):
    calculate_red(Q)
    calculate_black(Q)
    
plt.imshow(Q)
plt.colorbar()
plt.show()
