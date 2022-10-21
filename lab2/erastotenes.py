#!/usr/bin/env python
# authors: Mateusz Kocot, Łukasz Wroński

import math
import sys

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# find prime numbers in first range from 2 to ceil(sqrt(n)), it is calculated for every process 
def eratostenes_find_B(start, end):
    prime = {i: True for i in range(start, end + 1)}
    p = start
    while p * p <= end:
        if prime[p] == True:
            for i in range(p * p, end + 1, p):
                prime[i] = False

        p += 1
    
    return prime

def eratostenes_find_C(start, end, prime_B):
    prime = {i: True for i in range(start, end + 1)}
    for p, is_prime in prime_B.items():
        if is_prime:
            start_number = start if start % p == 0 else start + p - start % p
            for i in range(max(p * p, start_number), end + 1, p):
                prime[i] = False
    
    return prime


def get_interval(start, end, n, i):
    interval = math.ceil((end + 1 - start) / n)
    return (start + interval * i, min(end, start + interval * i + interval - 1))


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Missing the argument')
        exit()

    # Collect the argument
    N = int(sys.argv[1])

    # Calculate the interval
    sqrt = int(math.sqrt(N))
    start, end = get_interval(sqrt + 1, N, size, rank)
    print(f'Process {rank}. Computing the prime numbers in [{start}, {end}]')

    # Calculate the prime numbers
    prime_B = eratostenes_find_B(2, sqrt)
    prime_C = eratostenes_find_C(start, end, prime_B)
    
    # Handle the result
    if rank != 0: # If non zero, send the result to the process 0
        comm.send(prime_C, dest=0)
    else:         # Collect the results and merge them in the process 0
        prime = prime_B | prime_C
        for i in range(1, size):
            prime_i = comm.recv(source=i)
            prime |= prime_i 
        # print(prime)
