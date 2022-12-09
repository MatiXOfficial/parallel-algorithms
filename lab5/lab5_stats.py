import numpy as np
import pandas as pd

df = pd.read_csv('lab5.out', skiprows=2, skipfooter=1, names=['rank', 'line', 'time'])

print('##################')
print('##### Task 2 #####')
print('##################')

time_diff = df['time'].max() - df['time'].min()
print(f"Min: {df['time'].min():0.5f} s, Max: {df['time'].max():0.5f} s, Diff: {time_diff:0.5f} s")
print(f"Max time is {100 * df['time'].max() / df['time'].min():0.2f} % of min time")

# Task 3: ani blokowy, ani cykliczny, wygląda na przydzielanie do wolnych procesów.

print()
print('##################')
print('##### Task 4 #####')
print('##################')

print(np.unique(df['rank'], return_counts=True))

print()
print('##################')
print('##### Task 5 #####')
print('##################')

ranks = np.unique(df['rank'])
time_sum = 0
for rank in ranks:
    rank_time = np.sum(df[df['rank'] == rank]['time'])
    print(f'Rank {rank}: {rank_time:0.3f} s')
    time_sum += rank_time

# print(f'Whole program: {3.324939308:0.3f} s')
print(f'Whole program: {6.805106244:0.3f} s')
