import sys

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(f'results.csv', header=None, names=['CPUs', 'N', 'd', 'time [s]'])

# Acceleration
for d in np.unique(df['d']):
    df_d = df[df['d'] == d]
    
    cpus = df_d['CPUs'].values
    times = df_d['time [s]'].values
    
    accel = times[0] / times
    
    plt.plot(cpus, accel, marker='.', label=f'd={d}')

plt.title(f'Acceleration for different d')
plt.xlabel('CPUs')
plt.ylabel('Acceleration (S(P) = T(1) / T(P))')
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(f'plots/acceleration.png')
plt.close() 

# Efficiency
for d in np.unique(df['d']):
    df_d = df[df['d'] == d]
    
    cpus = df_d['CPUs'].values
    times = df_d['time [s]'].values
    
    accel = times[0] / times
    eff = accel / cpus
    
    plt.plot(cpus, eff, marker='.', label=f'd={d}')

plt.title(f'Efficiency for different d')
plt.xlabel('CPUs')
plt.ylabel('Efficiency (E(P) = S(P) / P)')
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(f'plots/efficiency.png')
plt.close() 

# Serial fraction
for d in np.unique(df['d']):
    df_d = df[df['d'] == d]
    
    cpus = df_d['CPUs'].values
    times = df_d['time [s]'].values
    
    accel = times[0] / times
    # sf = (1 / accel - 1 / cpus) / (1 - 1 / cpus)
    sf = (1 / accel - 1 / cpus)
    sf /= (1 - 1 / cpus)
    
    plt.plot(cpus, sf, marker='.', label=f'd={d}')

plt.title(f'Serial fraction for different d')
plt.xlabel('CPUs')
plt.ylabel('Serial fraction (f(P) = [1/S(p) - 1/p] / [1 - 1/p])')
plt.legend()
plt.grid()
plt.ylim(0, 0.2)
plt.tight_layout()

plt.savefig(f'plots/serial_fraction.png')
plt.close() 
