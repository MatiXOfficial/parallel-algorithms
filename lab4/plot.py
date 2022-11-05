from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv('./results.csv', header=None, names=['size', 'CPUs', 'iters', 'time [s]'])

# Acceleration
for size in np.unique(df['size']):
    df_size = df[df['size'] == size]
    
    cpus = df_size['CPUs'].values
    times = df_size['time [s]'].values
    
    accel = times[0] / times
    
    plt.plot(cpus, accel, marker='.', label=f'size={size}')
   
plt.title(f'Acceleration for various problem sizes')
plt.xlabel('CPUs')
plt.ylabel('Acceleration (S(P) = T(1) / T(P))')
plt.legend()
plt.grid()

plt.savefig(f'plots/acceleration.png')
plt.close() 

# Efficiency
for size in np.unique(df['size']):
    df_size = df[df['size'] == size]
    
    cpus = df_size['CPUs'].values
    times = df_size['time [s]'].values
    
    accel = times[0] / times
    eff = accel / cpus
    
    plt.plot(cpus, eff, marker='.', label=f'size={size}')
   
plt.title(f'Efficiency for various problem sizes')
plt.xlabel('CPUs')
plt.ylabel('Efficiency (E(P) = S(P) / P)')
plt.legend()
plt.grid()

plt.savefig(f'plots/efficiency.png')
plt.close() 
