# MDM3 Wave Speed Heatmap
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
salinity_range = np.linspace(33.9, 34.1, 200)  # Salinity (PPT)
temperature_range = np.linspace(-2, 2, 200)  # Temperature (°C)
depth = 60  # Depth (m)

# Creates grid for temperature and salinity
S, T = np.meshgrid(salinity_range, temperature_range)

# Wave speed equation
def sound_speed(T, S, Z):
    '''
    Equation taken from Urick, 1983
    '''
    return 1492.9 + 3*(T - 10) - 6e-3*(T - 10)**2 - 4e-2*(T - 18)**2 + 1.5*(S - 35) + Z/61

# Compute sound speed over grid
C = sound_speed(T, S, depth)

# Plotting
plt.figure(figsize=(9, 6))
im = plt.imshow(C, extent=[salinity_range.min(), salinity_range.max(), temperature_range.min(), temperature_range.max()], 
                origin='lower', aspect='auto', cmap='plasma')

plt.colorbar(im, label="Sound Speed (m/s)")
plt.title(f"Sound Speed Variation with Temperature and Salinity at Depth = {depth}m")
plt.xlabel("Salinity (PPT)")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()