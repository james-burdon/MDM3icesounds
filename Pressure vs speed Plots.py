import numpy as np
import matplotlib.pyplot as plt

def transmissionLoss(x, alpha):
    """Compute transmission loss (TL) and its derivative."""
    loss = 20 * np.log10(x) + alpha * x * 1e-3  # TL in dB
    dLoss = (20 / (x * np.log(10))) + alpha * 1e-3
    return loss, dLoss

def pressure(x, k, w, t, f):
    """Compute pressure field with attenuation."""
    H = k * x - w * t
    p = np.where(x != 0, (5/x) * np.cos(H), 0)
    # absorption coefficient α (approximation in dB/km)
    alpha = (3.3e-3 +
             (0.11 * (f / 1000)**2) / (1 + (f / 1000)**2) +
             (44 * (f / 1000)**2) / (4100 + (f / 1000)**2) +
             (3e-4) * (f / 1000)**2)
    loss, dLoss = transmissionLoss(x, alpha)
    # apply attenuation (convert dB to amplitude ratio)
    p = p * 10**((-alpha * x) / 20)
    return p, loss, dLoss

# constants
f = 10       # frequency (Hz)
w = 2 * np.pi * f

# parameter arrays
Ts = [-10, -2, 2]      # Temperatures (°C)
Ss = [30, 35, 40]      # Salinities (‰)
Zs = [0, 100, 5000]    # Depths (m)
colors = ['r', 'g', 'b']  # colors for salinities

# --- Pressure plots (3x3 grid) ---
fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
axes = np.array(axes)

for i, T in enumerate(Ts):        # Rows → temperature
    for j, Z in enumerate(Zs):    # Columns → depth
        ax = axes[i, j]

        for k, S in enumerate(Ss):
            # speed of sound including depth
            c = (1492.9 + 3 * (T - 10)
                 - (6e-3) * (T - 10) ** 2
                 - (4e-2) * (T - 18) ** 2
                 + 1.2 * (S - 35)
                 - 1e-2 * (T - 18) * (S - 35)
                 + Z / 61)  # depth contribution

            wave_k = w / c
            L = 2 * np.pi / wave_k
            x = np.linspace(1, 10 * L, 1000)
            t0 = 0
            p, loss, dLoss = pressure(x, wave_k, w, t0, f)

            ax.plot(x, p, color=colors[k], label=f'S={S}‰')
            ax.text(x[-1] - (L / 10), p[-1], f'{c:.1f} m/s',
                    color=colors[k], fontsize=8, ha='right', va='center')

        ax.set_title(f"T={T}°C, Z={Z} m", fontsize=10)
        ax.set_xlim(100)
        ax.set_ylim(-0.025, 0.025)
        ax.grid(True)

        if i == 2:
            ax.set_xlabel("x (m)")
        if j == 0:
            ax.set_ylabel("Pressure (a.u.)")

# add legend and layout improvements
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, title="Salinity (‰)", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle(f"Acoustic Pressure vs Distance (f = {f/1000:.1f} kHz)", fontsize=14)
plt.savefig('Pressure_Temp_Depth_Salinity.pdf')
plt.show()

# --- Transmission Loss Plot ---
plt.figure(figsize=(8, 5))
plt.plot(x, loss, 'b', label="Transmission Loss (dB)")
plt.title("Transmission Loss vs Distance")
plt.xlabel("Distance (m)")
plt.ylabel("Loss (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('Transmission_Loss.pdf')
plt.show()

# --- Derivative of TL Plot ---
plt.figure(figsize=(8, 5))
plt.plot(x, dLoss, 'm', label="d(TL)/dx (dB/m)")
plt.title("Derivative of Transmission Loss vs Distance")
plt.xlim(50)
plt.ylim(0, 0.2)
plt.xlabel("Distance (m)")
plt.ylabel("dLoss (dB/m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('Transmission_Loss_Derivative.pdf')
plt.show()
