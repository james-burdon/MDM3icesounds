import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.patheffects as path_effects

# --- Output folder ---
output_dir = r"C:\Users\jburd\Desktop\MDM3ice\outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Load dataset ---
df = pd.read_csv(
    r"C:\Users\jburd\Desktop\MDM3ice\cmems_mod_glo_phy-all_my_0.25deg_P1M-m_1760106916111_wide_2020-10-01T00-00-00.csv"
)

# --- Extraction Function ---
def extraction_function(file, no_days, position, tol=0.25):
    file["time"] = pd.to_datetime(file["time"])
    unique_days = file["time"].drop_duplicates()
    random_days = unique_days.sample(min(no_days, len(unique_days)), replace=len(unique_days) < no_days)
    file = file[file["time"].isin(random_days)]

    # Detect columns automatically
    theta_cols = [c for c in file.columns if "thetao" in c.lower()]
    sal_cols = [c for c in file.columns if "so" in c.lower()]

    # Average values
    file["theta"] = file[theta_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if theta_cols else np.nan
    file["salin"] = file[sal_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if sal_cols else np.nan

    # Drop missing essentials
    file = file.dropna(subset=["depth", "lat", "lon", "time"])

    # Bounding box filter
    lat_min, lat_max = position[0] - tol, position[0] + tol
    lon_min, lon_max = position[1] - tol, position[1] + tol
    mask = (file["lat"].between(lat_min, lat_max)) & (file["lon"].between(lon_min, lon_max))

    temps = file.loc[mask, ["time", "theta"]] if "theta" in file else pd.DataFrame(columns=["time", "theta"])
    salinity = file.loc[mask, ["time", "salin"]] if "salin" in file else pd.DataFrame(columns=["time", "salin"])
    depth = file.loc[mask, ["time", "depth"]] if "depth" in file else pd.DataFrame(columns=["time", "depth"])
    return temps, depth, salinity


# --- Extract Data Around Elephant Island ---
position = (-61.133, -55.2)
Ts, Ss, Zs = extraction_function(df, 5, position, tol=0.3)

# --- Ensure numeric + remove NaNs ---
for df_, col in [(Ts, "theta"), (Ss, "salin"), (Zs, "depth")]:
    if col not in df_.columns:
        df_[col] = np.nan
    df_[col] = pd.to_numeric(df_[col], errors="coerce")
    df_.dropna(subset=[col], inplace=True)

# --- Default values if empty ---
T_vals = Ts["theta"].to_numpy() if not Ts.empty else np.array([-2, 0, 2])
S_vals = Ss["salin"].to_numpy() if not Ss.empty else np.array([30, 35, 40])
Z_vals = Zs["depth"].to_numpy() if not Zs.empty else np.array([0, 2500, 5000])

# --- Reduce to 3 representative values ---
T_vals = np.linspace(np.min(T_vals), np.max(T_vals), 3)
S_vals = np.linspace(np.min(S_vals), np.max(S_vals), 3)
Z_vals = np.linspace(np.min(Z_vals), np.max(Z_vals), 3)
colors = ['#e41a1c', '#4daf4a', '#377eb8']  # red, green, blue (colorblind-safe)

# --- Acoustic Equations ---
def transmissionLoss(x, alpha):
    loss = 20 * np.log10(x) + alpha * x * 1e-3
    dLoss = (20 / (x * np.log(10))) + alpha * 1e-3
    return loss, dLoss

def pressure(x, k, w, t, f):
    H = k * x - w * t
    p = np.where(x != 0, (5/x) * np.cos(H), 0)
    alpha = (3.3e-3 +
             (0.11 * (f / 1000)**2) / (1 + (f / 1000)**2) +
             (44 * (f / 1000)**2) / (4100 + (f / 1000)**2) +
             (3e-4) * (f / 1000)**2)
    loss, dLoss = transmissionLoss(x, alpha)
    p = p * 10**((-alpha * x) / 20)
    return p, loss, dLoss


# --- Constants ---
f = 100  # frequency (Hz)
w = 2 * np.pi * f

# =======================
#   PRESSURE PLOTS
# =======================
fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
axes = np.array(axes)

for i, T in enumerate(T_vals):
    for j, Z in enumerate(Z_vals):
        ax = axes[i, j]
        for k, S in enumerate(S_vals):
            c = (1492.9 + 3 * (T - 10)
                 - (6e-3) * (T - 10)**2
                 - (4e-2) * (T - 18)**2
                 + 1.2 * (S - 35)
                 - 1e-2 * (T - 18) * (S - 35)
                 + Z / 61)
            wave_k = w / c
            L = 2 * np.pi / wave_k
            x = np.linspace(1, 200, 1000)
            p, loss, dLoss = pressure(x, wave_k, w, 0, f)
            p = p * np.exp(-0.1 * x)  # add decay for visualization
            ax.plot(x, p, color=colors[k], label=f"S={S:.1f}‰", lw=1.1)
            ax.text(x[-1] - (L / 8), p[-1], f"{c:.1f} m/s",
        color='black', fontsize=8.5, fontweight='bold',
        ha='right', va='center',
        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2.5, foreground='white', alpha=0.8)])


        # Styling per subplot
        ax.set_title(f"T={T:.2f}°C, Z={Z:.0f} m", fontsize=9, pad=3)
        ax.set_xlim(0, 150)
        ax.set_ylim(-0.025, 0.025)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)
        if i == 2:
            ax.set_xlabel("Distance (m)", fontsize=9)
        if j == 0:
            ax.set_ylabel("Pressure (a.u.)", fontsize=9)

# --- Unified legend ---
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='upper center', ncol=3,
           title="Salinity (‰)",
           fontsize=8, title_fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 1.02))

#  Smaller, softer title
plt.suptitle("Acoustic Pressure Decay vs Distance (f = 100 Hz)",
             fontsize=11, fontweight='normal', y=1.04, color='dimgray')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(output_dir, "Pressure_Temp_Depth_Salinity_Publication.pdf"),
            dpi=300, bbox_inches='tight')
plt.show()


# =======================
#   TRANSMISSION LOSS
# =======================
plt.figure(figsize=(7, 5))
plt.plot(x, loss, color='#377eb8', lw=1.2)
plt.title("Transmission Loss vs Distance", fontsize=12, fontweight='bold')
plt.xlabel("Distance (m)", fontsize=10)
plt.ylabel("Loss (dB)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Transmission_Loss_Publication.pdf"), dpi=300)
plt.show()

# =======================
#   DERIVATIVE OF TL
# =======================
# --- Derivative of TL ---
plt.figure(figsize=(8, 5))
plt.plot(x, dLoss, color='m', linewidth=1.5, label="d(TL)/dx (dB/m)")
plt.title("Derivative of Transmission Loss vs Distance", fontsize=12, fontweight='bold')
plt.xlabel("Distance (m)", fontsize=10)
plt.ylabel("dLoss (dB/m)", fontsize=10)

#Auto-scale the y-axis dynamically
plt.ylim(np.min(dLoss) * 0.95, np.max(dLoss) * 1.05)

# Slightly clearer grid & remove top/right spines
plt.grid(True, linestyle='--', alpha=0.6)
for spine in ['top', 'right']:
    plt.gca().spines[spine].set_visible(False)

plt.legend(fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Transmission_Loss_Derivative_Clean.pdf"),
            dpi=300, bbox_inches='tight')
plt.show()
