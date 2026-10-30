
import h5py
import numpy as np
import matplotlib.pyplot as plt

input_file = "optimised/output/variances.h5"
output_file = "plots/sigma_radius.png"

with h5py.File(input_file, "r") as f:
    radii = f["radii"][:]
    variance = f["variance"][:]
    redshifts = f["redshift"][:]
    snapshots = f["snapshots"][:]

sigma = np.sqrt(variance)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))

for i, (snapshot, z) in enumerate(zip(snapshots, redshifts)):
    ax.plot(radii, sigma[i], label=f"z = {z:.2f} (snap {snapshot:03d})")

ax.axvline(8.0, color="black", linestyle="--", linewidth=1.2)
ax.set_xlabel(r"$R\ [\mathrm{Mpc}/h]$")
ax.set_ylabel(r"$\sigma(R)$")
ax.set_title("RMS Density Fluctuation vs Smoothing Radius")
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(radii.min(), radii.max())
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(output_file, dpi=200)
plt.show()

idx = np.argmin(np.abs(radii - 8.0))
print(f"\nSigma at R = {radii[idx]:.3f} Mpc/h")
print("--------------------------------")
for snapshot, z, value in zip(snapshots, redshifts, sigma[:, idx]):
    print(f"Snapshot {snapshot:03d}, z = {z:8.3f}: sigma8 = {value:.6f}")

print(f"\nPlot saved to {output_file}")
